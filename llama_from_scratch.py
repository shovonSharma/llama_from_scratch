import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.nn import Module
from torch.optim.lr_scheduler import CosineAnnealingLR


@dataclass
class ModelArgs:
    dim: int = 512              # embedding dimension
    n_layers: int = 8           # number of model decoder blocks
    n_heads: int = 8            # number of heads for queries embedding
    n_kv_heads: int = 4         # number of heads for keys and values embedding
    vocab_size: int = -1        # Length of vocabulary (to be set later)
    multiple_of: int = 256      # Require to calculate dim of feedfoward network
    ffn_dim_multiplier: Optional[float] = None  # Require to calculate dim of feedfoward network
    norm_eps: float = 1e-5      # Default Epsilon value for RMSNorm
    rope_theta: float = 10000.0 # Default theta value for RoPE calculation

    max_batch_size: int = 10    # Max batch size
    max_seq_len: int = 256      # Max sequence length

    epochs: int = 2500          # Total number of training iteration
    log_interval: int = 10      # Number of interval to print the logs and loss values   
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'  # Assign device based on availability
    
    learning_rate: float = 3e-4 # Learning rate
    beta1: float = 0.9          # Adam beta1
    beta2: float = 0.95         # Adam beta2
    weight_decay: float = 0.1   # Weight decay
    grad_clip: float = 1.0      # Gradient clipping


class RMSNorm(Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x):
        return self.weight * self._norm(x)


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, device=None):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.
    """
    # We want to calculate the pairs (cos(m*theta_i), sin(m*theta_i))
    # where theta_i = 10000^(-2i/dim) for i in range(0, dim//2)
    # and m goes from 0 to end-1
    # Ultimately used in RoPE (Rotary Position Embedding)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # [end, dim//2]
    
    # Create complex numbers e^(i * theta) = cos(theta) + i*sin(theta)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # [end, dim//2]
    return freqs_cis


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to the query and key tensors.
    
    Args:
        xq: Query tensor of shape [batch_size, seq_len, n_heads, head_dim]
        xk: Key tensor of shape [batch_size, seq_len, n_kv_heads, head_dim]
        freqs_cis: Complex tensor containing the cos and sin freqs
                   of shape [seq_len, head_dim//2]
    
    Returns:
        Tuple of tensors after applying rotary embeddings
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    # Expand the freqs_cis tensor to match the batch and heads dimensions
    freqs_cis = freqs_cis[:xq.shape[1], :]
    
    # Apply the rotation by complex multiplication
    xq_out = torch.view_as_real(xq_ * freqs_cis.unsqueeze(0).unsqueeze(2)).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis.unsqueeze(0).unsqueeze(2)).flatten(3)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.head_dim = args.dim // args.n_heads
        
        # Each head forms a query vector. There are n_heads query heads.
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        
        # Keys and values only need n_kv_heads, enabling grouped-query attention
        self.wk = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        self.dropout = nn.Dropout(0.1)  # Add dropout for regularization
        
        # For grouped-query attention: if n_heads > n_kv_heads, each kv head serves multiple q heads
        self.kv_repeat = self.n_heads // self.n_kv_heads
        
        # Cache for kv computation - initialized during forward pass as needed
        self.cache_k = None
        self.cache_v = None
    
    def forward(
        self, 
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        seq_start_pos: int = 0
    ):
        batch_size, seq_len, _ = x.shape
        
        # Project to queries, keys, values
        xq = self.wq(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        xk = self.wk(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        
        # Apply rotary embeddings
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)
        
        # Handle KV caching for inference
        if use_cache:
            if self.cache_k is None:
                # Initialize cache if not exists
                max_seq_len = seq_len + seq_start_pos
                self.cache_k = torch.zeros(
                    (batch_size, max_seq_len, self.n_kv_heads, self.head_dim),
                    device=x.device, dtype=x.dtype
                )
                self.cache_v = torch.zeros(
                    (batch_size, max_seq_len, self.n_kv_heads, self.head_dim),
                    device=x.device, dtype=x.dtype
                )
                
            # Update KV cache for current sequence
            self.cache_k[:, seq_start_pos:seq_start_pos+seq_len] = xk
            self.cache_v[:, seq_start_pos:seq_start_pos+seq_len] = xv
            
            # Use cached KVs up to current position
            key = self.cache_k[:, :seq_start_pos+seq_len]
            value = self.cache_v[:, :seq_start_pos+seq_len]
        else:
            key, value = xk, xv
        
        # Handle grouped-query attention if n_heads > n_kv_heads
        # Repeating keys and values if necessary (for grouped-query attention)
        if self.kv_repeat > 1:
            key = key.unsqueeze(2).expand(-1, -1, self.kv_repeat, -1, -1).reshape(
                batch_size, key.shape[1], self.n_heads, self.head_dim
            )
            value = value.unsqueeze(2).expand(-1, -1, self.kv_repeat, -1, -1).reshape(
                batch_size, value.shape[1], self.n_heads, self.head_dim
            )
        
        # Transpose for attention computation
        q = xq.transpose(1, 2)  # (batch_size, n_heads, seq_len, head_dim)
        k = key.transpose(1, 2)  # (batch_size, n_heads, seq_len, head_dim)
        v = value.transpose(1, 2)  # (batch_size, n_heads, seq_len, head_dim)
        
        # Calculate attention scores
        # Scale attention scores by sqrt(head_dim) to prevent softmax saturation
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply causal mask for auto-regressive property
        if mask is not None:
            scores = scores + mask  # masked positions become -inf before softmax
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Compute the weighted sum of values
        output = torch.matmul(attention_weights, v)  # (batch_size, n_heads, seq_len, head_dim)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        # Final projection
        return self.wo(output)


class FeedForward(Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        
        hidden_dim = args.dim * 4
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * args.dim)
        # Round hidden_dim to the nearest multiple of args.multiple_of
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)
        
        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)  # For SwiGLU activation
        self.dropout = nn.Dropout(0.1)  # Add dropout for regularization
        
    def forward(self, x):
        # SwiGLU activation as used in LLaMa
        # SwiGLU(x) = Swish(xW1) * xW3
        swish = F.silu(self.w1(x))
        x_V = self.w3(x)
        
        # Element-wise multiplication
        hidden = swish * x_V
        
        # Final projection
        return self.dropout(self.w2(hidden))


class TransformerBlock(Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        
        self.attention = Attention(args)
        self.feed_forward = FeedForward(args)
        
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        
    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        seq_start_pos: int = 0
    ):
        # Pre-normalization for attention
        h = x + self.attention(
            self.attention_norm(x),
            freqs_cis=freqs_cis,
            mask=mask,
            use_cache=use_cache,
            seq_start_pos=seq_start_pos
        )
        
        # Pre-normalization for feed-forward
        out = h + self.feed_forward(self.ffn_norm(h))
        
        return out


class Llama(Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        
        self.layers = nn.ModuleList([
            TransformerBlock(args) for _ in range(args.n_layers)
        ])
        
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Precompute rotary embedding frequencies
        self.freqs_cis = precompute_freqs_cis(
            args.dim // args.n_heads, 
            args.max_seq_len * 2,  # Overallocate for potential longer sequences
            device=args.device
        )
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self, 
        tokens: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        seq_start_pos: int = 0
    ):
        batch_size, seq_len = tokens.shape
        
        # Get token embeddings
        h = self.tok_embeddings(tokens)
        
        # Get rotary embeddings for current sequence length
        freqs_cis = self.freqs_cis[:seq_len].to(h.device)
        
        # Prepare attention mask (causal, only attend to previous tokens)
        mask = None
        if seq_len > 1:
            mask = torch.full(
                (seq_len, seq_len), float("-inf"), device=tokens.device
            )
            mask = torch.triu(mask, diagonal=1)  # Upper triangular part is -inf
            # Reshape for broadcasting to batch size and heads
            mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        
        # Forward through transformer blocks
        for layer in self.layers:
            h = layer(h, freqs_cis, mask, use_cache, seq_start_pos)
        
        # Final normalization
        h = self.norm(h)
        
        # Language model head
        logits = self.output(h)
        
        # Calculate loss if targets are provided
        loss = None
        if targets is not None:
            # Reshape and calculate cross entropy loss
            batch_size, seq_len, vocab_size = logits.shape
            logits_flat = logits.view(batch_size * seq_len, vocab_size)
            targets_flat = targets.view(batch_size * seq_len)
            loss = F.cross_entropy(logits_flat, targets_flat)
        
        return logits, loss
    
    def reset_cache(self):
        """Reset KV cache for all attention layers"""
        for layer in self.layers:
            layer.attention.cache_k = None
            layer.attention.cache_v = None
    
    def generate(
        self, 
        prompt_tokens: torch.Tensor, 
        max_new_tokens: int, 
        temperature: float = 1.0, 
        top_k: Optional[int] = None
    ):
        """
        Generate new text with an autoregressive sampling loop
        
        Args:
            prompt_tokens: Starting tokens (B, T)
            max_new_tokens: Number of tokens to generate
            temperature: Controls randomness in sampling
            top_k: If set, only sample from the top k most probable tokens
        
        Returns:
            Generated token sequence
        """
        self.eval()  # Set to eval mode
        
        # Move to device and get batch size
        prompt_tokens = prompt_tokens.to(self.args.device)
        batch_size, prompt_length = prompt_tokens.shape
        
        # Reset the KV cache
        self.reset_cache()
        
        # Initialize generated sequence with prompt
        generated_tokens = prompt_tokens.clone()
        
        # First, process the entire prompt and cache keys/values
        with torch.no_grad():
            logits, _ = self(prompt_tokens, use_cache=True, seq_start_pos=0)
            
        # Now generate new tokens one by one
        current_pos = prompt_length
        for _ in range(max_new_tokens):
            # Get the next token predictions using the last token and cached KVs
            with torch.no_grad():
                next_token_logits, _ = self(
                    generated_tokens[:, -1:],  # Only the last token
                    use_cache=True,
                    seq_start_pos=current_pos
                )
                
                # Focus on the last token's prediction
                next_token_logits = next_token_logits[:, -1, :]
                
                # Apply temperature
                if temperature > 0:
                    next_token_logits = next_token_logits / temperature
                
                # Optional top-k sampling
                if top_k is not None:
                    v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                    next_token_logits[next_token_logits < v[:, [-1]]] = float('-inf')
                
                # Convert logits to probabilities
                probs = F.softmax(next_token_logits, dim=-1)
                
                # Sample from the distribution
                next_token = torch.multinomial(probs, num_samples=1)
                
            # Append the sampled token to the sequence
            generated_tokens = torch.cat([generated_tokens, next_token], dim=1)
            current_pos += 1
        
        return generated_tokens


def get_batch(data, batch_size, block_size, device):
    """Get a random batch of data"""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)


def estimate_loss(model, data, eval_iters, batch_size, block_size, device):
    """Estimate loss on train and validation sets"""
    out = {}
    model.eval()
    
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data[split], batch_size, block_size, device)
            with torch.no_grad():
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
        
    model.train()
    return out


def train_model(model, data, args):
    """Train the model and track metrics"""
    device = args.device
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )
    
    # Learning rate scheduler
    scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs,
        eta_min=args.learning_rate / 10
    )
    
    # Track loss metrics
    train_losses = []
    val_losses = []
    eval_interval = args.log_interval
    eval_iters = 10  # Number of batches to estimate loss
    
    # Start training
    print(f"Training on {device}...")
    start_time = time.time()
    
    for epoch in range(args.epochs):
        # Get a batch of data
        x, y = get_batch(data['train'], args.max_batch_size, args.max_seq_len, device)
        
        # Forward pass
        logits, loss = model(x, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        
        # Update weights
        optimizer.step()
        
        # Update learning rate
        scheduler.step()
        
        # Evaluate
        if epoch % eval_interval == 0 or epoch == args.epochs - 1:
            losses = estimate_loss(model, data, eval_iters, args.max_batch_size, args.max_seq_len, device)
            train_losses.append(losses['train'])
            val_losses.append(losses['val'])
            
            # Print progress
            elapsed = time.time() - start_time
            print(f"Epoch {epoch}/{args.epochs} | Time: {elapsed:.2f}s | "
                  f"Train loss: {losses['train']:.4f} | Val loss: {losses['val']:.4f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.6f}")
    
    # Plot training curve
    plt.figure(figsize=(10, 6))
    x_axis = [i * eval_interval for i in range(len(train_losses))]
    plt.plot(x_axis, train_losses, label='Train Loss')
    plt.plot(x_axis, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_loss.png')
    plt.close()
    
    return train_losses, val_losses


def generate_text(model, tokenizer, prompt, max_tokens=100, temperature=0.8, top_k=40):
    """Generate text from a prompt"""
    # Encode the prompt
    prompt_tokens = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(model.args.device)
    
    # Generate tokens
    generated_tokens = model.generate(
        prompt_tokens, 
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k
    )
    
    # Decode the generated tokens
    generated_text = tokenizer.decode(generated_tokens[0].tolist())
    
    return generated_text


# Example usage
if __name__ == "__main__":
    # Setup vocabulary and tokenizer here
    # For example, using a simple character-level tokenizer
    chars = "abcdefghijklmnopqrstuvwxyz0123456789 .,:;!?'\n"
    vocab = {ch: i for i, ch in enumerate(chars)}
    vocab_size = len(vocab)
    
    # Create a simple tokenizer
    class SimpleTokenizer:
        def __init__(self, vocab):
            self.vocab = vocab
            self.inv_vocab = {v: k for k, v in vocab.items()}
            
        def encode(self, text):
            return [self.vocab.get(ch, self.vocab[' ']) for ch in text]
            
        def decode(self, tokens):
            return ''.join(self.inv_vocab.get(token, ' ') for token in tokens)
    
    tokenizer = SimpleTokenizer(vocab)
    
    # Set model arguments
    args = ModelArgs()
    args.vocab_size = vocab_size  # Update vocabulary size
    
    # Create the model
    model = Llama(args).to(args.device)
    
    # Print model size
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params/1e6:.2f}M")
    
    # Load and prepare dataset
    # This is a placeholder - you'd need to provide actual data
    # data = {'train': train_data, 'val': val_data}
    
    # Train the model
    # train_losses, val_losses = train_model(model, data, args)
    
    # Example generation
    # generated_text = generate_text(model, tokenizer, "Once upon a time")
    # print(generated_text)
