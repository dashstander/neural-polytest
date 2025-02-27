import torch
from torch import nn
from torch.nn.functional import softmax, relu
import math
from transformer_lens.hook_points import HookedRootModule, HookPoint


# Add hook points to the PyTorch model
class HookedMultiheadAttention(HookedRootModule):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        # Parameters (same as before)
        self.W_Q = nn.Parameter(torch.empty(n_heads, d_model, self.d_head))
        self.W_K = nn.Parameter(torch.empty(n_heads, d_model, self.d_head))
        self.W_V = nn.Parameter(torch.empty(n_heads, d_model, self.d_head))
        self.W_O = nn.Parameter(torch.empty(n_heads, self.d_head, d_model))
        
        # Biases
        self.b_Q = nn.Parameter(torch.zeros(n_heads, self.d_head))
        self.b_K = nn.Parameter(torch.zeros(n_heads, self.d_head))
        self.b_V = nn.Parameter(torch.zeros(n_heads, self.d_head))
        self.b_O = nn.Parameter(torch.zeros(d_model))
        
        # Hook points for debugging/analysis
        self.hook_q = HookPoint()
        self.hook_k = HookPoint()
        self.hook_v = HookPoint()
        self.hook_attn_scores = HookPoint()
        self.hook_attn = HookPoint()
        self.hook_z = HookPoint()
        self.hook_output = HookPoint()
            
    def forward(self, query, key, value):
        """
        x: [batch, seq_len, d_model]
        """
        batch_size, seq_len, _ = query.shape
          
        # Project to queries, keys, and values for each head
        q = self.hook_q(torch.einsum("bsd,hde->bhse", query, self.W_Q) + self.b_Q.unsqueeze(1))
        k = self.hook_k(torch.einsum("bsd,hde->bhse", key, self.W_K) + self.b_K.unsqueeze(1)) 
        v = self.hook_v(torch.einsum("bsd,hde->bhse", value, self.W_V) + self.b_V.unsqueeze(1))
        
        # Calculate attention scores and apply mask if provided
        attn_scores = self.hook_attn_scores(torch.einsum("bhse,bhte->bhst", q, k) / math.sqrt(self.d_head))
        
        
        # Apply softmax to get attention weights
        attn_weights = self.hook_attn(softmax(attn_scores, dim=-1))
        
        # Apply attention weights to values
        z = self.hook_z(torch.einsum("bhst,bhte->bhse", attn_weights, v))
        
        # Project back to d_model dimension
        output = torch.einsum("bhse,hed->bsd", z, self.W_O) + self.b_O
        
        return self.hook_output(output)

# Hooked versions of other layers
class HookedLayerNorm(HookedRootModule):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
        self.hook_norm = HookPoint()
        
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True, unbiased=False) + self.eps
        normalized = (x - mean) / std
        return self.hook_norm(normalized * self.weight + self.bias)


class HookedEncoderLayer(HookedRootModule):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.attention = HookedMultiheadAttention(d_model, n_heads)
        self.ff_linear_up = nn.Linear(d_model, d_ff)
        self.ff_linear_down = nn.Linear(d_ff, d_model)
        self.attention_norm = HookedLayerNorm(d_model)
        self.ff_norm = HookedLayerNorm(d_model)
        
        # Hook points for the MLP
        self.hook_mlp_input = HookPoint()
        self.hook_mlp_pre = HookPoint() 
        self.hook_mlp_act = HookPoint()
        self.hook_mlp_output = HookPoint()
        
        # Hook points for the residual connections
        self.hook_attn_input = HookPoint()
        self.hook_attn_output = HookPoint()
        self.hook_ff_input = HookPoint()
        self.hook_ff_output = HookPoint()
        
    def forward(self, x):
        # Pre-norm architecture
        attn_input = self.hook_attn_input(x)
        normed_x = self.attention_norm(attn_input)
        attention_out = self.attention(normed_x, normed_x, normed_x)
        x = attn_input + self.hook_attn_output(attention_out)
        
        ff_input = self.hook_ff_input(x)
        normed_x = self.ff_norm(ff_input)
        mlp_input = self.hook_mlp_input(normed_x)
        mlp_pre = self.hook_mlp_pre(self.ff_linear_up(mlp_input))
        mlp_act = self.hook_mlp_act(relu(mlp_pre))
        mlp_output = self.hook_mlp_output(self.ff_linear_down(mlp_act))
        x = ff_input + self.hook_ff_output(mlp_output)
        
        return x

class HookedDecoderLayer(HookedRootModule):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.self_attention = HookedMultiheadAttention(d_model, n_heads)
        self.cross_attention = HookedMultiheadAttention(d_model, n_heads)
        self.ff_linear_up = nn.Linear(d_model, d_ff)
        self.ff_linear_down = nn.Linear(d_ff, d_model)
        self.self_attention_norm = HookedLayerNorm(d_model)
        self.cross_attention_norm = HookedLayerNorm(d_model)
        self.ff_norm = HookedLayerNorm(d_model)
        
        # Hook points for the MLP
        self.hook_mlp_input = HookPoint()
        self.hook_mlp_pre = HookPoint()
        self.hook_mlp_act = HookPoint()
        self.hook_mlp_output = HookPoint()
        
        # Hook points for the residual connections
        self.hook_self_attn_input = HookPoint()
        self.hook_self_attn_output = HookPoint()
        self.hook_cross_attn_input = HookPoint()
        self.hook_cross_attn_output = HookPoint()
        self.hook_ff_input = HookPoint()
        self.hook_ff_output = HookPoint()
        
    def forward(self, x, encoder_output):
        # Cross-attention
        cross_attn_input = self.hook_cross_attn_input(x)
        normed_x = self.cross_attention_norm(cross_attn_input)
        cross_attn = self.cross_attention(normed_x, encoder_output, encoder_output)
        x = cross_attn_input + self.hook_cross_attn_output(cross_attn)

        # Self-attention
        self_attn_input = self.hook_self_attn_input(x)
        normed_x = self.self_attention_norm(self_attn_input)
        self_attn = self.self_attention(normed_x, normed_x, normed_x)
        x = self_attn_input + self.hook_self_attn_output(self_attn)
        
        
        # MLP
        ff_input = self.hook_ff_input(x)
        normed_x = self.ff_norm(ff_input)
        mlp_input = self.hook_mlp_input(normed_x)
        mlp_pre = self.hook_mlp_pre(self.ff_linear_up(mlp_input))
        mlp_act = self.hook_mlp_act(relu(mlp_pre))
        mlp_output = self.hook_mlp_output(self.ff_linear_down(mlp_act))
        x = ff_input + self.hook_ff_output(mlp_output)
        
        return x


class HookedPositionalEmbedding(HookedRootModule):
    def __init__(self, seq_len: int, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(seq_len, d_model) * 0.02)
        self.hook_pos_embed = HookPoint()
    
    def forward(self):
        return self.hook_pos_embed(self.weight)


class PolynomialTransformer(HookedRootModule):
    def __init__(self, p: int, d_model: int, n_heads: int, d_ff: int, n_layers: int):
        super().__init__()
        self.p = p
        
        # Token embedding
        self.embedding = nn.Embedding(p + 1, d_model)  # field elements + sep token
        self.hook_embed = HookPoint()
        self.hook_encoder_input = HookPoint()
        self.hook_decoder_input = HookPoint()
        
        # Positional embeddings
        self.pos_embedding_enc = HookedPositionalEmbedding(2*p + 1, d_model)  # left coeffs + sep + right coeffs
        self.pos_embedding_dec = HookedPositionalEmbedding(p, d_model)  # output coefficients
        
        # Encoder and decoder layers
        self.encoder_layers = nn.ModuleList([
            HookedEncoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            HookedDecoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        
        # Final normalization and output
        self.final_norm = HookedLayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, p)
        self.hook_logits = HookPoint()
        
        # Setup all hook points
        self.setup()
    
    def forward(self, left_poly, right_poly):
        batch_size, p = left_poly.shape
        
        # Create encoder input sequence [left_coeffs, sep, right_coeffs]
        sep_token = torch.full((batch_size, 1), self.p, dtype=torch.long, device=left_poly.device)
        encoder_input = torch.cat([left_poly, sep_token, right_poly], dim=1)
        
        # Embed encoder inputs
        encoder_x = self.hook_embed(self.embedding(encoder_input))
        encoder_pos_embed = self.pos_embedding_enc()
        encoder_x = self.hook_encoder_input(encoder_x + encoder_pos_embed)
        
        # Run through encoder layers
        for encoder_layer in self.encoder_layers:
            encoder_x = encoder_layer(encoder_x)
        
        # Create decoder input (learned positional embeddings)
        decoder_pos_embed = self.pos_embedding_dec()
        decoder_x = self.hook_decoder_input(decoder_pos_embed.unsqueeze(0).repeat(batch_size, 1, 1))
        
        # Run through decoder layers
        for decoder_layer in self.decoder_layers:
            decoder_x = decoder_layer(decoder_x, encoder_x)
        
        # Final layer norm and output projection
        decoder_x = self.final_norm(decoder_x)
        logits = self.hook_logits(self.output_proj(decoder_x))
        
        return logits
