import torch
from torch import nn
from torch.nn.functional import softmax, relu
import math
import numpy as np
import jax
import jax.numpy as jnp


from neural_polytest.torch.transformer import PolynomialTransformer



def convert_from_equinox(jax_model):
    """
    Convert a JAX Equinox polynomial transformer model to PyTorch.
    
    Args:
        jax_model: The loaded JAX Equinox model
        
    Returns:
        A PyTorch version of the model with converted weights
    """
    import jax.numpy as jnp
    import numpy as np
    
    # Extract model parameters
    p = jax_model.p
    d_model = jax_model.embedding.weight.shape[1]
    n_heads = jax_model.encoder_layers[0].attention.num_heads
    d_ff = jax_model.encoder_layers[0].ff_linear_up.weight.shape[1]
    n_layers = len(jax_model.encoder_layers)
    
    # Create PyTorch model with the same architecture
    torch_model = PolynomialTransformer(
        p=p,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        n_layers=n_layers
    )
    
    # Helper function to convert JAX arrays to PyTorch tensors
    def convert_to_tensor(x):
        return torch.tensor(np.array(x))
    
    # Convert and load embedding weights
    torch_model.embedding.weight.data = convert_to_tensor(jax_model.embedding.weight)
    
    # Convert positional embeddings
    torch_model.pos_embedding_enc.weight.data = convert_to_tensor(jax_model.pos_embedding_enc.weight)
    torch_model.pos_embedding_dec.weight.data = convert_to_tensor(jax_model.pos_embedding_dec.weight)
    
    # Convert encoder layers
    for i, (jax_layer, torch_layer) in enumerate(zip(jax_model.encoder_layers, torch_model.encoder_layers)):
        # Convert attention weights
        for head_idx in range(n_heads):
            # In JAX: query_proj is [d_model, n_heads * d_head]
            # In PyTorch: W_Q is [n_heads, d_model, d_head]
            d_head = d_model // n_heads
            
            # Extract weights for each head from JAX model and reshape for PyTorch
            jax_W_Q = jax_layer.attention.query_proj.weight[:, head_idx*d_head:(head_idx+1)*d_head]
            jax_W_K = jax_layer.attention.key_proj.weight[:, head_idx*d_head:(head_idx+1)*d_head]
            jax_W_V = jax_layer.attention.value_proj.weight[:, head_idx*d_head:(head_idx+1)*d_head]
            
            torch_layer.attention.W_Q.data[head_idx] = convert_to_tensor(jax_W_Q)
            torch_layer.attention.W_K.data[head_idx] = convert_to_tensor(jax_W_K)
            torch_layer.attention.W_V.data[head_idx] = convert_to_tensor(jax_W_V)
            
            # Extract biases for each head
            if jax_layer.attention.use_query_bias:
                jax_b_Q = jax_layer.attention.query_proj.bias[head_idx*d_head:(head_idx+1)*d_head]
                torch_layer.attention.b_Q.data[head_idx] = convert_to_tensor(jax_b_Q)
            if jax_layer.attention.use_key_bias:
                jax_b_K = jax_layer.attention.key_proj.bias[head_idx*d_head:(head_idx+1)*d_head]
                torch_layer.attention.b_K.data[head_idx] = convert_to_tensor(jax_b_K)
            if jax_layer.attention.use_value_bias:
                jax_b_V = jax_layer.attention.value_proj.bias[head_idx*d_head:(head_idx+1)*d_head]
                torch_layer.attention.b_V.data[head_idx] = convert_to_tensor(jax_b_V)
            
            # For W_O, we need to transpose from JAX [d_head, d_model] to PyTorch [d_head, d_model]
            jax_W_O_head = jax_layer.attention.output_proj.weight.reshape(n_heads, d_head, d_model)[head_idx]
            torch_layer.attention.W_O.data[head_idx] = convert_to_tensor(jax_W_O_head)
        
        # Convert output bias
        if jax_layer.attention.use_output_bias:
            torch_layer.attention.b_O.data = convert_to_tensor(jax_layer.attention.output_proj.bias)
        
        # Convert feedforward layers
        torch_layer.ff_linear_up.weight.data = convert_to_tensor(jax_layer.ff_linear_up.weight)
        torch_layer.ff_linear_up.bias.data = convert_to_tensor(jax_layer.ff_linear_up.bias)
        torch_layer.ff_linear_down.weight.data = convert_to_tensor(jax_layer.ff_linear_down.weight)
        torch_layer.ff_linear_down.bias.data = convert_to_tensor(jax_layer.ff_linear_down.bias)
        
        # Convert layer norms
        torch_layer.attention_norm.weight.data = convert_to_tensor(jax_layer.attention_norm.norm.weight)
        torch_layer.attention_norm.bias.data = convert_to_tensor(jax_layer.attention_norm.norm.bias)
        torch_layer.ff_norm.weight.data = convert_to_tensor(jax_layer.ff_norm.norm.weight)
        torch_layer.ff_norm.bias.data = convert_to_tensor(jax_layer.ff_norm.norm.bias)
    
    # Convert decoder layers
    for i, (jax_layer, torch_layer) in enumerate(zip(jax_model.decoder_layers, torch_model.decoder_layers)):
        # Convert self-attention
        for head_idx in range(n_heads):
            d_head = d_model // n_heads
            
            # Self-attention
            jax_W_Q = jax_layer.self_attention.query_proj.weight[:, head_idx*d_head:(head_idx+1)*d_head]
            jax_W_K = jax_layer.self_attention.key_proj.weight[:, head_idx*d_head:(head_idx+1)*d_head]
            jax_W_V = jax_layer.self_attention.value_proj.weight[:, head_idx*d_head:(head_idx+1)*d_head]
            
            torch_layer.self_attention.W_Q.data[head_idx] = convert_to_tensor(jax_W_Q)
            torch_layer.self_attention.W_K.data[head_idx] = convert_to_tensor(jax_W_K)
            torch_layer.self_attention.W_V.data[head_idx] = convert_to_tensor(jax_W_V)
            
            # Extract biases for each head
            if jax_layer.self_attention.use_query_bias:
                jax_b_Q = jax_layer.self_attention.query_proj.bias[head_idx*d_head:(head_idx+1)*d_head]
                torch_layer.self_attention.b_Q.data[head_idx] = convert_to_tensor(jax_b_Q)
            if jax_layer.self_attention.use_key_bias:
                jax_b_K = jax_layer.self_attention.key_proj.bias[head_idx*d_head:(head_idx+1)*d_head]
                torch_layer.self_attention.b_K.data[head_idx] = convert_to_tensor(jax_b_K)
            if jax_layer.self_attention.use_value_bias:
                jax_b_V = jax_layer.self_attention.value_proj.bias[head_idx*d_head:(head_idx+1)*d_head]
                torch_layer.self_attention.b_V.data[head_idx] = convert_to_tensor(jax_b_V)
            
            jax_W_O_head = jax_layer.self_attention.output_proj.weight.reshape(n_heads, d_head, d_model)[head_idx]
            torch_layer.self_attention.W_O.data[head_idx] = convert_to_tensor(jax_W_O_head)
            
            # Cross-attention
            jax_W_Q = jax_layer.cross_attention.query_proj.weight[:, head_idx*d_head:(head_idx+1)*d_head]
            jax_W_K = jax_layer.cross_attention.key_proj.weight[:, head_idx*d_head:(head_idx+1)*d_head]
            jax_W_V = jax_layer.cross_attention.value_proj.weight[:, head_idx*d_head:(head_idx+1)*d_head]
            
            torch_layer.cross_attention.W_Q.data[head_idx] = convert_to_tensor(jax_W_Q)
            torch_layer.cross_attention.W_K.data[head_idx] = convert_to_tensor(jax_W_K)
            torch_layer.cross_attention.W_V.data[head_idx] = convert_to_tensor(jax_W_V)
            
            if jax_layer.cross_attention.use_query_bias:
                jax_b_Q = jax_layer.cross_attention.query_proj.bias[head_idx*d_head:(head_idx+1)*d_head]
                torch_layer.cross_attention.b_Q.data[head_idx] = convert_to_tensor(jax_b_Q)
            if jax_layer.cross_attention.use_key_bias:
                jax_b_K = jax_layer.cross_attention.key_proj.bias[head_idx*d_head:(head_idx+1)*d_head]
                torch_layer.cross_attention.b_K.data[head_idx] = convert_to_tensor(jax_b_K)
            if jax_layer.cross_attention.use_value_bias:
                jax_b_V = jax_layer.cross_attention.value_proj.bias[head_idx*d_head:(head_idx+1)*d_head]
                torch_layer.cross_attention.b_V.data[head_idx] = convert_to_tensor(jax_b_V)
            
            jax_W_O_head = jax_layer.cross_attention.output_proj.weight.reshape(n_heads, d_head, d_model)[head_idx]
            torch_layer.cross_attention.W_O.data[head_idx] = convert_to_tensor(jax_W_O_head)
        
        # Convert output biases
        if jax_layer.self_attention.use_output_bias:
            torch_layer.self_attention.b_O.data = convert_to_tensor(jax_layer.self_attention.output_proj.bias)
        if jax_layer.cross_attention.use_output_bias:
            torch_layer.cross_attention.b_O.data = convert_to_tensor(jax_layer.cross_attention.output_proj.bias)
        
        # Convert feedforward layers
        torch_layer.ff_linear_up.weight.data = convert_to_tensor(jax_layer.ff_linear_up.weight)
        torch_layer.ff_linear_up.bias.data = convert_to_tensor(jax_layer.ff_linear_up.bias)
        torch_layer.ff_linear_down.weight.data = convert_to_tensor(jax_layer.ff_linear_down.weight)
        torch_layer.ff_linear_down.bias.data = convert_to_tensor(jax_layer.ff_linear_down.bias)
        
        # Convert layer norms
        torch_layer.self_attention_norm.weight.data = convert_to_tensor(jax_layer.self_attention_norm.norm.weight)
        torch_layer.self_attention_norm.bias.data = convert_to_tensor(jax_layer.self_attention_norm.norm.bias)
        torch_layer.cross_attention_norm.weight.data = convert_to_tensor(jax_layer.cross_attention_norm.norm.weight)
        torch_layer.cross_attention_norm.bias.data = convert_to_tensor(jax_layer.cross_attention_norm.norm.bias)
        torch_layer.ff_norm.weight.data = convert_to_tensor(jax_layer.ff_norm.norm.weight)
        torch_layer.ff_norm.bias.data = convert_to_tensor(jax_layer.ff_norm.norm.bias)
    
    # Convert final layer norm and output projection
    torch_model.final_norm.weight.data = convert_to_tensor(jax_model.final_norm.norm.weight)
    torch_model.final_norm.bias.data = convert_to_tensor(jax_model.final_norm.norm.bias)
    torch_model.output_proj.weight.data = convert_to_tensor(jax_model.output_proj.weight)
    torch_model.output_proj.bias.data = convert_to_tensor(jax_model.output_proj.bias)
    
    return torch_model
