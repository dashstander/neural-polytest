from dataclasses import dataclass

import math
from typing import Optional
from functools import partial

import jax
import jax.numpy as jnp

import equinox as eqx



def dot_product_attention_weights(
    query,
    key,
    mask = None,
):
    query = query / math.sqrt(query.shape[-1])
    logits = jnp.einsum("sd,Sd->sS", query, key)
    if mask is not None:
        if mask.shape != logits.shape:
            raise ValueError(
                f"mask must have shape (query_seq_length, "
                f"kv_seq_length)=({query.shape[0]}, "
                f"{key.shape[0]}). Got {mask.shape}."
            )
        logits = jnp.where(mask, logits, jnp.finfo(logits.dtype).min)

    with jax.numpy_dtype_promotion("standard"):
        dtype = jnp.result_type(logits.dtype, jnp.float32)
    weights = jax.nn.softmax(logits.astype(dtype)).astype(logits.dtype)
    return weights


def dot_product_attention(query, key_, value, mask):
    weights = dot_product_attention_weights(query, key_, mask)
    attn = jnp.einsum("sS,Sd->sd", weights, value)
    return attn


class LayerNorm(eqx.Module):
    norm: eqx.nn.LayerNorm

def __init__(self, d_model):
    self.norm = eqx.nn.LayerNorm(d_model)

def __call__(self, x):
    return jax.vmap(jax.vmap(self.norm))(x)



class MultiheadAttention(eqx.Module):
    query_proj: eqx.nn.Linear
    key_proj: eqx.nn.Linear
    value_proj: eqx.nn.Linear
    output_proj: eqx.nn.Linear

    num_heads: int = eqx.field(static=True)
    query_size: int = eqx.field(static=True)
    key_size: int = eqx.field(static=True)
    value_size: int = eqx.field(static=True)
    output_size: int = eqx.field(static=True)
    qk_size: int = eqx.field(static=True)
    vo_size: int = eqx.field(static=True)
    use_query_bias: bool = eqx.field(static=True)
    use_key_bias: bool = eqx.field(static=True)
    use_value_bias: bool = eqx.field(static=True)
    use_output_bias: bool = eqx.field(static=True)

    def __init__(
        self,
        num_heads: int,
        query_size: int,
        key_size: Optional[int] = None,
        value_size: Optional[int] = None,
        output_size: Optional[int] = None,
        qk_size: Optional[int] = None,
        vo_size: Optional[int] = None,
        use_query_bias: bool = False,
        use_key_bias: bool = False,
        use_value_bias: bool = False,
        use_output_bias: bool = False,
        dtype=None,
        *,
        key,
    ):
        dtype = jnp.float32 if dtype is None else dtype  # Default to float32 if not specified
        qkey, kkey, vkey, okey = jax.random.split(key, 4)

        if key_size is None:
            key_size = query_size
        if value_size is None:
            value_size = query_size
        if qk_size is None:
            qk_size = query_size // num_heads
        if vo_size is None:
            vo_size = query_size // num_heads
        if output_size is None:
            output_size = query_size

        self.query_proj = eqx.nn.Linear(
            query_size,
            num_heads * qk_size,
            use_bias=use_query_bias,
            dtype=dtype,
            key=qkey,
        )
        self.key_proj = eqx.nn.Linear(
            key_size, num_heads * qk_size, use_bias=use_key_bias, dtype=dtype, key=kkey
        )
        self.value_proj = eqx.nn.Linear(
            value_size,
            num_heads * vo_size,
            use_bias=use_value_bias,
            dtype=dtype,
            key=vkey,
        )
        self.output_proj = eqx.nn.Linear(
            num_heads * vo_size,
            output_size,
            use_bias=use_output_bias,
            dtype=dtype,
            key=okey,
        )

        self.num_heads = num_heads
        self.query_size = query_size
        self.key_size = key_size
        self.value_size = value_size
        self.output_size = output_size
        self.qk_size = qk_size
        self.vo_size = vo_size
        self.use_query_bias = use_query_bias
        self.use_key_bias = use_key_bias
        self.use_value_bias = use_value_bias
        self.use_output_bias = use_output_bias

    def __call__(
        self,
        query,
        key_,
        value,
        mask = None,
        process_heads = None,
    ):
        query_seq_length, _ = query.shape
        kv_seq_length, _ = key_.shape
        kv_seq_length2, _ = value.shape
        if kv_seq_length != kv_seq_length2:
            raise ValueError("key and value must both be sequences of equal length.")

        query_heads = self._project(self.query_proj, query)
        key_heads = self._project(self.key_proj, key_)
        value_heads = self._project(self.value_proj, value)

        if process_heads is not None:
            q_shape, k_shape, v_shape = (
                query_heads.shape,
                key_heads.shape,
                value_heads.shape,
            )
            query_heads, key_heads, value_heads = process_heads(
                query_heads, key_heads, value_heads
            )

            if (
                query_heads.shape != q_shape
                or key_heads.shape != k_shape
                or value_heads.shape != v_shape
            ):
                raise ValueError(
                    "process_heads must not change the shape of the heads."
                )

        if mask is not None and mask.ndim == 3:
            attn = jax.vmap(dot_product_attention, in_axes=1, out_axes=1)(
                query_heads, key_heads, value_heads, mask=mask
            )
        else:
            attn = jax.vmap(partial(dot_product_attention, mask=mask), in_axes=1, out_axes=1)(
                query_heads, key_heads, value_heads
            )
        attn = attn.reshape(query_seq_length, -1)

        return jax.vmap(self.output_proj)(attn)

    def _project(self, proj, x):
        seq_length, _ = x.shape
        projection = jax.vmap(proj)(x)
        return projection.reshape(seq_length, self.num_heads, -1)
    


class FieldEmbed(eqx.Module):
    """Learns embeddings for elements of a finite field F_p."""
    p: int
    embed_dim: int
    embedding: eqx.nn.Embedding

    def __init__(self, p: int, embed_dim: int, *, key):
        super().__init__()
        self.p = p
        self.embed_dim = embed_dim
        
        self.embedding = eqx.nn.Embedding(
            num_embeddings=p,
            embedding_size=embed_dim,
            key=key
        )
    
    def __call__(self, coeffs):
        """Maps coefficients to their embeddings."""
        return jax.vmap(jax.vmap(self.embedding))(coeffs)


class PolyEncoder(eqx.Module):
    """Encodes a polynomial into a fixed-dimensional vector representation.
    
    Args:
        p (int): Size of the finite field
        in_dim (int): Dimension of input coefficient embeddings
        poly_dim (int): Dimension of output polynomial encoding
    """
    p: int
    in_dim: int
    poly_dim: int
    proj_up: eqx.nn.Linear
    proj_mix: eqx.nn.Linear

    def __init__(self, p: int, in_dim: int, poly_dim: int, *, key):
        super().__init__()
        self.p = p
        self.in_dim = in_dim
        self.poly_dim = poly_dim

        k1, k2 = jax.random.split(key)
        self.proj_up = eqx.nn.Linear(in_dim, poly_dim, use_bias=False, key=k1)
        self.proj_mix = eqx.nn.Linear(p, 1, use_bias=False, key=k2)

    def __call__(self, coeffs):
        """Encodes embedded coefficients into a single vector.
        
        Args:
            coeffs: Array of shape (batch, p, in_dim) of embedded coefficients
            
        Returns:
            Array of shape (batch, poly_dim) encoding the full polynomial
        """
        x = jax.vmap(self.proj_up)(coeffs.transpose(0, 2, 1)).transpose(0, 2, 1)
        return jnp.squeeze(jax.vmap(self.proj_mix)(x), axis=1)


class PolynomialUnembed(eqx.Module):
    """Projects model outputs back to polynomial coefficients using batched matrix multiplication.
    
    Args:
        p (int): Size of the finite field
        model_dim (int): Dimension of model's internal representation
        
    The module learns p matrices of shape (p, model_dim), one for each coefficient
    of each output position. This is stored in the (p, p, model_dim) tensor `unembed`. 
    The slice from `unembed[i, :, :]` maps the activations to logits predicting the value (in [0, ..., p-1])
    of the ith degree coefficient (e.g. `unembed[2, :, :] @ activations` gets the logits for the x^2 term).
    Each unembedding projection is applied in parallel using vmap.
    """
    unembed: jnp.ndarray

    def __init__(self, p: int, input_dim: int, *, key):
        super().__init__()
        # Kaiming initialization
        std = math.sqrt(2.0 / input_dim)
        self.unembed = jax.random.normal(key, (p, p, input_dim)) * std

    def __call__(self, x):
        """Projects model representations to polynomial coefficients.
        
        Args:
            x: Array of shape (batch, model_dim) containing model outputs
            
        Returns:
            Array of shape (batch, p, p) containing predicted coefficients
        """
        return jax.vmap(lambda v: jnp.einsum('ijk,k->ij', self.unembed, v))(x)


class PolynomialMultiplicationMLP(eqx.Module):
    """Neural network for multiplying polynomials over finite fields.
    
    Args:
        p (int): Size of the finite field
        embed_dim (int): Dimension for embedding field elements
        poly_dim (int): Dimension for encoding full polynomials
        model_dim (int): Internal dimension for computation
    """
    p: int
    embed_dim: int
    poly_dim: int
    model_dim: int
    field_embed: FieldEmbed
    poly_encode: PolyEncoder
    linear0: eqx.nn.Linear
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear
    unembed: eqx.nn.Linear

    def __init__(self, p: int, embed_dim: int, poly_dim: int, model_dim: int, *, key):
        super().__init__()
        self.p = p
        self.embed_dim = embed_dim
        self.poly_dim = poly_dim
        self.model_dim = model_dim

        keys = jax.random.split(key, 6)
        self.field_embed = FieldEmbed(p, embed_dim, key=keys[0])
        self.poly_encode = PolyEncoder(p, embed_dim, poly_dim, key=keys[1])
        self.linear0 = eqx.nn.Linear(2 * poly_dim, model_dim, key=keys[2])
        self.linear1 = eqx.nn.Linear(model_dim, model_dim, key=keys[3])
        self.linear2 = eqx.nn.Linear(model_dim, model_dim, key=keys[4])
        self.unembed = eqx.nn.Linear(model_dim, p**2, key=keys[5])

    def __call__(self, poly_x, poly_y):
        """Computes product of two polynomials.
        
        Args:
            poly_x: Array of shape (batch, p) containing coefficients of first polynomial
            poly_y: Array of shape (batch, p) containing coefficients of second polynomial
            
        Returns:
            PolynomialPredictions containing logits for product coefficients
        """
        # Handle single polynomial case
        if poly_x.ndim == 1:
            poly_x = poly_x[None, :]
            poly_y = poly_y[None, :]

        _, p = poly_x.shape
            
        embed_x = self.field_embed(poly_x)
        embed_y = self.field_embed(poly_y)
        
        x_enc = self.poly_encode(embed_x)
        y_enc = self.poly_encode(embed_y)
        xy_encoding = jnp.concatenate([x_enc, y_enc], axis=1)

        activations = jax.vmap(self.linear0)(xy_encoding)
        activations = jax.vmap(self.linear1)(jax.nn.relu(activations))
        activations = jax.vmap(self.linear2)(jax.nn.relu(activations))
        logits = jax.vmap(self.unembed)(jax.nn.relu(jax.nn.relu(activations)))
        
        return PolynomialPredictions(logits.reshape(-1, p, p))


class EncoderLayer(eqx.Module):
    attention: MultiheadAttention
    ff_linear_up: eqx.nn.Linear
    ff_linear_down: eqx.nn.Linear
    attention_norm: LayerNorm
    ff_norm: LayerNorm

    def __init__(self, d_model: int, n_heads: int, d_ff: int, *, key):
        attention_key, ff_key1, ff_key2 = jax.random.split(key, 3)
        self.attention = MultiheadAttention(
            num_heads=n_heads,
            query_size=d_model,
            key_size=d_model,
            value_size=d_model,
            key=attention_key,
        )
        self.ff_linear_up = eqx.nn.Linear(d_model, d_ff, key=ff_key1)
        self.ff_linear_down = eqx.nn.Linear(d_ff, d_model, key=ff_key2)
        
        # Add LayerNorm layers
        self.attention_norm = LayerNorm(d_model)
        self.ff_norm = LayerNorm(d_model)

    def __call__(self, x):
        # Pre-norm architecture (more stable)
        normed_x = jax.vmap(self.attention_norm)(x)
        attention_out = self.attention(normed_x, normed_x, normed_x)
        x = x + attention_out
        
        normed_x = jax.vmap(self.ff_norm)(x)
        ff_out = jax.vmap(self.ff_linear_up)(normed_x)
        ff_out = jax.nn.relu(ff_out)
        ff_out = jax.vmap(self.ff_linear_down)(ff_out)
        x = x + ff_out
        return x


class DecoderLayer(eqx.Module):
    self_attention: MultiheadAttention
    cross_attention: MultiheadAttention
    ff_linear_up: eqx.nn.Linear
    ff_linear_down: eqx.nn.Linear
    self_attention_norm: LayerNorm
    cross_attention_norm: LayerNorm
    ff_norm: LayerNorm

    def __init__(self, d_model: int, n_heads: int, d_ff: int, *, key):
        keys = jax.random.split(key, 4)
        self.self_attention = MultiheadAttention(
            num_heads=n_heads,
            query_size=d_model,
            key_size=d_model,
            value_size=d_model,
            key=keys[0],
        )
        self.cross_attention = MultiheadAttention(
            num_heads=n_heads,
            query_size=d_model,
            key_size=d_model,
            value_size=d_model,
            key=keys[1],
        )
        self.ff_linear_up = eqx.nn.Linear(d_model, d_ff, key=keys[2])
        self.ff_linear_down = eqx.nn.Linear(d_ff, d_model, key=keys[3])
        
        self.self_attention_norm = LayerNorm(d_model)
        self.cross_attention_norm = LayerNorm(d_model)
        self.ff_norm = LayerNorm(d_model)

    def __call__(self, x, encoder_output):
        # Pre-norm architecture
        # Self attention
        normed_x = jax.vmap(self.self_attention_norm)(x)
        self_attn = self.self_attention(normed_x, normed_x, normed_x)
        x = x + self_attn

        # Cross attention to encoder outputs
        normed_x = jax.vmap(self.cross_attention_norm)(x)
        cross_attn = self.cross_attention(normed_x, encoder_output, encoder_output)
        x = x + cross_attn
        
        # Feedforward
        normed_x = jax.vmap(self.ff_norm)(x)
        ff_out = jax.vmap(self.ff_linear_up)(normed_x)
        ff_out = jax.nn.relu(ff_out)
        ff_out = jax.vmap(self.ff_linear_down)(ff_out)
        x = x + ff_out
        return x


class PolynomialTransformerEncoderDecoder(eqx.Module):
    embedding: eqx.nn.Embedding
    pos_embedding_enc: jnp.ndarray
    pos_embedding_dec: jnp.ndarray
    encoder_layers: list[EncoderLayer]
    decoder_layers: list[DecoderLayer]
    output_proj: eqx.nn.Linear
    final_norm: LayerNorm  # Added final layer norm
    p: int

    def __init__(self, p: int, d_model: int, n_heads: int, d_ff: int, n_layers: int, *, key):
        self.p = p
        encoder_seq_len = 2*p + 1  # left coeffs + sep + right coeffs
        decoder_seq_len = p  # output coefficients
        vocab_size = p + 1  # field elements + sep token
        
        keys = jax.random.split(key, 4 + 2*n_layers)
        
        # Token embedding
        self.embedding = eqx.nn.Embedding(
            num_embeddings=vocab_size,
            embedding_size=d_model,
            key=keys[0]
        )
        
        # Positional embeddings
        self.pos_embedding_enc = jax.random.normal(keys[1], (encoder_seq_len, d_model)) * 0.02
        self.pos_embedding_dec = jax.random.normal(keys[2], (decoder_seq_len, d_model)) * 0.02
        
        # Multiple encoder and decoder layers
        self.encoder_layers = [
            EncoderLayer(d_model, n_heads, d_ff, key=keys[i+3])
            for i in range(n_layers)
        ]
        self.decoder_layers = [
            DecoderLayer(d_model, n_heads, d_ff, key=keys[i+3+n_layers])
            for i in range(n_layers)
        ]
        
        # Output projection and final normalization
        self.output_proj = eqx.nn.Linear(d_model, p, key=keys[-1])
        self.final_norm = LayerNorm(d_model)

    def __call__(self, left_poly, right_poly):
        batch_size, p = left_poly.shape
        
        # Create encoder input sequence [left_coeffs, sep, right_coeffs]
        sep_token = jnp.full((batch_size, 1), self.p)
        encoder_input = jnp.concatenate([left_poly, sep_token, right_poly], axis=1)
        
        # Embed encoder inputs
        encoder_x = jax.vmap(jax.vmap(self.embedding))(encoder_input)
        encoder_x = encoder_x + self.pos_embedding_enc
        
        # Run through encoder layers
        for encoder_layer in self.encoder_layers:
            encoder_x = jax.vmap(encoder_layer)(encoder_x)
        
        # Create decoder input (learned start tokens)
        decoder_x = self.pos_embedding_dec[None].repeat(batch_size, axis=0)
        
        # Run through decoder layers
        for decoder_layer in self.decoder_layers:
            decoder_x = jax.vmap(decoder_layer)(decoder_x, encoder_x)
        
        # Final layer norm before output projection
        decoder_x = jax.vmap(self.final_norm)(decoder_x)
        
        # Project to logits
        logits = jax.vmap(self.output_proj)(jax.lax.transpose(decoder_x, (0, 2, 1)))
        
        return logits
    

@dataclass
class PolynomialPredictions:
    """Predictions for polynomial coefficients over a finite field.
    
    Attributes:
        logits: Array of shape (batch, p, p) containing raw logits
               logits[b, i, j] is the logit for coefficient i being value j
               in polynomial b of the batch
    """
    logits: jnp.ndarray
    
    @property
    def batch_size(self) -> int:
        return self.logits.shape[0]
    
    @property
    def max_degree(self) -> int:
        return self.logits.shape[1] - 1
    
    @property
    def field_size(self) -> int:
        return self.logits.shape[2]
    
    def get_coefficient_logits(self, degree: int) -> jnp.ndarray:
        """Get logits for a specific degree's coefficient across batch."""
        return self.logits[:, degree, :]
        
    def get_polynomial_logits(self, batch_idx: int) -> jnp.ndarray:
        """Get all coefficient logits for a specific polynomial in batch."""
        return self.logits[batch_idx]

    def get_predictions(self) -> jnp.ndarray:
        """Get most likely coefficient values.
        
        Returns:
            Array of shape (batch, degree+1) containing predicted coefficients
        """
        return jnp.argmax(self.logits, axis=-1)
