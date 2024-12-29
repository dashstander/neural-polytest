from dataclasses import dataclass
import math
import jax
import jax.numpy as jnp
import equinox as eqx


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
    unembed: PolynomialUnembed

    def __init__(self, p: int, embed_dim: int, poly_dim: int, model_dim: int, *, key):
        super().__init__()
        self.p = p
        self.embed_dim = embed_dim
        self.poly_dim = poly_dim
        self.model_dim = model_dim

        keys = jax.random.split(key, 5)
        self.field_embed = FieldEmbed(p, embed_dim, key=keys[0])
        self.poly_encode = PolyEncoder(p, embed_dim, poly_dim, key=keys[1])
        self.linear0 = eqx.nn.Linear(2 * poly_dim, model_dim, key=keys[2])
        self.linear1 = eqx.nn.Linear(model_dim, model_dim, key=keys[3])
        self.unembed = PolynomialUnembed(p, model_dim, key=keys[4])

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
            
        embed_x = self.field_embed(poly_x)
        embed_y = self.field_embed(poly_y)
        
        x_enc = self.poly_encode(embed_x)
        y_enc = self.poly_encode(embed_y)
        xy_encoding = jnp.concatenate([x_enc, y_enc], axis=1)

        activations = jax.vmap(self.linear0)(xy_encoding)
        logits = self.unembed(jax.nn.relu(activations))
        
        return PolynomialPredictions(logits)
    

class TransformerEncoderLayer(eqx.Module):
    """Single transformer encoder layer with self-attention and feedforward."""
    attention: eqx.nn.MultiheadAttention
    ff_linear_up: eqx.nn.Linear
    ff_linear_down: eqx.nn.Linear

    def __init__(self, d_model: int, n_heads: int, d_ff: int, *, key):
        attention_key, ff_key1, ff_key2 = jax.random.split(key, 3)
        self.attention = eqx.nn.MultiheadAttention(
            num_heads=n_heads,
            query_size=d_model,
            key_size=d_model,
            value_size=d_model,
            key=attention_key,
            inference=None,
            dropout_p=None
        )
        self.ff_linear_up = eqx.nn.Linear(d_model, d_ff, key=ff_key1)
        self.ff_linear_down = eqx.nn.Linear(d_ff, d_model, key=ff_key2)

    def __call__(self, x):
        
        # Self attention
        attention_out = self.attention(x, x, x, inference=True)
  
        #x = x + attention_out
        
        # Feedforward block
        ff_out = jax.vmap(self.ff_linear_up)(attention_out)
        ff_out = jax.nn.relu(ff_out)
        ff_out = jax.vmap(self.ff_linear_down)(ff_out)
        #x = x + ff_out
        
        return attention_out


class PolynomialTransformerEncoder(eqx.Module):
    """Transformer encoder for polynomial multiplication."""
    embedding: eqx.nn.Embedding
    pos_embedding: jnp.ndarray
    encoder_layer: TransformerEncoderLayer
    output_proj: eqx.nn.Linear
    p: int
    sequence_weights: eqx.nn.Linear

    def __init__(self, p: int, d_model: int, n_heads: int, d_ff: int, *, key):
        self.p = p
        seq_len = 2*p + 1  # left coeffs + sep + right coeffs
        vocab_size = p + 1  # field elements + sep token
        
        keys = jax.random.split(key, 4)
        
        # Token embedding
        self.embedding = eqx.nn.Embedding(
            num_embeddings=vocab_size,
            embedding_size=d_model,
            key=keys[0]
        )
        
        # Learned positional embedding
        self.pos_embedding = jax.random.normal(keys[1], (seq_len, d_model)) * 0.02
        
        # Transformer layer
        self.encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            key=keys[2]
        )
        self.sequence_weights = eqx.nn.Linear(
            seq_len,
            1,
            key=keys[3]
        )
                
        self.output_proj = eqx.nn.Linear(
            d_model,
            p * p,  # output logits for each coefficient
            key=keys[4]
        )

    def __call__(self, left_poly, right_poly):
        # Create input sequence [left_coeffs, sep, right_coeffs]
        batch_size, p = left_poly.shape
        sep_token = jnp.full((batch_size, 1), self.p)  # p is our sep token index
        x = jnp.concatenate([left_poly, sep_token, right_poly], axis=1)
        
        # Embed tokens and add positional embedding
        x = jax.vmap(jax.vmap(self.embedding))(x)
        x = x + self.pos_embedding
        
        # Apply transformer layer
        x = jax.vmap(self.encoder_layer)(x)
        
        # Learned weighted averaging over sequence length instead of mean
        # x shape is (batch, seq_len, d_model)
        #x = jnp.swapaxes(x, 1, 2)  # -> (batch, d_model, seq_len)
        x = jax.vmap(self.sequence_weights)(x)  # -> (batch, d_model)
        x = x.squeeze(1)  # -> (batch, d_model)
        
        # Project to output logits
        logits = jax.vmap(self.output_proj)(x)
        logits = logits.reshape(batch_size, p, p)
        
        return PolynomialPredictions(logits)


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
