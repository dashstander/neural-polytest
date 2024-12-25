from dataclasses import dataclass
import math
import jax
import jax.numpy as jnp
import equinox as eqx



class FieldEmbed(eqx.Module):
    """Learns embeddings for elements of a finite field F_p.
    
    Args:
        p (int): Size of the finite field
        embed_dim (int): Dimension of the embedding vectors
    """
    p: int
    embed_dim: int
    embedding: jnp.ndarray

    def __init__(self, p: int, embed_dim: int, *, key):
        super().__init__()
        self.p = p
        self.embed_dim = embed_dim
        
        # Initialize embeddings with scaled normal distribution
        scale = math.sqrt(1.0 / embed_dim)
        self.embedding = jax.random.normal(key, (p, embed_dim)) * scale
    
    def __call__(self, coeffs):
        """Maps coefficients to their embeddings.
        
        Args:
            coeffs: Array of shape (..., p) containing field elements in [0, p-1]
                   Each row represents coefficients of a polynomial
                   
        Returns:
            Array of shape (..., p, embed_dim) containing embedded coefficients
        """
        return self.embedding[coeffs]


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

    def __init__(self, p: int, model_dim: int, *, key):
        super().__init__()
        # Kaiming initialization
        std = math.sqrt(2.0 / model_dim)
        self.unembed = jax.random.normal(key, (p, p, model_dim)) * std

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
    unembed: PolynomialUnembed

    def __init__(self, p: int, embed_dim: int, poly_dim: int, model_dim: int, *, key):
        super().__init__()
        self.p = p
        self.embed_dim = embed_dim
        self.poly_dim = poly_dim
        self.model_dim = model_dim

        keys = jax.random.split(key, 4)
        self.field_embed = FieldEmbed(p, embed_dim, key=keys[0])
        self.poly_encode = PolyEncoder(p, embed_dim, poly_dim, key=keys[1])
        self.linear0 = eqx.nn.Linear(2 * poly_dim, model_dim, key=keys[2])
        self.unembed = PolynomialUnembed(p, model_dim, key=keys[3])

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


@dataclass
class PolynomialPredictions:
    """Predictions for polynomial coefficients over a finite field.
    
    Attributes:
        logits: Array of shape (batch, degree+1, p) containing raw logits
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
