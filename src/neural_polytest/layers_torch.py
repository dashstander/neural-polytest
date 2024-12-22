import math
import torch
from torch import nn
from torch.nn.functional import relu


class FieldEmbed(nn.Module):
    """Learns embeddings for elements of a finite field F_p.
    
    Args:
        p (int): Size of the finite field
        embed_dim (int): Dimension of the embedding vectors
    """
    def __init__(self, p: int, embed_dim: int):
        super().__init__()
        self.p = p
        self.embed_dim = embed_dim
        # Max norm prevents embeddings from exploding while training
        self.field = nn.Embedding(num_embeddings=p, embedding_dim=embed_dim, max_norm=math.sqrt(embed_dim))
    
    def forward(self, coeffs):
        """Maps coefficients to their embeddings.
        
        Args:
            coeffs: Tensor of shape (..., p) containing field elements in [0, p-1]
                   Each row represents coefficients of a polynomial
                   
        Returns:
            Tensor of shape (..., p, embed_dim) containing embedded coefficients
        """
        assert coeffs.max() < self.p
        assert coeffs.shape[-1] == self.p
        return self.field(coeffs)
    

class PolyEncoder(nn.Module):
    """Encodes a polynomial into a fixed-dimensional vector representation.
    
    Args:
        p (int): Size of the finite field
        in_dim (int): Dimension of input coefficient embeddings
        poly_dim (int): Dimension of output polynomial encoding
    """
    def __init__(self, p: int, in_dim: int, poly_dim: int):
        super().__init__()
        self.p = p
        self.in_dim = in_dim
        self.poly_dim = poly_dim

        # Project each coefficient embedding to larger dimension
        self.proj_up = nn.Linear(in_dim, poly_dim, bias=False)
        # Mix across coefficient positions
        self.proj_mix = nn.Linear(p, 1, bias=False)

    def forward(self, coeffs):
        """Encodes embedded coefficients into a single vector.
        
        Args:
            coeffs: Tensor of shape (batch, p, in_dim) of embedded coefficients
            
        Returns:
            Tensor of shape (batch, poly_dim) encoding the full polynomial
        """
        x = self.proj_up(coeffs)  # Shape: (batch, p, poly_dim)
        # Mix across coefficient positions and squeeze out singleton dimension
        return torch.swapaxes(self.proj_mix(torch.swapaxes(x, -2, -1)), -2, -1).squeeze()


class PolynomialUnembed(nn.Module):
    """Projects model outputs back to polynomial coefficients using batched matrix multiplication.
    
    Args:
        p (int): Size of the finite field
        model_dim (int): Dimension of model's internal representation
        
    The module learns p matrices of shape (p, model_dim), one for each coefficient
    of each output position. This is stored in the (p, p, model_dim) tensor `unembed`. 
    The slice from `unembed[i, :, :]` maps the activations to logits predicting the value (in [0, ..., p-1])
    of the ith degree coefficient (e.g. `unembed[2, :, :] @ activations` gets the logits for the $x^2$ term).
    Each unembedding projection is applied in parallel using vmap.
    """
    def __init__(self, p: int, model_dim: int):
        super().__init__()
        # Initialize with Kaiming initialization since input comes after ReLU
        std = math.sqrt(2.0 / model_dim)  
        self.unembed = nn.Parameter(torch.randn(p, p, model_dim) * std)
        # Create vmapped matmul operation once at initialization
        self.matmul = torch.vmap(torch.matmul)
            
    def forward(self, x):
        """Projects model representations to polynomial coefficients.
        
        Args:
            x: Tensor of shape (batch, model_dim) containing model outputs
            
        Returns:
            Tensor of shape (batch, p, p) containing predicted coefficients
        """
        batch = x.shape[0]
        # Tile unembedding matrices for batch dimension and apply vmapped matmul
        return self.matmul(self.unembed.tile((batch, 1, 1, 1)), x)


class PolynomialMultiplicationPerceptron(nn.Module):
    """Neural network for multiplying polynomials over finite fields.
    
    Args:
        p (int): Size of the finite field
        embed_dim (int): Dimension for embedding field elements
        poly_dim (int): Dimension for encoding full polynomials
        model_dim (int): Internal dimension for computation
    
    Architecture:
    1. Embed field elements 
    2. Encode each polynomial into a vector
    3. Concatenate encodings and transform
    4. Apply ReLU nonlinearity
    5. Project back to coefficient space
    """
    def __init__(self, p, embed_dim, poly_dim, model_dim):
        super().__init__()
        self.p = p
        self.embed_dim = embed_dim
        self.poly_dim = poly_dim
        self.model_dim = model_dim

        self.field_embed = FieldEmbed(p, embed_dim)
        self.poly_encode = PolyEncoder(p, embed_dim, poly_dim)
        self.linear0 = nn.Linear(2 * poly_dim, model_dim, bias=True)
        self.unembed = PolynomialUnembed(p, model_dim)

    def forward(self, poly_x, poly_y):
        """Computes product of two polynomials.
        
        Args:
            poly_x: Tensor of shape (batch, p) or (p,) containing coefficients of first polynomial
            poly_y: Tensor of shape (batch, p) or (p,) containing coefficients of second polynomial
            
        Returns:
            Tensor of shape (batch, p, p) containing predicted coefficients of product
        """
        assert poly_x.shape == poly_y.shape
        if len(poly_x.shape) == 1:
            poly_x = poly_x.unsqueeze(0)
            poly_y = poly_y.unsqueeze(0)

        _, p = poly_x.shape
        assert p == self.p

        embed_x, embed_y = self.field_embed(poly_x), self.field_embed(poly_y)
        xy_encoding = torch.concat([self.poly_encode(embed_x), self.poly_encode(embed_y)], dim=1)

        activations = self.linear0(xy_encoding)

        return self.unembed(relu(activations))