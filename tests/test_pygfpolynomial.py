import pytest
import numpy as np
import galois
from neural_polytest import PyGFPolynomial

@pytest.fixture(params=[2, 3, 5, 7, 11])
def gf_setup(request):
    """Setup finite field and modulus polynomial for testing.
    Tests GF(2), GF(3), GF(5), GF(7), GF(3^3), and GF(11)"""
    p = request.param
    GF = galois.GF(p)
    # x^p - x for each field
    modulus = galois.Poly([1] + [0]*(p-2) + [-1, 0], field=GF)
    gf_poly = PyGFPolynomial(p, 42)
    return GF, modulus, gf_poly, p


def test_polynomial_multiplication(gf_setup):
    """Test that our Rust polynomial multiplication matches Galois library results."""
    GF, modulus, gf_poly, p = gf_setup
    batch_size = 50
    
    # Generate batch of test cases
    batch_x, batch_y, batch_xy = gf_poly.generate_batch(batch_size)
    batch_x = np.flip(batch_x, axis=1)
    batch_y = np.flip(batch_y, axis=1)
    batch_xy = np.flip(batch_xy, axis=1)
    
    for i in range(batch_size):
        poly_x = galois.Poly(batch_x[i], field=GF)
        poly_y = galois.Poly(batch_y[i], field=GF)
        poly_xy = galois.Poly(batch_xy[i], field=GF)
        
        _, prod = divmod(poly_x * poly_y, modulus)
        assert poly_xy == prod, f'Field GF({p}), modulus {modulus}, Mismatch at index {i}:\nExpected: {prod}\nGot: {poly_xy}'
