from functools import partial
import jax
import jax.numpy as jnp



def vmap_n(func, n):
    for _ in range(n):
        func = jax.vmap(func)
    return func


def irrep_matrix(p):
    omega = jnp.exp(2j * jnp.pi / (p-1))
    roots_of_unity = jnp.pow(omega, jnp.arange(p-1))
    fft_mat = jnp.pow(jnp.expand_dims(roots_of_unity, 1), jnp.arange(p-1))
    fft_mat = jnp.concat([jnp.zeros((1, p-1)), fft_mat], axis=0)
    zero_ind = jnp.zeros((p, 1)).at[0, 0].set(jnp.sqrt(p-1))
    return jnp.concat([zero_ind, fft_mat], axis=1)


def _poly_monoid_fft(fft_matrix, x):
    """Apply 1D monoid FFT using a pre-computed transform matrix"""
    return jnp.matmul(fft_matrix, x)


def poly_monoid_fft(tensor, p, n):
    """Apply monoid FFT to a multi-dimensional tensor"""
    
    fft_matrix = irrep_matrix(p)
    _fft = vmap_n(partial(_poly_monoid_fft, fft_matrix))

    cycle = tuple([n - 1] + list(range(n - 1)))
    print(cycle)

    def _apply(carry, _):
        # Apply the FFT over n-1 dims, then permute so that the last dim is brought to the front
        # e.g. (0, 1, 2) -> (2, 0, 1) -> (1, 2, 0) -> (0, 1, 2)
        value = _fft(carry)
        return jnp.permute_dims(value, axes=cycle), value
        
    tensor_hat, _  = jax.lax.scan(_apply, tensor.astype(complex), None, length=n)

    return tensor_hat

def poly_monoid_ifft(tensor, p, n):
    ifft_matrix = jnp.conj(irrep_matrix(p)).T / (p - 1)
    _ifft = vmap_n(partial(_poly_monoid_fft, ifft_matrix))

    cycle = tuple([n - 1] + list(range(n - 1)))
    
    def _apply(_, x):
        return jnp.permute_dims(_ifft(x), axes=cycle)
    return jnp.real(jax.lax.fori_loop(0, n, _apply, tensor))
