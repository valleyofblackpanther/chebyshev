import numpy as np
from scipy.fft import dct
from numba import jit

class ChebyshevOptimized:
    def __init__(self, a, b, n, func):
        self.a, self.b = a, b
        self.c = self._compute_coefficients(a, b, n, func)

    def _compute_coefficients(self, a, b, n, func):
        # Compute Chebyshev nodes
        k = np.arange(n)
        x = np.cos(np.pi * (k + 0.5) / n) * (0.5 * (b - a)) + (0.5 * (b + a))
        f = np.vectorize(func)(x)

        # Compute coefficients using DCT, more accurate for this context
        c = dct(f, type=2, norm='ortho') * 2 / n
        c[0] /= 2  # Adjust the first coefficient
        return c[:n]  # Keep only the first n coefficients

    @staticmethod
    @jit(nopython=True)
    def _clenshaw_algorithm(y, c):
        d, dd = c[-1], 0
        for cj in c[-2:0:-1]:
            d, dd = 2.0 * y * d - dd + cj, d
        return y * d - dd + 0.5 * c[0]

    def eval(self, x):
        assert(self.a <= x <= self.b), "x is outside the interval [a, b]"
        y = (2.0 * x - self.a - self.b) / (self.b - self.a)
        return self._clenshaw_algorithm(y, self.c)

# Example usage:
# Define the function to approximate
def func(x):
    return np.sin(x)

# Create an instance for the interval [0, np.pi], degree 10
cheb_approx = ChebyshevOptimized(0, np.pi, 10, func)

# Evaluate the approximation at some point
print(cheb_approx.eval(np.pi / 4))
