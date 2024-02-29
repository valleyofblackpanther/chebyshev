import numpy as np

class ChebyshevOptimized:
    def __init__(self, a, b, n, func):
        self.a = a
        self.b = b
        self.func = func

        # Vectorized computation of Chebyshev nodes and function values
        k = np.arange(n)
        x = np.cos(np.pi * (k + 0.5) / n) * (0.5 * (b - a)) + (0.5 * (b + a))
        f = func(x)

        # Compute coefficients using vectorized operations
        fac = 2.0 / n
        c = np.fft.rfft(f) * fac
        c[0] /= 2
        self.c = c.real[:n]  # Keep only the first n coefficients for consistency with the original implementation

    def eval(self, x):
        assert(self.a <= x <= self.b)
        y = (2.0 * x - self.a - self.b) / (self.b - self.a)
        
        # Clenshaw's recurrence with vectorized computation
        d = self.c[-1]
        dd = 0
        for cj in self.c[-2:0:-1]:
            (d, dd) = (2.0 * y * d - dd + cj, d)
        return y * d - dd + 0.5 * self.c[0]
