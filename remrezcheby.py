import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

class Chebyshev:
    def __init__(self, a, b, n, func):
        self.a = a
        self.b = b
        self.func = func
        self.n = n
        self.coefficients = np.zeros(n)
        self._compute_coefficients()

    def _compute_coefficients(self):
        bma = 0.5 * (self.b - self.a)
        bpa = 0.5 * (self.b + self.a)
        f = [self.func(np.cos(np.pi * (k + 0.5) / self.n) * bma + bpa) for k in range(self.n)]
        fac = 2.0 / self.n
        self.coefficients = [fac * sum([f[k] * np.cos(np.pi * j * (k + 0.5) / self.n)
                           for k in range(self.n)]) for j in range(self.n)]

    def eval(self, x):
        a, b = self.a, self.b
        # Ensure x is a NumPy array for element-wise comparison
        x = np.asarray(x)
        # Check that all values in x are within [a, b]
        assert np.all((a <= x) & (x <= b)), "x values are out of bounds"
        y = (2.0 * x - a - b) / (b - a)
        y2 = 2.0 * y
        d, dd = self.coefficients[-1], 0
        for cj in self.coefficients[-2:0:-1]:
            d, dd = y2 * d - dd + cj, d
        return y * d - dd + 0.5 * self.coefficients[0]


def remez_for_chebyshev(chebyshev, num_iter=10):
    # Initial extremal frequencies (using Chebyshev nodes)
    extremal_freqs = np.cos((np.pi * (np.arange(chebyshev.n) + 0.5)) / chebyshev.n)
    
    for _ in range(num_iter):
        # Solve for new coefficients using the extremal frequencies
        A = np.cos(np.outer(np.arange(chebyshev.n), np.arccos(extremal_freqs))) 
        b = chebyshev.func(extremal_freqs)
        chebyshev.coefficients = scipy.linalg.solve(A, b)
        
        # Compute the error across the frequency range
        dense_freqs = np.linspace(chebyshev.a, chebyshev.b, 1000)
        errors = chebyshev.func(dense_freqs) - chebyshev.eval(dense_freqs)
        max_error = np.max(np.abs(errors))
        
        # Find new extremal frequencies where the error is maximal
        new_extremals = dense_freqs[np.argsort(np.abs(errors))[-chebyshev.n:]]
        
        # Check for convergence
        if np.allclose(new_extremals, extremal_freqs, atol=1e-6):
            break
        
        extremal_freqs = new_extremals

    return chebyshev

# Example usage
def func_to_approximate(x):
    return np.tanh(x)

# Create Chebyshev object
chebyshev = Chebyshev(0, 1, 20, func_to_approximate)

# Run the Remez-like optimization algorithm
optimized_chebyshev = remez_for_chebyshev(chebyshev)

# Print the optimized coefficients
print("Optimized Coefficients:", optimized_chebyshev.coefficients)

def plot_chebyshev(chebyshev):
    # Calculate Chebyshev points for plotting
    chebyshev_points_x = np.cos(np.pi * (np.arange(chebyshev.n) + 0.5) / chebyshev.n) * 0.5 * (chebyshev.b - chebyshev.a) + 0.5 * (chebyshev.b + chebyshev.a)
    chebyshev_points_y = chebyshev.func(chebyshev_points_x)
    
    # Dense set of points for smooth curves
    x_dense = np.linspace(chebyshev.a, chebyshev.b, 1000)
    y_dense = chebyshev.func(x_dense)
    y_poly = chebyshev.eval(x_dense)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(x_dense, y_dense, 'r-', label='Original Function')
    plt.plot(chebyshev_points_x, chebyshev_points_y, 'bo', label='Chebyshev Points')
    plt.plot(x_dense, y_poly, 'k--', label='Fitted Polynomial')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Chebyshev Polynomial Approximation')
    plt.show()

# Assuming the rest of the code and example usage is defined and executed
plot_chebyshev(optimized_chebyshev)