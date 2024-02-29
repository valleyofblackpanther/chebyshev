import numpy as np
import matplotlib.pyplot as plt

def initialize_points(interval, num_points):
    """Step 1: Initialize points across the interval."""
    return np.linspace(interval[0], interval[1], num_points)

def polynomial_fit(f, x_points):
    """Step 2: Fit a polynomial to the function at the given points."""
    # This is a placeholder for a more complex fitting process
    y_points = f(x_points)
    degree = len(x_points) - 1
    coeffs = np.polyfit(x_points, y_points, degree)
    return coeffs

def error_analysis(f, coeffs, interval):
    """Step 3: Analyze error across the interval."""
    x_fine = np.linspace(interval[0], interval[1], 1000)
    y_fine = f(x_fine)
    y_poly = np.polyval(coeffs, x_fine)
    error = y_fine - y_poly
    return x_fine, error

# Example usage
def example_function(x):
    return np.tanh(x)

interval = [0, np.pi]  # Define the interval of interest
num_initial_points = 5  # Starting with 5 points

# Step 1: Initialize points
x_points = initialize_points(interval, num_initial_points)

# Step 2: Fit polynomial
coeffs = polynomial_fit(example_function, x_points)

# Step 3: Error analysis
x_fine, error = error_analysis(example_function, coeffs, interval)

# Plotting for visualization
plt.figure(figsize=(10, 5))
plt.plot(x_fine, example_function(x_fine), label='Original Function')
plt.plot(x_fine, np.polyval(coeffs, x_fine), '--', label='Polynomial Fit')
plt.scatter(x_points, example_function(x_points), color='red', label='Initial Points')
plt.title('Initial Polynomial Fit and Error Analysis')
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(x_fine, error, label='Error')
plt.title('Error Analysis')
plt.legend()
plt.show()
