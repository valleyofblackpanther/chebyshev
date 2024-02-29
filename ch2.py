import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def tanh(x):
    # No change needed; tanh is already suitable for [-1, 1]
    return np.tanh(x)

def normalize(x):
    # Normalize from [0, 1] to [-1, 1]
    return 2 * x - 1

def denormalize(y):
    # Denormalize from [-1, 1] to [0, 1]
    return (y + 1) / 2

def chebyshev_nodes(n):
    # Generates Chebyshev nodes in the domain [-1, 1] and then maps them to [0, 1]
    nodes = np.cos((2 * np.arange(1, n + 1) - 1) * np.pi / (2 * n))
    return denormalize(nodes)

def chebyPolyfit(f, n):
    X = chebyshev_nodes(n)  # Nodes are now correctly mapped to [0, 1]
    Y = f(normalize(X))  # Apply the function to normalized nodes
    
    # Perform a Chebyshev fit using numpy's polynomial module to the normalized values
    coefs = np.polynomial.chebyshev.chebfit(normalize(X), Y, n)
    
    # Define the Chebyshev polynomial using the coefficients, in the normalized domain [-1, 1]
    p = np.polynomial.Chebyshev(coefs, domain=[-1, 1])
    
    # Plot the original function over [0, 1]
    x_plot = np.linspace(0, 1, 200)
    plt.plot(x_plot, f(normalize(x_plot)), label='tanh(x)')
    
    # Plot the Chebyshev points
    plt.plot(X, Y, 'o', markerfacecolor='none', label='Chebyshev points')
    
    # Plot the fitted polynomial, denormalizing both the polynomial and the x values for correct plotting
    plt.plot(x_plot, p(normalize(x_plot)), label='Fitted polynomial')
    
    plt.legend()
    plt.show()
    
    return coefs

# Example usage:
n = 5  # Degree of the polynomial
chebyPolyfit(tanh, n)
