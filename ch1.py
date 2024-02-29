import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def tanh(x):
    return np.tanh(x)

def chebyshev_nodes(n):
    # Generates Chebyshev nodes in the domain [0, 1]
    return np.cos((2*np.arange(1, n+1) - 1) * np.pi / (2 * n)) / 2 + 0.5

def chebyPolyfit(f, n):
    X = chebyshev_nodes(n)
    Y = f(X)
    
    # Perform a Chebyshev fit using numpy's polynomial module
    coefs = np.polynomial.chebyshev.chebfit(X, Y, n)
    
    # Define the Chebyshev polynomial using the coefficients
    p = np.polynomial.Chebyshev(coefs, domain=[0, 1])
    
    # Plot the original function
    x_plot = np.linspace(0, 1, 200)
    plt.plot(x_plot, f(x_plot), label='tanh(x)')
    
    # Plot the Chebyshev points
    plt.plot(X, Y, 'o', markerfacecolor='none', label='Chebyshev points')
    
    # Plot the fitted polynomial
    plt.plot(x_plot, p(x_plot), label='Fitted polynomial')
    
    plt.legend()
    plt.show()
    
    return coefs

# Example usage:
n = 20  # Degree of the polynomial
chebyPolyfit(tanh, n)
