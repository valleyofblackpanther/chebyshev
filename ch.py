import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Chebyshev as Ch

def f(x):
    # Define your function here. For example:
    return np.sin(x)

def chebyPolyfit(f, n):
    k = np.arange(1, n+1)
    X = np.cos((2*k - 1) * np.pi / (2 * n))
    Y = f(X)
    p = np.polynomial.chebyshev.chebfit(X, Y, n)
    
    # Plotting the results
    # Plot the original function
    x_plot = np.linspace(-1, 1, 200)
    plt.plot(x_plot, f(x_plot), label='Original function')
    
    # Plot the Chebyshev points
    plt.plot(X, Y, 'o', markerfacecolor='none', label='Chebyshev points')
    
    # Plot the fitted polynomial
    p_poly = Ch(coef=p)
    plt.plot(x_plot, p_poly(x_plot), label='Fitted polynomial')
    
    plt.legend()
    plt.show()

    return p

# Example usage:
n = 10  # Degree of the polynomial
chebyPolyfit(f, n)
