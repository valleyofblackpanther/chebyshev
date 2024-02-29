import numpy as np
import matplotlib.pyplot as plt

def chebyPolyfit(f, n, a=0, b=1):
    # Generates Chebyshev nodes
    k = np.arange(1, n+2)
    X = np.cos((2*k - 1) * np.pi / (2 * (n+1)))
    X = (b-a)/2 * X + (a+b)/2
    
    # Evaluate the function at Chebyshev nodes
    Y = f(X)
    
    # Perform a polynomial fit using the Chebyshev nodes
    P = np.polynomial.chebyshev.chebfit(X, Y, n)
    
    # Plot the original function
    x = np.linspace(a, b, 200)
    plt.plot(x, f(x), 'r', label='Original function')
    
    # Plot the Chebyshev nodes
    plt.plot(X, Y, 'o', markerfacecolor='k', label='Chebyshev points')
    
    # Plot the fitted polynomial
    y = np.polynomial.chebyshev.chebval(x, P)
    plt.plot(x, y, 'k--', label='Fitted polynomial')
    
    plt.legend()
    plt.show()
    
    return P

# Example usage:
def tanh(x):
    return np.tanh(x)

n = 10  # Degree of the polynomial
P = chebyPolyfit(tanh, n, 0, 1)  # Fit in the domain [0, 1]
