1. [Chebyshev Polynomial Fitting and Visualization](https://github.com/valleyofblackpanther/chebyshev/blob/master/ch.py)

This Python script demonstrates how to approximate a function using a Chebyshev polynomial of the first kind and visualize the results. The approximation is performed over the interval \([-1, 1]\) using the function's values at Chebyshev nodes. This method is particularly useful for functions that can be closely approximated by polynomials, offering a way to analyze and visualize the accuracy of the polynomial approximation.

## Features

- Approximation of a user-defined function using Chebyshev polynomials.
- Visualization of the original function, Chebyshev points, and the fitted polynomial.
- Utilization of NumPy for numerical operations and Matplotlib for plotting.

## Usage

1. **Define the function to approximate**: Modify the `f(x)` function within the script to return the function you wish to approximate.
2. **Set the degree of the polynomial**: Adjust the `n` variable to change the degree of the Chebyshev polynomial used for the approximation.
3. **Run the script**: Execute the script to see the approximation and visualization.

The script will plot three elements:
- The original function over the interval \([-1, 1]\).
- The Chebyshev points used for the polynomial fitting.
- The fitted Chebyshev polynomial approximation.

## Function Description

- `f(x)`: User-defined function to approximate. Modify this function as needed.
- `chebyPolyfit(f, n)`: Approximates the function `f` using a Chebyshev polynomial of degree `n` and visualizes the approximation.

## Example

By default, the script approximates the sine function (`np.sin(x)`) using a 10th-degree Chebyshev polynomial. This can be changed to any function and degree as required.

2. [Chebyshev Polynomial Approximation of tanh(x)](https://github.com/valleyofblackpanther/chebyshev/blob/master/ch1.py)

This Python script showcases the approximation of the hyperbolic tangent function (`tanh(x)`) using Chebyshev polynomials within the interval \([0, 1]\). The script generates Chebyshev nodes for the specified degree, calculates the function values at these nodes, and performs a polynomial fit using these values. It employs NumPy for numerical calculations and Matplotlib for visualizing the original function, the Chebyshev nodes, and the polynomial approximation.

## Features

- Generation of Chebyshev nodes within the domain \([0, 1]\).
- Approximation of the `tanh(x)` function using Chebyshev polynomials.
- Visualization of the original function, Chebyshev nodes, and the polynomial approximation.

## Usage

To use this script, follow these steps:

1. **Define the Function**: The script currently approximates `tanh(x)`. This can be replaced or modified in the `tanh(x)` function definition if desired.
2. **Specify the Degree**: Set the variable `n` to the desired degree of the Chebyshev polynomial for the approximation.
3. **Run the Script**: Execute the script to perform the approximation and visualize the results.

The script will output a plot that includes:
- The original `tanh(x)` function plotted across the domain \([0, 1]\).
- The Chebyshev nodes used for the polynomial fitting, marked on the plot.
- The Chebyshev polynomial approximation of the `tanh(x)` function.

## Function Descriptions

- `tanh(x)`: Defines the hyperbolic tangent function to be approximated.
- `chebyshev_nodes(n)`: Generates Chebyshev nodes within the interval \([0, 1]\) for a given degree `n`.
- `chebyPolyfit(f, n)`: Performs the polynomial fit using Chebyshev nodes and plots the original function, nodes, and approximation.

## Example

By setting `n = 20`, the script approximates the `tanh(x)` function using a 20th-degree Chebyshev polynomial and visualizes the approximation alongside the original function and nodes.

3. [Enhanced Chebyshev Polynomial Approximation of tanh(x)](https://github.com/valleyofblackpanther/chebyshev/blob/master/ch2.py)

This script enhances the approximation of the hyperbolic tangent function (`tanh(x)`) using Chebyshev polynomials by incorporating normalization and denormalization processes. These processes allow the approximation to be performed over the interval \([0, 1]\), with the original Chebyshev nodes defined in the interval \([-1, 1]\). This approach facilitates a more versatile function approximation, particularly useful for functions defined in different domains.

## Features

- Normalization of the input domain from \([0, 1]\) to \([-1, 1]\) for Chebyshev polynomial fitting.
- Denormalization of Chebyshev nodes for accurate plotting and function evaluation in the original domain.
- Approximation of `tanh(x)` function over the interval \([0, 1]\) using Chebyshev polynomials.
- Visualization of the original function, Chebyshev nodes, and the polynomial approximation within the \([0, 1]\) domain.

## Usage

The script consists of several key functions:
- `tanh(x)`: The hyperbolic tangent function to be approximated.
- `normalize(x)` and `denormalize(y)`: Functions to normalize and denormalize the data between \([-1, 1]\) and \([0, 1]\), respectively.
- `chebyshev_nodes(n)`: Generates Chebyshev nodes within the \([-1, 1]\) interval and maps them to \([0, 1]\).
- `chebyPolyfit(f, n)`: Approximates the function `f` using a Chebyshev polynomial of degree `n` and visualizes the results within the \([0, 1]\) domain.

To use this script:
1. **Define the Degree**: Set `n` to the desired polynomial degree for the approximation.
2. **Run the Script**: Execute the script to perform the function approximation and visualize the fitting alongside the original function and Chebyshev nodes.

## Example

The default script configuration approximates the `tanh(x)` function over \([0, 1]\) using a 5th-degree Chebyshev polynomial. Adjust `n` to change the approximation's degree.

4.[Chebyshev Polynomial Fitting and Visualization](https://github.com/valleyofblackpanther/chebyshev/blob/master/ch3.py)

This Python script demonstrates how to perform Chebyshev polynomial fitting on any given function within a specified domain and visualize the fitting process. The script calculates the Chebyshev nodes within the given interval, evaluates the function at these nodes, and uses the evaluations to perform a polynomial fit. It then plots the original function, the Chebyshev nodes, and the polynomial approximation for comparison.

## Features

- Calculation of Chebyshev nodes within a user-specified interval.
- Evaluation of a given function at the Chebyshev nodes for polynomial fitting.
- Visualization of the original function, the Chebyshev nodes, and the polynomial approximation.
- Use of NumPy for numerical calculations and Matplotlib for plotting.

## Usage

The script contains the `chebyPolyfit` function which takes the following arguments:
- `f`: The function to approximate.
- `n`: The degree of the Chebyshev polynomial.
- `a`, `b`: The interval \([a, b]\) over which the function is approximated.

To use the script, define the function you wish to approximate and specify the degree of the polynomial and the domain for the fit. For example:

```python
def my_function(x):
    return np.sin(x)  # Define your function here

n = 10  # Degree of the polynomial
P = chebyPolyfit(my_function, n, -1, 1)  # Fit in the domain [-1, 1]
```

## Function Descriptions

- `chebyPolyfit(f, n, a=0, b=1)`: Performs the Chebyshev polynomial fit of the function `f` within the interval \([a, b]\) and plots the original function, Chebyshev nodes, and the approximation.

## Example

The script is set up to approximate the hyperbolic tangent function (`tanh(x)`) over the interval \([0, 1]\) using a 10th-degree Chebyshev polynomial by default. This can be easily modified to fit other functions or intervals as needed.

5. [Optimized Chebyshev Polynomial Approximation](https://github.com/valleyofblackpanther/chebyshev/blob/master/op.py)

This Python module introduces the `ChebyshevOptimized` class, an optimized implementation for approximating and evaluating functions using Chebyshev polynomials. Utilizing the Fast Fourier Transform (FFT) for efficient coefficient computation and Clenshaw's recurrence for evaluation, this class offers a fast and accurate method for function approximation within a user-defined interval.

## Features

- Efficient computation of Chebyshev polynomial coefficients using FFT.
- Vectorized operations for fast execution.
- Accurate evaluation of approximated functions using Clenshaw's recurrence.
- Flexibility to approximate any function defined over a specific interval.

## Usage

To use the `ChebyshevOptimized` class, follow these steps:

1. **Initialization**: Instantiate the class with the interval `[a, b]` over which you wish to approximate a function, the degree `n` of the Chebyshev polynomial, and the function `func` itself.

```python
from numpy import cos, pi

def my_func(x):
    return cos(pi * x)

a = -1  # Start of the interval
b = 1   # End of the interval
n = 10  # Degree of the polynomial

chebyshev = ChebyshevOptimized(a, b, n, my_func)
```

2. **Evaluation**: Call the `eval(x)` method with a value of `x` within the interval `[a, b]` to evaluate the approximated function.

```python
x = 0.5
print(f"Approximated value at x={x}: {chebyshev.eval(x)}")
```

## Class Description

- `ChebyshevOptimized(a, b, n, func)`: Initializes the approximation process by computing Chebyshev nodes, evaluating the function at these nodes, and calculating the polynomial coefficients.
- `eval(x)`: Evaluates the approximated function at a given point `x` within the interval `[a, b]`.

## Example

The provided example demonstrates how to approximate the cosine function over the interval `[-1, 1]` using a 10th-degree Chebyshev polynomial.

6. [Enhanced Chebyshev Polynomial Approximation with JIT Optimization](https://github.com/valleyofblackpanther/chebyshev/blob/master/op2.py)

This Python module offers an optimized approach to computing and evaluating Chebyshev polynomials for function approximation, utilizing the Discrete Cosine Transform (DCT) for efficient coefficient calculation and JIT compilation for performance-critical numerical algorithms. This implementation aims to provide fast, accurate function approximations over specified intervals using Chebyshev polynomial techniques.

## Features

- Efficient Chebyshev coefficient computation using DCT.
- Fast evaluation of Chebyshev polynomial approximations with JIT-compiled Clenshaw's algorithm.
- Accurate and optimized for high-performance computing tasks.
- Support for arbitrary functions over user-defined intervals.

## Usage

To use the `ChebyshevOptimized` class:

1. **Initialization**: Create an instance by specifying the interval `[a, b]`, the degree `n` of the polynomial, and the function `func` to approximate.

```python
from numpy import sin, pi
from chebyshev_optimized import ChebyshevOptimized

def my_function(x):
    return sin(x)

chebyshev = ChebyshevOptimized(0, pi, 10, my_function)
```

2. **Evaluation**: Use the `eval(x)` method to evaluate the approximation at a specific point within the interval `[a, b]`.

```python
x = pi / 4
print(f"Approximated value at x={x}: {chebyshev.eval(x)}")
```

## Class Description

- `ChebyshevOptimized(a, b, n, func)`: Constructor that initializes the approximation process.
- `_compute_coefficients(a, b, n, func)`: Computes the Chebyshev coefficients using DCT for a given function over the interval `[a, b]`.
- `_clenshaw_algorithm(y, c)`: JIT-compiled method for evaluating the Chebyshev polynomial using Clenshaw's algorithm.
- `eval(x)`: Evaluates the approximated function at a given point `x` within the interval `[a, b]`.

## Example

The example provided demonstrates how to approximate the sine function over the interval `[0, π]` using a 10th-degree Chebyshev polynomial.

7. [Polynomial Fit and Error Analysis](https://github.com/valleyofblackpanther/chebyshev/blob/master/r3.py)

This Python script provides a framework for fitting a polynomial to a given function over a specified interval and analyzing the error of the fit. It demonstrates initializing points across the interval, performing a polynomial fit to the function at these points, and conducting an error analysis to compare the original function to the polynomial approximation. Visualization of the fit and error analysis are also included.

## Features

- Initialization of points across a specified interval.
- Polynomial fitting to a function based on initialized points.
- Error analysis of the polynomial fit compared to the original function.
- Visualization of the original function, polynomial fit, and error.


## Usage

To use this script:

1. **Define the function to approximate**: Modify the `example_function` to fit the function of your interest.

2. **Set the interval and number of points**: Adjust the `interval` and `num_initial_points` variables to define the range and resolution of the fit.

3. **Run the script**: Execute the script to perform the polynomial fit and visualize the results and error analysis.

## Functions Description

- `initialize_points(interval, num_points)`: Initializes points uniformly across the given `interval` with `num_points`.
- `polynomial_fit(f, x_points)`: Fits a polynomial to the function `f` at `x_points` and returns the coefficients.
- `error_analysis(f, coeffs, interval)`: Analyzes the error across the interval between the original function `f` and the polynomial represented by `coeffs`.

## Visualization

The script generates two plots:
- The first plot shows the original function, the polynomial fit, and the initial points used for the fitting.
- The second plot focuses on the error between the original function and the polynomial fit across the interval.

## Example

The provided example approximates the hyperbolic tangent function over the interval `[0, π]` using a polynomial fit based on 5 initial points.

8. [Simplified Remez Exchange Algorithm for FIR Filter Design](https://github.com/valleyofblackpanther/chebyshev/blob/master/remez.py)

This Python script provides a simplified implementation of the Remez exchange algorithm. The algorithm is a powerful tool for designing finite impulse response (FIR) filters that meet specified criteria in terms of passband, stopband, and transition band characteristics. This implementation includes functions for generating an initial guess of extremal frequencies, solving a linear system to find filter coefficients, computing the error between the actual and desired filter response, and iteratively updating the extremal frequencies to minimize the maximum error.

## Features

- Generation of an initial guess for extremal frequencies.
- Placeholder function for solving the linear system associated with filter design.
- Computation of error between the actual and the desired filter response.
- Iterative refinement of extremal frequencies to minimize the error.

## Usage

To use this script for FIR filter design:

1. **Define the Desired Response Function**: Customize the `desired_response` function to reflect the frequency response you aim for in your filter.
2. **Specify Filter Parameters**: Set the number of taps (filter order + 1) and band edges for your filter.
3. **Run the Algorithm**: Execute the script to run the Remez algorithm and obtain the filter coefficients.

## Functions Description

- `remez_initial_guess(num_taps, band_edges)`: Generates an initial guess for extremal frequencies based on the filter order and band edges.
- `solve_linear_system(extremal_freqs)`: Solves the linear system to find the filter coefficients. This is a placeholder in the provided script.
- `compute_error(filter_coeffs, desired_response_func, all_freqs)`: Computes the error between the filter's actual response and the desired response.
- `find_new_extremals(errors, all_freqs)`: Identifies new extremal frequencies where the error between the actual and desired response is maximal.
- `remez_algorithm(desired_response_func, num_taps, band_edges)`: Main function to execute the Remez exchange algorithm loop.

## Example

The script includes an example usage section where a desired response function is defined, and the Remez algorithm is called with a specified number of taps and band edges.

9. [Basic Remez Exchange Algorithm Framework](https://github.com/valleyofblackpanther/chebyshev/blob/master/remrezcheby.py)

This Python script introduces a basic framework for implementing the Remez exchange algorithm, an iterative method used in the design of FIR filters that optimally meet a specified frequency response. The script outlines the core steps of the algorithm, including initializing extremal frequencies, solving for filter coefficients, computing the approximation error, and iteratively refining the solution to minimize the maximum error across the filter's frequency band.

## Key Features

- **Initial Guess Generation**: Provides a method for generating an initial guess of extremal frequencies within specified band edges.
- **Linear System Solver Placeholder**: Includes a placeholder function for solving the linear system that determines the filter coefficients.
- **Error Computation**: Implements a method to compute the error between the desired and actual filter responses.
- **Extremal Frequency Refinement**: Details an approach for iteratively finding new extremal frequencies to minimize the error.

## Usage Instructions

To utilize this script for designing an FIR filter:

1. **Define the Desired Frequency Response**: Modify the `desired_response` function to represent the target frequency response of your filter.
2. **Set the Number of Taps and Band Edges**: Adjust the `num_taps` and `band_edges` parameters to specify the filter's order and frequency band of interest.
3. **Execute the Algorithm**: Run the script to apply the Remez exchange algorithm, which will output the initial filter coefficients based on your specifications.

## Detailed Description of Functions

- `remez_initial_guess(num_taps, band_edges)`: Generates evenly spaced initial guesses for extremal frequencies.
- `solve_linear_system(extremal_freqs)`: A placeholder for the method to solve the linear system for filter coefficients, currently returning random values for illustration.
- `compute_error(filter_coeffs, desired_response_func, all_freqs)`: Computes the error between the filter's calculated response and the desired response.
- `find_new_extremals(errors, all_freqs)`: Identifies frequencies with maximal error to adjust extremal frequencies in the next iteration.
- `remez_algorithm(desired_response_func, num_taps, band_edges)`: Orchestrates the Remez algorithm process, iterating to refine filter coefficients.

## Example

An example usage is provided within the script, demonstrating how to approximate a specific desired frequency response within a defined interval using the Remez algorithm framework.
