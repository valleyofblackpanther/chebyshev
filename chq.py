@QFunc
def compute_tanh(precision: QParam[int], x: QNum, tanh_x: Output[QNum]):
    # 1. Define breakpoints for piecewise segments:
    num_breakpoints=5
    breakpoints = [i/5 for i in range(0,num_breakpoints)]  # Example breakpoints

    # 2. Calculate slopes and intercepts for each segment using Taylor polynomials:
    slopes = []
    intercepts = []
    for i in range(len(breakpoints) - 1):
        # Calculate a and b for the linear function in this segment using Taylor expansion
        a, b = calculate_taylor_coefficients(breakpoints[i], breakpoints[i+1], precision)
        slopes.append(a)
        intercepts.append(b)