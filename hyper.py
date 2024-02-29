def create_breakpoints(n):
    # Create n+1 breakpoints between 0 and 1
    breakpoints = [i/(2**n) for i in range(2**n + 1)]
    return breakpoints

n = 3  # For example, if n=3, it creates 2^3 intervals
breakpoints = create_breakpoints(n)
print(breakpoints)
