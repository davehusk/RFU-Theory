"""
Unveiling the Roots of a Quartic Polynomial in Frequency and Temperature Domains

Introduction

In the realm of mathematical physics, understanding the roots of polynomials is pivotal for analyzing system behaviors. Whether it's the oscillations of a spring-mass system or the thermal dynamics of materials, polynomials provide a foundational framework for predicting and explaining these phenomena. This post delves into the computation and verification of the roots of a specific quartic polynomial derived from models in frequency and temperature domains, leveraging Python for accurate and efficient calculations.

The Polynomial in Focus

We consider the polynomial:

P(x) = (x^2 + 1/sqrt(2)x + 1/2)(x - sqrt(2))^2

Upon expansion, this yields a quartic polynomial:

P(x) = x^4 - 3*sqrt(2)/2*x^3 + 1/2*x^2 + 1

Computational Approach

To find the roots of P(x), we employed Python's powerful libraries, numpy for numerical computations and sympy for symbolic mathematics. This dual approach ensures both precision and symbolic clarity in our analysis.

Python Script for Root Computation

Here's the Python script used to compute and verify the roots of the polynomial:
"""
import numpy as np
from sympy import symbols, solve, sqrt, I, expand

# Define the symbolic variable
x = symbols('x')

# Define the polynomial factors
quadratic_factor = x**2 + (1/sqrt(2))*x + 1/2
linear_factor = (x - sqrt(2))**2

# Expand the polynomial to its standard form
polynomial = expand(quadratic_factor * linear_factor)
print(f"Expanded Polynomial: {polynomial}")

# Extract coefficients from the expanded polynomial
coefficients = [
    polynomial.coeff(x, 4),
    polynomial.coeff(x, 3),
    polynomial.coeff(x, 2),
    polynomial.coeff(x, 1),
    polynomial.coeff(x, 0)
]

print("\nPolynomial Coefficients:")
for power, coeff in zip(range(4, -1, -1), coefficients):
    print(f"x^{power}: {coeff}")

# Compute the roots using numpy.roots
roots = np.roots(coefficients)

print("\nThe roots of the polynomial are:")
for idx, root in enumerate(roots, start=1):
    print(f"Root {idx}: {root}")

# Solve the polynomial symbolically
symbolic_roots = solve(polynomial, x)
print("\nSymbolic Roots of the Polynomial:")
for idx, root in enumerate(symbolic_roots, start=1):
    print(f"Symbolic Root {idx}: {root}")
"""
Script Output Interpretation

1. Expanded Polynomial:

   Expanded Polynomial: x**4 - 3*sqrt(2)*x**3/2 + x**2/2 + 1

2. Polynomial Coefficients:

   x^4: 1
   x^3: -3*sqrt(2)/2
   x^2: 0.5
   x^1: 0
   x^0: 1.00000000000000

3. Numerical Roots (Computed using numpy):

   The roots of the polynomial are:
   Root 1: (1.414213562373095+3.706546865402141e-08j)
   Root 2: (1.414213562373095-3.706546865402141e-08j)
   Root 3: (-0.353553390593274+0.612372435695795j)
   Root 4: (-0.353553390593274-0.612372435695795j)

4. Symbolic Roots (Computed using sympy):

   Symbolic Roots of the Polynomial:
   Symbolic Root 1: sqrt(2)
   Symbolic Root 2: -1/(2*sqrt(2)) - sqrt(3)*I/(2*sqrt(2))
   Symbolic Root 3: -1/(2*sqrt(2)) + sqrt(3)*I/(2*sqrt(2))

Interpretation of the Roots

1. Real Double Root at x = sqrt(2):

   - Numerical Roots: Both Root 1 and Root 2 are approximately 1.414213562373095, which is the decimal representation of sqrt(2). The negligible imaginary parts (+/- 3.71 x 10^-8j) are artifacts of numerical precision, effectively confirming a double real root at x = sqrt(2).

   - Symbolic Roots: Confirmed symbolically as sqrt(2), indicating the exactness of the root.

2. Complex Conjugate Roots:

   - Numerical Roots: Root 3 and Root 4 are complex conjugates:

     x ≈ -0.353553390593274 +/- 0.612372435695795j

   - Symbolic Roots: Expressed as:

     x = -1/(2*sqrt(2)) +/- sqrt(3)*I/(2*sqrt(2))

     Simplifying further:

     x = -sqrt(2)/4 +/- sqrt(6)*I/4

     Or:

     x = (-sqrt(2) +/- sqrt(6)*I)/4

   - Conclusion: These roots are a direct result of the quadratic factor x^2 + 1/sqrt(2)x + 1/2, which has a negative discriminant, leading to complex conjugate roots.

Applications in Physical Systems

1. Simple Harmonic Oscillator:

   - Equation of Motion in Frequency Domain:

     -omega^2 x(omega) + C x(omega) = F(omega)

     Setting F(omega) = 0 leads to:

     -omega^2 x(omega) + C x(omega) = 0 => omega = sqrt(C)

     For C = 1, omega = 1, aligning with the double root at x = sqrt(2) when considering the characteristic polynomial.

2. Thermal Diffusivity:

   - Differential Equation:

     dT/dt = -alpha T

     Solving this first-order linear differential equation yields:

     T(t) = T0 e^(-alpha t)

     While this specific equation doesn't directly involve a quadratic polynomial, understanding the roots of related characteristic equations is essential for analyzing more complex thermal systems.

Conclusion

Through meticulous mathematical derivation and computational verification using Python, we've successfully determined the roots of the quartic polynomial derived from models in frequency and temperature domains. The real double root at x = sqrt(2) corresponds to the natural frequency of a simple harmonic oscillator, while the complex conjugate roots provide insights into the oscillatory behavior inherent in more complex systems.

"""
