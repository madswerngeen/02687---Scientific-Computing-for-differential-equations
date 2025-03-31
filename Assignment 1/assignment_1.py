import sympy as sp

# Define the variables
c0, c1, c2, c3, c4, h = sp.symbols('c0 c1 c2 c3 c4 h')

# Define the system of equations
eq1 = sp.Eq(c0 + c1 +      c2 +      c3 +      c4, 0)
eq2 = sp.Eq(     c1 + 2*   c2 + 3*   c3 + 4*   c4, 0)
eq3 = sp.Eq(     c1 + 2**2*c2 + 3**2*c3 + 4**2*c4, 2/sp.Pow(h, 2))
eq4 = sp.Eq(     c1 + 2**3*c2 + 3**3*c3 + 4**3*c4, 0)
eq5 = sp.Eq(     c1 + 2**4*c2 + 3**4*c3 + 4**4*c4, 0)

# Solve the system
solution = sp.solve((eq1, eq2, eq3, eq4, eq5), (c0, c1, c2, c3, c4))
print(solution)

# Define the variables
a2, a1, c0, c1, c2, h = sp.symbols('a2 a1 c0 c1 c2 h')

# Define the system of equations
eq1 = sp.Eq(      a2 + a1 + c0 + c1 +      c2, 0)
eq2 = sp.Eq(   -2*a2 - a1 +      c1 +    2*c2, 0)
eq3 = sp.Eq( 2**2*a2 + a1 +      c1 + 2**2*c2, 2/sp.Pow(h, 2))
eq4 = sp.Eq(-2**3*a2 - a1 +      c1 + 2**3*c2, 0)
eq5 = sp.Eq( 2**4*a2 + a1 +      c1 + 2**4*c2, 0)

# Solve the system
solution = sp.solve((eq1, eq2, eq3, eq4, eq5), (a2, a1, c0, c1, c2))
print(solution)