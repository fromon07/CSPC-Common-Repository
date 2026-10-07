"""
PW2 Lab B Part 2 -- three routes to a minimum.

Compare gradient descent, Newton, and SLSQP on two functions:
  2A: f(x) = (x-3)**2 + 1          (easy, one minimum at x=3)
  2B: g(x) = x**4 - 3*x**2 + x + 5 (harder, several stationary points)
Run:  python warmup.py
"""
import numpy as np
from scipy.optimize import newton, minimize

# ---------- 2A: easy convex function ----------
def f(x):   return (x-3)**2 + 1
def df(x):  return 2*(x-3)
def d2f(x): return 2.0
# TODO 2A: minimise f three ways from x0=0 and print each result:
#   (1) gradient descent by hand (loop x = x - lr*df(x) until the step is tiny)
#   (2) scipy.optimize.newton(df, x0, fprime=d2f)
#   (3) scipy.optimize.minimize(f, x0, method="SLSQP")

def gradient_descent(f, df, x0, lr=0.1, tol=1e-6, max_iter=1000):
    x = x0
    for i in range(max_iter):
        step = lr * df(x)
        if abs(step) < tol:
            break
        x -= step
    return x
for lr in [0.5, 0.1, 0.01, 0.001]:
    xmin1 = gradient_descent(f, df, x0=0, lr=lr)
    print(f"Gradient descent (lr={lr}) found minimum at x={xmin1:.6f}")

xmin2 = minimize(f, x0=0, method="SLSQP").x[0]
print(f"SLSQP found minimum at x={xmin2:.6f}")

xmin3 = newton(df, x0=0, fprime=d2f)
print(f"Newton's method found minimum at x={xmin3:.6f}")
# ---------- 2B: harder landscape ----------
def g(x):   return x**4 - 3*x**2 + x + 5
def dg(x):  return 4*x**3 - 6*x + 1
def d2g(x): return 12*x**2 - 6

# TODO 2B: run the same three methods on g, from x0=0 AND from x0=2.
#   For Newton (which solves dg(x)=0), also check the sign of d2g at the answer:
#   d2g > 0 means a minimum, d2g < 0 means a maximum.
#   In your README note: do the methods agree? did Newton find a minimum or
#   another stationary point? how did the starting point change the result?
print("\nStarting from x0=0:")
for lr in [0.1, 0.01, 0.001]:
    xming1 = gradient_descent(g, dg, x0=0, lr=lr)
    print(f"Gradient descent for g (lr={lr}, x0=0) found minimum at x={xming1:.6f}")
xming2 = minimize(g, x0=0, method="SLSQP").x[0]
print(f"SLSQP for g (x0=0) found minimum at x={xming2:.6f}")
xming3 = newton(dg, x0=0, fprime=d2g)
print(f"Newton's method for g (x0=0) found stationary point at x={xming3:.6f}, d2g={d2g(xming3):.6f}")

print("\nStarting from x0=2:")
for lr in [0.1, 0.01, 0.001]:
    xming1_2 = gradient_descent(g, dg, x0=2, lr=lr)
    print(f"Gradient descent for g (lr={lr}, x0=2) found minimum at x={xming1_2:.6f}")
xming2_2 = minimize(g, x0=2, method="SLSQP").x[0]
print(f"SLSQP for g (x0=2) found minimum at x={xming2_2:.6f}")  
xming3_2 = newton(dg, x0=2, fprime=d2g)
print(f"Newton's method for g (x0=2) found stationary point at x={xming3_2:.6f}, d2g={d2g(xming3_2):.6f}")