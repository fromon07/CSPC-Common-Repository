"""
PW2 Lab B Part 3 -- fit a reaction's rate constant to measured data.

A first-order reaction decays as  C(t) = C0 * exp(-k*t).  You have noisy
concentration-vs-time measurements; find the k that best matches them.
Complete the TODOs. Run:  python kinetics.py
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# TODO 1: read kinetics.csv (columns time, concentration) into arrays t, C.
#         Set C0 = the first concentration.
t, C = np.loadtxt('/home/aydan/Desktop/CSPC/PW2/Lab B/kinetics.csv', delimiter=',', skiprows=1, unpack=True)
C0 = C[0]

# TODO 2: write total_error(k) = sum of (measured - C0*exp(-k*t))^2.
#         This is the "how bad" number: small when the model matches the data.
def total_error(k):
    return np.sum((C - C0 * np.exp(-k * t)) ** 2)

# TODO 3: minimise total_error with scipy.optimize.minimize (method "SLSQP",
#         bounds [(0, 5)], start x0=0.5). Print the fitted k.
result = minimize(total_error, x0=0.5, bounds=[(0, 5)], method="SLSQP")
print("Fitted k:", result.x[0])
# TODO 4: plot the measured data (points) and your fitted curve (line) together.
#         Save as kinetics.png.
t_fit = np.linspace(t[0], t[-1], 100)
C_fit = C0 * np.exp(-result.x[0]*t_fit)
plt.figure(figsize=(8, 6))
plt.scatter(t, C, label='Measured Data', color='blue')
plt.plot(t_fit, C_fit, label='Fitted Curve', color='red')
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.title('Kinetics_Fit')
plt.legend()
plt.savefig('/home/aydan/Desktop/CSPC/PW2/Lab B/kinetics.png')