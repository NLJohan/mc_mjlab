import numpy as np
import matplotlib.pyplot as plt

def mujoco_impedance(p, dmin, dmax, width, midpoint, power):
    # Normalize penetration depth relative to width
    x = np.clip(p / width, 0, 1)
    
    # MuJoCo's internal sigmoidal spline interpolation logic
    # split by the midpoint threshold
    cond = x < midpoint
    
    # Initialize output array
    y = np.zeros_like(x)
    
    # Left side of the midpoint
    y[cond] = dmin + (dmax - dmin) * (midpoint ** (1 - power)) * (x[cond] ** power)
    
    # Right side of the midpoint
    y[~cond] = dmax - (dmax - dmin) * ((1 - midpoint) ** (1 - power)) * ((1 - x[~cond]) ** power)
    
    # Flat line beyond width
    y[p >= width] = dmax
    return y

# Simulation parameters
width = 0.01  # 10mm
dmin, dmax = 0.9, 0.95
p = np.linspace(0, 0.015, 500)  # Penetration depth from 0 to 15mm

# Plotting different midpoints
plt.figure(figsize=(8, 5))
for mid in [0.5]:
    d = mujoco_impedance(p, dmin, dmax, width, mid, power=5)
    plt.plot(p * 1000, d, label=f'impedance')

plt.axvline(x=width*1000, color='red', linestyle='--', label='width (10mm)')
plt.title("solimp depth-dependent impedance")
plt.xlabel("Penetration Depth (mm)")
plt.ylabel("Impedance")
plt.legend()
plt.grid(True)
plt.show()
