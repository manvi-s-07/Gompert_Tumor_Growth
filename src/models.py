import numpy as np 
import matplotlib.pyplot as plt

# =======================================================
# TUMOR GROWTH MODELS
# =======================================================

def exponential_model(v0, r, t):
    volume = v0 * np.exp((r*t))
    return volume 

def logistic_model(v0, r, k, t):
    volume = (v0 * k * np.exp((r * t)))/((k - v0) + (v0 * np.exp((r * t))))
    return volume

def gompertz_model(v0, k, a, t):
    volume = k * (np.exp(-np.log(k/v0) * np.exp(-a * t)))
    return volume

# Curve fits for parameters 

def gompertz_for_fit(t, v0, k, a):
    return gompertz_model(v0, k, a, t)

def logistic_for_fit(t, v0, r, k):
    return logistic_model(v0, r, k, t)

def exponential_for_fit(t, v0, r):
    return exponential_model(v0, r, t)



