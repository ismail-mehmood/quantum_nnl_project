import numpy as np
from .common import *

def mach_zehnder_unitary(phi):
    BS = beam_splitter()
    P = phase_shift(phi)
    return BS @ P @ BS # phase shift multiplied with beam splitter matrices

def mzi(intervals=10):
    phases = np.linspace(0, 2*np.pi, intervals)
    input_state = np.array([1, 0])  # Single photon in mode 0
    results = []

    for phi in phases:
        U = mach_zehnder_unitary(phi)
        output_state = U @ input_state
        probs = np.abs(output_state)**2  # Probabilities for modes
        results.append(probs)
        
    return phases, results
