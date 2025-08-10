import numpy as np
from .common import *

def ramsey_unitary(phi):
    BS = beam_splitter()  # acts like π/2 rotation
    P = phase_shift(phi)  # free evolution / phase accumulation
    return BS @ P @ BS

def ramsey(intervals=10):
    phases = np.linspace(0, 2*np.pi, intervals)
    input_state = np.array([1, 0])
    results = []

    for phi in phases:
        U = ramsey_unitary(phi)
        output_state = U @ input_state
        probs = np.abs(output_state) ** 2
        results.append(probs)

    return phases, results
