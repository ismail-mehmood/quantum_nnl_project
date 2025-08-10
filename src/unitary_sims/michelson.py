import numpy as np
from .common import *

def michelson_unitary(phi):
    BS = beam_splitter()
    # Michelson effectively doubles the optical path phase in one arm
    P = phase_shift(2 * phi)
    return BS @ P @ BS

def michelson(intervals=10):
    phases = np.linspace(0, 2*np.pi, intervals)
    input_state = np.array([1, 0])
    results = []

    for phi in phases:
        U = michelson_unitary(phi)
        output_state = U @ input_state
        probs = np.abs(output_state) ** 2
        results.append(probs)

    return phases, results
