import numpy as np

# Beam splitter matrix, now with tuneable reflectivity
def beam_splitter(theta=np.pi/4):
    return np.array([[np.cos(theta), 1j*np.sin(theta)],
                     [1j*np.sin(theta), np.cos(theta)]])


# U(phi)
def phase_shift(phi):
    return np.array([[np.exp(1j * phi), 0],
                     [0, 1]])

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
