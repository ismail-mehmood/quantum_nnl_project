import numpy as np

# Beam splitter matrix
def beam_splitter():
    return (1/np.sqrt(2)) * np.array([[1, 1j],
                                      [1j, 1]])

# U(phi)
def phase_shift(phi):
    return np.array([[np.exp(1j * phi), 0],
                     [0, 1]])

def mach_zehnder_unitary(phi):
    BS = beam_splitter()
    P = phase_shift(phi)
    return BS @ P @ BS # phase shift multiplied with beam splitter matrices
