import numpy as np

# Beam splitter matrix, now with tuneable reflectivity
def beam_splitter(theta=np.pi/4):
    return np.array([[np.cos(theta), 1j*np.sin(theta)],
                     [1j*np.sin(theta), np.cos(theta)]])


# U(phi)
def phase_shift(phi):
    return np.array([[np.exp(1j * phi), 0],
                     [0, 1]])