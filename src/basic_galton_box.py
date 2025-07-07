import matplotlib.pyplot as plt
import numpy as np
import random

def _get_unitary_single(R):

    return np.array([[np.sqrt(1-R), 1j*np.sqrt(R)],
                    [1j*np.sqrt(R), np.sqrt(1-R)]])

def unitary(R, depth):
    
    # 1BS => 2 modes
    # 2 BS => 4 modes
    
    n_modes = 2 * depth 
    
    unitary = np.eye(n_modes, n_modes, dtype=complex)
    
    for i in reversed(range(1, depth+1)):
        u = np.eye(n_modes, dtype=complex)
        for k in reversed(range(i)):
            u[2*k+depth-i:2*k+2+depth-i, 2*k+depth-i:2*k+2+depth-i] = _get_unitary_single(R)
            
        unitary = np.dot(unitary, u)
        #print('u = ', i, u, '\n')
    
    return unitary
    

def sample_basic_board(depth=1, n_photons=1):

    R = 0.5
    
    u = unitary(R, depth)
    input_modes = [0]*2*depth
    input_modes[depth] = 1
    amplitudes = np.dot(u, input_modes)
    probabilities = [abs(a)**2 for a in amplitudes]
    #print(probabilities)
    
    if abs(sum(probabilities)-1) > 1e-6:
        raise Exception('probabilities sum not equal to 1')
    
    samples = random.choices(
            np.arange(0,u.shape[0]),
            weights=probabilities,
            k=n_photons
            )
    
    labels, counts = np.unique(samples, return_counts=True)

    plt.bar(labels, counts, align='center')
    plt.gca().set_xticks(labels)
    plt.xlabel('Output bins')
    plt.ylabel('Counts')
    plt.show()
