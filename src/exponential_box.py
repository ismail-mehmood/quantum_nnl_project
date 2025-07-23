from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np
from enum import Enum

class CircuitType(Enum):
    FULL = "full"
    FAST = "fast"

def biased_galton_n_layer(n, shots=1000, mode="full", initial_theta=np.pi/2, decay_factor=3.0):

    layer_index = 1

    theta = initial_theta * np.exp(-layer_index / decay_factor)

    if mode == "full":
        total_qubits = 2 * n + 2  # control + 2n + 1 pegs
        qc = QuantumCircuit(total_qubits, n + 1)

        control = 0

        # Start ball in the middle peg
        centre = n + 1
        qc.x(centre)

        # For each layer, peg is at index i (excluding control qubit)
        for i in range(n):

            # Set layer index
            layer_index = i + 1

            # Reset control for next layer
            qc.reset(control)

            # Apply biased rotation to control
            qc.ry(theta, control)

            # Every peg per layer, working right to left
            for j in range(i + 1):
                middle = centre - i + (2 * j) # Gets centre qubit relative to board position
                left = middle + 1
                right = middle - 1 

                # Controlled-SWAP: ball goes right
                qc.cswap(control, middle, right)

                # CNOT to entangle (centre → control)
                qc.cx(middle, control)

                # Controlled-SWAP: ball goes left
                qc.cswap(control, middle, left)

                # Inter-peg logic (optional)
                if j < i:
                    qc.cx(middle + 1, control)

        # Measure qubits corresponding to odds, these are the pegs/bins
        for i in range(0, n + 1):
            qc.measure(2 * i + 1, i)

    else:
        # Fast version
        qc = QuantumCircuit(n, n)
        for i in range(n):
            qc.ry(theta, i)
        qc.measure(range(n), range(n))

    # Simulate
    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Bin counts
    bin_counts = {i: 0 for i in range(n + 1)}
    if mode == "full":
        for bitstring, count in counts.items():
            for i, bit in enumerate(reversed(bitstring)):
                if bit == '1':
                    bin_counts[i] += count
                    break
    else:
        for bitstring, count in counts.items():
            num_ones = bitstring.count('1')
            bin_counts[num_ones] += count

    return bin_counts
