from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# Hadamard gate acts as ..., default shots 1000

# One layer box, (1 peg, 2 bins)

def galton_one_layer(shots=1000):
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)

    # Simulate
    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Get bin index, by number of right turns (1)
    bin_counts = {}
    for bitstring, count in counts.items():
        num_ones = bitstring.count('1')
        bin_counts[num_ones] = bin_counts.get(num_ones, 0) + count

    return bin_counts


# Two layer box, (3 pegs, 4 bins)

def galton_two_layer(shots=1000):
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.h(1)
    qc.measure(0, 0)

    # Finish me!