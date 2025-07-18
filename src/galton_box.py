from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from enum import Enum

class CircuitType(Enum):
    FULL = "full"
    FAST = "fast"

# One layer box, (1 peg, 2 bins)

def galton_one_layer(shots=1000, mode="full"):

    if mode == "full":
        qc = QuantumCircuit(4, 2)  # 4 qubits, but only 2 classical bits needed
        qc.x(2)                    # Input qubit |0100⟩
        qc.h(0)                    # Control qubit (superposition)
        qc.cswap(0, 1, 2)          # Swap left (1) with input (2)
        qc.cx(2, 0)                # Enforce control = 1 if input fell
        qc.cswap(0, 2, 3)          # Swap right (3) with input (2)
        qc.measure(1, 0)           # Measure left outcome
        qc.measure(3, 1)           # Measure right outcome
    else: 
        qc = QuantumCircuit(1, 1)  # 1 qubit, initialised to |0⟩
        qc.h(0)                    # Hadamard gate
        qc.measure(0, 0)           # Measure outcome

    # Simulate
    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Get bin index, by number of right turns (1)
    bin_counts = {0: 0, 1: 0}
    if mode == "full":
        for bitstring, count in counts.items():
            left = bitstring[0]   # Classical bit 0 → left peg (qubit 1)
            right = bitstring[1]  # Classical bit 1 → right peg (qubit 3)
            if right == '1':
                bin_counts[1] += count
            elif left == '1':
                bin_counts[0] += count
    else:
        for bitstring, count in counts.items():
            num_ones = bitstring.count('1')
            bin_counts[num_ones] = bin_counts.get(num_ones, 0) + count

    return bin_counts

# Two layer box, (3 pegs, 4 bins)

def galton_two_layer(shots=1000):
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.h(1)
    qc.measure([0, 1], [0, 1]) # measure both qubits for right turn count

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

# N-Layer Galton Board based on above format

def galton_n_layer(n, shots=1000):
    qc = QuantumCircuit(n, n)
    for i in range(0, n):
        qc.h(i)
    
    qc.measure(range(n), range(n)) # measure all qubits for right turn count

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