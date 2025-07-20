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

def galton_two_layer(shots=1000, mode="full"):

    if mode == "full":
        qc = QuantumCircuit(6, 3)  # 6 qubits, 3 classical bits

        # Qubit roles because apparently six numbers is too many for my brain:
        # q0 - control qubit
        # q1 - leftmost output peg
        # q2 - intermediate peg
        # q3 - input qubit (ball starts here)
        # q4 - right intermediate peg
        # q5 - rightmost output peg

        # Step 1: Initialize input ball in q3 (|000100⟩)
        qc.x(3)  # q3 = 1

        # First peg
        qc.h(0)                   # Superpose control qubit
        qc.cswap(0, 2, 3)         # Conditional swap: input to left
        qc.cx(3, 0)               # Update control if ball moved
        qc.cswap(0, 3, 4)         # Conditional swap: input to right

        # Reset control and prepare for second layer
        qc.reset(0)
        qc.h(0)

        # Second layer — peg 1 (left path)
        qc.cswap(0, 1, 2)         # Conditional swap: ball left
        qc.cx(2, 0)               # Update control

        # Second layer — peg 2 (right path)
        qc.cswap(0, 2, 3)         # Conditional swap: center to right
        qc.cx(3, 0)               # Update control
        qc.cswap(0, 3, 4)         # Continue to far-right
        qc.cx(4, 0)               # Update control
        qc.cswap(0, 4, 5)         # Final rightmost fall

        # Measure pegs: left = q1, middle = q3, right = q5
        qc.measure(1, 0)
        qc.measure(3, 1)
        qc.measure(5, 2)

    else:
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.h(1)
        qc.measure([0, 1], [0, 1]) # measure both qubits for right turn count

    # Simulate
    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Post-process: bin based on measurement result
    bin_counts = {0: 0, 1: 0, 2: 0}
    if mode == "full":
        for bitstring, count in counts.items():
            # bitstring order: c2 c1 c0 → [right, middle, left]
            left = bitstring[0]
            middle = bitstring[1]
            right = bitstring[2]

            if right == '1':
                bin_counts[2] += count
            elif middle == '1':
                bin_counts[1] += count
            elif left == '1':
                bin_counts[0] += count
    else:
        for bitstring, count in counts.items():
            num_ones = bitstring.count('1')
            bin_counts[num_ones] = bin_counts.get(num_ones, 0) + count

    return bin_counts


# N-Layer Galton Board based on above format

def galton_n_layer(n, shots=1000, mode="full"):

    if mode == "full":
        total_qubits = 2 * n + 2  # control + 2n + 1 pegs
        qc = QuantumCircuit(total_qubits, n + 1)

        control = 0

        # Start ball in the middle peg
        centre = n + 1
        qc.x(centre)

        # For each layer, peg is at index i (excluding control qubit)
        for i in range(n):

            # Reset control for next layer
            qc.reset(control)

            # Apply Hadamard to control
            qc.h(control)

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
            qc.h(i)
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
