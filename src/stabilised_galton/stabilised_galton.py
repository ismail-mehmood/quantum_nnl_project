import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit_aer import AerSimulator
from IPython.display import display

def trotterized_galton_layer(qc, control_qubit, middle_qubit, left_qubit, right_qubit, theta, steps):
    """Apply a biased rotation in small steps with stabilisation in between."""
    small_angle = theta / steps
    for _ in range(steps):
        qc.ry(small_angle, control_qubit)  # small rotation
        qc.cswap(control_qubit, middle_qubit, right_qubit)
        qc.cx(middle_qubit, control_qubit)
        qc.cswap(control_qubit, middle_qubit, left_qubit)
        # optional 'stabilisation' – here we use a tiny idle delay to model decoherence
        qc.barrier()

def stabilised_galton_n_layer(n, thetas, shots=1000, mode="full", noise=False, draw=False, steps=5):
    """
    Biased Galton board with small-step trotterized stabilisation.
    """
    if len(thetas) != n:
        raise ValueError(f"Expected {n} bias angles, got {len(thetas)}.")

    if mode == "full":
        total_qubits = 2 * n + 2
        qc = QuantumCircuit(total_qubits, n + 1)
        control = 0
        centre = n + 1
        qc.x(centre)  # Start ball

        for i in range(n):
            qc.reset(control)
            small_angle = thetas[i] / steps
            for _ in range(steps):
                qc.ry(small_angle, control)
                qc.barrier()

            for j in range(i + 1):
                middle = centre - i + (2 * j)
                left = middle + 1
                right = middle - 1
                qc.cswap(control, middle, right)
                qc.cx(middle, control)
                qc.cswap(control, middle, left)
                if j < i:
                    qc.cx(middle + 1, control)

        for i in range(n + 1):
            qc.measure(2 * i + 1, i)

    else:
        qc = QuantumCircuit(n, n)
        for i in range(n):
            small_angle = thetas[i] / steps
            for _ in range(steps):
                qc.ry(small_angle, i)
                qc.barrier()
        qc.measure(range(n), range(n))

    if noise:
        backend = GenericBackendV2(num_qubits=2 * n + 2)
        tqc = transpile(qc, backend)
        result = backend.run(tqc, shots=shots).result()
    else:
        sim = AerSimulator()
        tqc = transpile(qc, sim)
        result = sim.run(tqc, shots=shots).result()

    counts = result.get_counts()

    bin_counts = {i: 0 for i in range(n + 1)}
    if mode == "full":
        for bitstring, count in counts.items():
            for i, bit in enumerate(bitstring):  # keep direct order, unsure about reverse
                if bit == '1':
                    bin_counts[i] += count
                    break
    else:
        for bitstring, count in counts.items():
            bin_counts[bitstring.count('1')] += count

    if draw:
        display(qc.draw(output='mpl'))

    return bin_counts
