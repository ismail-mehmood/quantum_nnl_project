# Quantum MZI

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

def mzi_circuit(intervals=10, shots=1000):
    phases = np.linspace(0, 2*np.pi, intervals)
    probs = []

    for phi in phases:

        qc = QuantumCircuit(1, 1)

        # Input: |1> (photon in mode 0)
        qc.x(0)

        # First beam splitter
        qc.h(0)

        # Phase shift
        qc.rz(phi, 0)

        # Second beam splitter
        qc.h(0)

        # Measurement
        qc.measure(0, 0)

        # Simulate
        sim = AerSimulator()
        tqc = transpile(qc, sim)
        result = sim.run(tqc, shots=shots).result()
        counts = result.get_counts()

        p0 = counts.get('0', 0) / shots
        p1 = counts.get('1', 0) / shots
        probs.append([p0, p1])

    return phases, probs
