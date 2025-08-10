import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt

def run_interferometer(build_circuit, intervals=10, shots=1000, plot=False, label=None):
    """
    Runs an interferometer simulation.

    Parameters:
        build_circuit: function(phase: float) -> QuantumCircuit
        intervals: number of phase points from 0 to 2π
        shots: number of measurement shots
        plot: if True, adds result to current matplotlib plot
        label: label for plotting legend
    Returns:
        phases (np.ndarray), probs (np.ndarray with columns [P0, P1])
    """
    phases = np.linspace(0, 2*np.pi, intervals)
    probs = []

    sim = AerSimulator()

    for phi in phases:
        qc = build_circuit(phi)
        qc.measure(0, 0)

        tqc = transpile(qc, sim)
        result = sim.run(tqc, shots=shots).result()
        counts = result.get_counts()

        p0 = counts.get('0', 0) / shots
        p1 = counts.get('1', 0) / shots
        probs.append([p0, p1])

    probs = np.array(probs)

    if plot:
        plt.plot(phases, probs[:, 0], label=f"{label} (P0)")
        plt.plot(phases, probs[:, 1], '--', label=f"{label} (P1)")

    return phases, probs
