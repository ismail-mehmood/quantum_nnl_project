import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# Ramsey probability calculation
def ramsey_probability_phase(phi, shots=2000, noise=False, backend=None):
    qc = QuantumCircuit(1, 1)
    qc.h(0)             # First pi/2 pulse
    qc.rz(phi, 0)       # (Free evolution) phase shift
    qc.h(0)             # Second pi/2 pulse
    qc.measure(0, 0)

    if noise:
        tqc = transpile(qc, backend)
        result = backend.run(tqc, shots=shots).result()
    else:
        sim = AerSimulator()
        tqc = transpile(qc, sim)
        result = sim.run(tqc, shots=shots).result()

    counts = result.get_counts()
    prob_0 = counts.get('0', 0) / shots
    return prob_0

# Plot helper
def plot_ramsey_fringe(phases, ideal, noisy, t2=None, title="Ramsey Fringes"):
    ideal = np.array(ideal)
    noisy = np.array(noisy)

    plt.figure(figsize=(8, 5))
    plt.plot(phases, ideal, "b-", label="Ideal")
    plt.plot(phases, noisy, "ro-", label="Noisy")

    if t2 is not None:
        times = np.linspace(0, 1, len(phases))
        envelope = np.exp(-times / t2)
        damped = 0.5 + (ideal - 0.5) * envelope
        plt.plot(phases, damped, "g--", label=f"Expected T2 decay (T2={t2})")

    plt.xlabel("Phase (rad)")
    plt.ylabel("P(|0⟩)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
