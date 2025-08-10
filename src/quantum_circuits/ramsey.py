from qiskit import QuantumCircuit
from .common import run_interferometer

def ramsey_builder(phi):
    qc = QuantumCircuit(1, 1)
    qc.h(0)       # π/2 pulse
    qc.rz(phi, 0) # free evolution
    qc.h(0)       # π/2 pulse
    return qc

def ramsey_circuit(intervals=10, shots=1000, plot=False):
    return run_interferometer(ramsey_builder, intervals, shots, plot, "Ramsey")
