from qiskit import QuantumCircuit
from .common import run_interferometer

def mzi_builder(phi):
    qc = QuantumCircuit(1, 1)
    qc.x(0)       # photon in mode 0
    qc.h(0)       # first BS
    qc.rz(phi, 0) # phase shift
    qc.h(0)       # second BS
    return qc

def mzi_circuit(intervals=10, shots=1000, plot=False):
    return run_interferometer(mzi_builder, intervals, shots, plot, "MZI")
