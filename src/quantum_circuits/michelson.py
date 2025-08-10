from qiskit import QuantumCircuit
from .common import run_interferometer

def michelson_builder(phi):
    qc = QuantumCircuit(1, 1)
    qc.x(0)       # photon in mode 0
    qc.h(0)       # first BS
    qc.rz(phi, 0) # first pass
    qc.h(0)       # reflection
    qc.rz(phi, 0) # second pass
    qc.h(0)       # recombine
    return qc

def michelson_circuit(intervals=10, shots=1000, plot=False):
    return run_interferometer(michelson_builder, intervals, shots, plot, "Michelson")
