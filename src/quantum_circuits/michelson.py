from qiskit import QuantumCircuit
from .common import run_interferometer

def michelson_builder(phi):
    qc = QuantumCircuit(1, 1)
    # Start in |0> by default (why was this |1>?)
    qc.h(0)          # first BS
    qc.rz(2 * phi, 0)    # phase shift in one arm
    # no gate here for reflection; mirrors add phase only, no mode mixing
    qc.h(0)          # recombine BS
    qc.measure(0, 0)
    return qc


def michelson_circuit(intervals=10, shots=1000, plot=False):
    return run_interferometer(michelson_builder, intervals, shots, plot, "Michelson")
