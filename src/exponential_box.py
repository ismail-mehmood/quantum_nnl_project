import time
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.providers.fake_provider import GenericBackendV2
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from enum import Enum

class CircuitType(Enum):
    FULL = "full"
    FAST = "fast"

# Optimise for exponential and laplace
def target_exponential(n, scale=1.5):
    bins = np.arange(n + 1)
    probs = np.exp(-bins / scale)
    probs /= np.sum(probs)  # Normalise
    return probs

def target_laplace_distribution(n_bins, decay_factor=1.0):
    center = n_bins // 2
    bins = np.arange(n_bins + 1)
    probs = np.exp(-np.abs(bins - center) / decay_factor)
    probs /= np.sum(probs)  # Normalise
    return probs

# Least aquares regression with interrupts for my sanity
def loss_function_layerwise(params, n, shots, target_probs, plot=False, start_time=None, max_time=None, best_tracker=None):
    # Time check for forced termination, minimize runs its own internal iterations
    if start_time is not None and max_time is not None:
        if time.time() - start_time > max_time:
            print("\nTime limit reached. Forcing optimization to stop.")
            raise TimeoutError  # Interrupt minimize()

    counts = biased_galton_n_layer(n, shots=shots, thetas=params)
    sim_probs = np.array([counts[i] for i in range(n + 1)]) / shots
    loss = np.sum((sim_probs - target_probs) ** 2)

    # Update result tracker
    if best_tracker is not None:
        if loss < best_tracker["loss"]:
            best_tracker["loss"] = loss
            best_tracker["params"] = params.copy()

    # Live plotting during optimization
    if plot:
        plt.clf()
        bins = np.arange(n + 1)
        plt.bar(bins, sim_probs, alpha=0.6, label="Simulated")
        plt.plot(bins, target_probs, "ro--", label="Target Distribution")
        plt.title(f"Loss: {loss:.4f}")
        plt.xlabel("Bins")
        plt.ylabel("Probability")
        plt.legend()
        plt.pause(0.3)

    return loss

# Optimise
def optimise_layerwise(n=5, shots=2000, target="laplace", scale=1.5, decay=1.0, max_time=10):
    # Select target distribution
    if target == "exponential":
        target_probs = target_exponential(n, scale)
    elif target == "laplace":
        target_probs = target_laplace_distribution(n, decay)
    else:
        raise ValueError("Target must be 'exponential' or 'laplace'")

    # Initial guess
    x0 = np.linspace(np.pi/1.5, np.pi/6, n)
    bounds = [(0.01, np.pi)] * n

    plt.ion()
    start_time = time.time()
    best_tracker = {"loss": np.inf, "params": x0.copy()}

    try:
        res = minimize(loss_function_layerwise, x0,
                       args=(n, shots, target_probs, True, start_time, max_time, best_tracker),
                       method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 500})
    except TimeoutError:
        print("\nOptimisation stopped due to time limit.")
        res = None

    plt.ioff()
    plt.show()

    # Use best result (whether timeout occurred or not)
    final_params = best_tracker["params"]
    final_loss = best_tracker["loss"]

    print("\nBest Optimised thetas:", final_params)
    print("Best Loss:", final_loss)
    return final_params, final_loss


def biased_galton_n_layer(n, shots=1000, mode="full", thetas=None, noise=False):

    if thetas is None:
        thetas = [np.pi/4] * n

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

            # Apply biased rotation to control
            qc.ry(thetas[i], control)

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
            qc.ry(thetas[i], i)
        qc.measure(range(n), range(n))

    # Simulate
    if noise:
        backend = GenericBackendV2(num_qubits=2*n+2)
        tqc = transpile(qc, backend)
        job = backend.run(tqc)
        result = job.result()
    else:
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
