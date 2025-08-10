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

def loss_function_layerwise(params, n, shots, target_probs, target, plot=False, start_time=None, max_time=None, best_tracker=None):
    # Time check for forced termination
    if start_time is not None and max_time is not None:
        if time.time() - start_time > max_time:
            print("\nTime limit reached. Forcing optimization to stop.")
            raise TimeoutError

    thetas = params

    counts = biased_galton_n_layer(n, shots=shots, thetas=thetas, mode="full")
    sim_probs = np.array([counts.get(i, 0) for i in range(n + 1)]) / shots

    loss = np.sqrt(0.5 * np.sum((np.sqrt(target_probs) - np.sqrt(sim_probs))**2)) # hellinger distance

    # Update result tracker
    if best_tracker is not None:
        if loss < best_tracker["loss"]:
            best_tracker["loss"] = loss
            best_tracker["params"] = thetas.copy()

    # Live plotting during optimization
    if plot and (best_tracker is None or loss < best_tracker["loss"] * 1.02):
        plt.clf()
        bins = np.arange(n + 1)
        plt.bar(bins, sim_probs, alpha=0.6, label="Simulated")
        plt.plot(bins, target_probs, "ro--", label=f"Target {target.capitalize()}")
        plt.title(f"Loss: {loss:.4f}")
        plt.xlabel("Bins")
        plt.ylabel("Probability")
        plt.legend()
        plt.pause(0.1)

    return loss

def optimise_layerwise(n=5, shots=2000, target="exponential", scale=0.5, decay=1.0, max_time=120, multi_start=3):
    # Select target distribution
    if target == "exponential":
        target_probs = target_exponential(n, scale)
    elif target == "laplace":
        target_probs = target_laplace_distribution(n, decay)
    else:
        raise ValueError("Target must be 'exponential' or 'laplace'")

    plt.ion()
    start_time = time.time()
    best_tracker = {
        "loss": np.inf,
        "params": None
    }

    def loss_func(params):
        return loss_function_layerwise(
            params, n, shots, target_probs, target,
            plot=True, start_time=start_time, max_time=max_time, best_tracker=best_tracker
        )

    # Optimize all thetas directly
    num_thetas = n * (n + 1) // 2
    bounds = [(0.01, np.pi)] * num_thetas  # Safe range for thetas

    # Multi-start strategy: try multiple initial points
    best_loss = np.inf
    for _ in range(multi_start):
        x0 = np.random.uniform(np.pi / 4, 3 * np.pi / 4, num_thetas)  # Random initial thetas
        try:
            res = minimize(loss_func, x0, method="COBYLA", bounds=bounds, options={"maxiter": 500, "rhobeg": 0.5})
            if res.fun < best_loss:
                best_loss = res.fun
        except TimeoutError:
            print("\nOptimization stopped due to time limit.")
            break

    plt.ioff()
    plt.show()

    # Extract best tracked result
    final_thetas = best_tracker["params"]
    final_loss = best_tracker["loss"]

    print("\nOptimizer exit (best loss):", best_loss)
    print("Best Optimized thetas:", final_thetas)
    print("Best Loss:", final_loss)

    # Final plot
    final_counts = biased_galton_n_layer(n, shots=shots, thetas=final_thetas, mode="full")
    sim_probs = np.array([final_counts.get(i, 0) for i in range(n + 1)]) / shots
    plt.figure()
    bins = np.arange(n + 1)
    plt.bar(bins, sim_probs, alpha=0.6, label="Simulated")
    plt.plot(bins, target_probs, "ro--", label=f"Target {target.capitalize()} (Param={scale if target == 'exponential' else decay})")
    plt.title(f"Final Fit, Loss: {final_loss:.4f}")
    plt.xlabel("Bins")
    plt.ylabel("Probability")
    plt.legend()
    plt.show()

    return final_thetas, final_loss


def biased_galton_n_layer(n, shots=1000, mode="full", thetas=None, noise=False):

    if thetas is None:
        thetas = [np.pi/2] * (n * (n+1) // 2)

    if mode == "full":
        total_qubits = 2 * n + 2  # control + 2n + 1 pegs
        qc = QuantumCircuit(total_qubits, n + 1)

        control = 0

        # Start ball in the middle peg
        centre = n + 1
        qc.x(centre)

        # theta counter
        counter = 0

        # For each layer, peg is at index i (excluding control qubit)
        for i in range(n):
            # Every peg per layer, working right to left
            for j in range(i + 1):

                # Reset control for each peg
                qc.reset(control)

                # Apply biased rotation to control
                qc.ry(thetas[counter], control)
                counter += 1

                middle = centre - i + (2 * j) # Gets centre qubit relative to board position
                left = middle + 1
                right = middle - 1 

                # Controlled-SWAP: ball goes right
                qc.cswap(control, middle, right)

                # CNOT to entangle (centre → control)
                qc.cx(middle, control)

                # Controlled-SWAP: ball goes left
                qc.cswap(control, middle, left)

                # for all except first peg, corrective CNOT
                if j > 0:
                    qc.cx(middle+1, control)
            
            # resetting control for next layer:
            if i < n-1:
                qc.reset(control)

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
        job = backend.run(tqc, shots=shots)
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
            for i, bit in enumerate(bitstring):
                if bit == '1':
                    bin_counts[i] += count
                    break
    else:
        for bitstring, count in counts.items():
            num_ones = bitstring.count('1')
            bin_counts[num_ones] += count

    return bin_counts
