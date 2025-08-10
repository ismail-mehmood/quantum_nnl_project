import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom, norm, expon, laplace

def plot_bins(bin_counts, n, shots=None, overlay=None, title="Galton Box", scale=0):

    bins = np.arange(n + 1) # get bin count
    counts = [bin_counts.get(b, 0) for b in bins] # retrieve results

    plt.bar(bins, counts, alpha=0.7, label="Quantum Results")

    if overlay is not None:
        if shots is None:
            raise ValueError("Parameter 'shots' must be provided for overlay plots.")

        if overlay.lower() == "binomial":
            p = 0.5 # Adjust for other gates, potential fun here!
            expected = shots * binom.pmf(bins, n, p)
            plt.plot(bins, expected, "ro--", label="Binomial Expected")

        elif overlay.lower() == "gaussian":
            # Gaussian approximation to binomial
            mu = n * 0.5 # mu = np
            sigma = np.sqrt(n * 0.5 * 0.5) # sigma = root(np(1 - p))
            # Probability density function
            expected = shots * norm.pdf(bins, mu, sigma)
            plt.plot(bins, expected, "go--", label="Gaussian Approx.")

        elif overlay.lower() == "exponential":
            
            # ***Discrete*** exponential (same logic as target_exponential)
            probs = np.exp(-bins / scale)
            probs /= np.sum(probs)  # normalize to sum to 1 over discrete bins
            expected = probs * shots

            # Plot
            plt.plot(bins, expected, "ro--", label="Exponential Approx.")

        elif overlay.lower() == "laplace":

            # Double sided Exponential Probability Density Function
            # Discrete Laplace (two-sided exponential centered at n/2)
            probs = np.exp(-np.abs(bins - n/2) / scale)
            probs /= np.sum(probs)  # normalize
            expected = probs * shots

            # Plot
            plt.plot(bins, expected, "ro--", label="Laplacian Approx.")

        else:
            raise ValueError("Unsupported overlay option: choose 'binomial' or 'gaussian'")

    # Pretty plot
    plt.title(str(n) + "-Layer " + title)
    plt.xlabel("Bin index (number of 1s)")
    plt.ylabel("Counts")
    plt.xticks(bins)
    plt.grid(True, linestyle="--", alpha=0.3)
    if overlay:
        plt.legend()
    plt.tight_layout()
    plt.show()

def plot_fringe(phases, results, labels=None, title="Mach–Zehnder Interference"):
    """
    Plot Mach–Zehnder interferometer output probabilities vs phase.

    Parameters:
    -----------
    phases : array-like
        List/array of phase angles (radians), e.g. np.linspace(0, 2*np.pi, 50)
    results : list of array-like
        Each element is a 2-element array/list [P0, P1] for output mode probabilities.
        E.g. results[i] = [prob_mode0, prob_mode1] for phase[i].
    labels : list of str, optional
        Labels for output modes (default: ["Mode 0", "Mode 1"]).
    title : str, optional
        Plot title.
    """
    results = np.array(results)
    probs_mode0 = results[:, 0]
    probs_mode1 = results[:, 1]

    if labels is None:
        labels = ["Output Mode 0", "Output Mode 1"]

    # Plot
    plt.figure(figsize=(7, 4))
    plt.plot(phases, probs_mode0, "o-", label=labels[0])
    plt.plot(phases, probs_mode1, "s-", label=labels[1])

    # Styling
    plt.xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], ["0", "π/2", "π", "3π/2", "2π"])
    plt.xlabel("Phase Shift (φ)")
    plt.ylabel("Probability")
    plt.title(title)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
