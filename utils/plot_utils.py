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
            
            # Exponential Probability Density Function
            expected = shots * expon.pdf(bins, scale=scale)

            # Plot
            plt.plot(bins, expected, "ro--", label="Exponential Approx.")

        elif overlay.lower() == "laplace":

            # Double sided Exponential Probability Density Function
            expected = shots * laplace.pdf(bins, scale=scale, loc=n/2)

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
