import numpy as np
from scipy.stats import norm
import math

def distribution_distance(result, target_distribution="hadamard_walk", metric="tvd", **target_distribution_params):
    """
    Calculates the distance between an observed distribution of Quantum Galton Board results and a target distribution.
    Args:
        result (dict): A dictionary where keys are bin numbers and values are the frequency.
        target_distribution (str): The target distribution type, either "gaussian", "exponential", "binomial", or "hadamard_walk".
        metric (str): The distance metric to use:
            "tvd" for total variation distance,
            "kl" for Kullback-Leibler divergence,
            "js" for Jensen-Shannon distance,
            "w1" for 1-D Wasserstein-1 (or Earth Mover's) distance.
        **target_distribution_params: Additional parameters for the target distribution, namely:
            "p" for Gaussian and Binomial (n is inferred)
            "lamda" for Exponential (dw spelt wrong as is keyword I'm not that dumb)
            none for Hadamard Quantum Walk (this is the default, or binomial with p=0.5).
    Returns:
        float: The calculated distance between the observed and target distributions.
    To account for stochastic uncertainty, can call this function with multiple runs, and average the distance metric. 
    For N runs: D_bar = 1/N * sum(D)
    Can also compute std error of the distance metric values.
    For N runs: s_D = sqrt(1/(N-1) * sum(D - D_bar)**2)
    Assuming the number of runs is high enough for the Central Limit Theorem to apply (ideally N > 30), we can then perform a t-test
    to determine if the average distance we observe (which controls for stochastic uncertainty through averaging over many runs)
    is statistically significant.
    H_0: mu_D = 0
    H_1: mu_D != 0
    t = (D_bar - 0) / (s_D / sqrt(N))
    Can use scipy.stats.ttest_1samp, which takes a list of observations and the expected mean under H_0 (0 here), 
    with N-1 degrees of freedom (handled by scipy), to estimate the p-value, and compare to the chosen significance level.
    """


    #### STEP 1: Calculate Observed Probabilities ####
    observed_probs = {outcome: count / sum(result.values()) for outcome, count in result.items()} # get observed probability of each outcome



    #### STEP 2: Calculate Target Distribution Probabilities ####
    num_bins = result.keys().__len__() # number of outcomes
    num_layers = num_bins - 1 # bins = n+1, so layers is bins-1
    target_probs = {}

    # calculating target distribution probabilities based on parameters (i.e. discretising)
    if target_distribution == "hadamard_walk": # Hadamard walk is a special case where p = 0.5 so use the binomial distribution
        p = 0.5
        for outcome in range(num_bins):
            target_probs[outcome] = math.comb(num_layers, outcome) * (p ** outcome) * ((1 - p) ** (num_layers - outcome))
    elif target_distribution == "gaussian":
        p = target_distribution_params["p"]
        mu = num_layers * p # Mean is np with p = 0.5
        sigma = np.sqrt(num_layers * p * p) # Std dev is root np(1-p)
        for outcome in range(num_bins):
            target_probs[outcome] = norm.pdf(outcome, mu, sigma) # get pdf for each outcome (discretising)
    elif target_distribution == "exponential":
        lamda = target_distribution_params["lamda"]
        for outcome in range(num_bins):
            target_probs[outcome] = lamda * np.exp(-1 * lamda * outcome) # pdf for exp dist. is lambda * exp(-lambda * x)
    elif target_distribution == "binomial":
        p = target_distribution_params["p"]
        for outcome in range(num_bins): 
            target_probs[outcome] = math.comb(num_layers, outcome) * (p ** outcome) * ((1 - p) ** (num_layers - outcome)) 
    
    # normalising target probabilities
    total_pdf = sum(target_probs.values()) # get sum of pdf values
    for outcome in range(num_bins):
        target_probs[outcome] /= total_pdf # normalise so probs sum to 1 for comparability



    #### STEP 3: Compute Distance Metric ####
    if metric == "tvd": # total variation distance
        distance = 0.5 * sum(abs(observed_probs[outcome] - target_probs[outcome]) for outcome in range(num_bins)) # calculate TVD
    elif metric == "kl": # Kullback-Leibler divergence
        distance = sum(observed_probs[outcome] * np.log(observed_probs[outcome] / target_probs[outcome]) 
                        if observed_probs[outcome] > 0 else 0 # controls for observation of 0 in given bin
                        for outcome in range(num_bins))
    elif metric == "js": # Jensen-Shannon distance
        M = {outcome: 0.5 * (observed_probs[outcome] + target_probs[outcome]) for outcome in range(num_bins)}
        P1 = sum(observed_probs[outcome] * np.log(observed_probs[outcome] / M[outcome]) for outcome in range(num_bins))
        P2 = sum(target_probs[outcome] * np.log(target_probs[outcome] / M[outcome]) for outcome in range(num_bins))
        distance = np.sqrt(0.5 * (P1 + P2)) # JS = 0.5 * (KL(P1 || M) + KL(P2 || M)), sqrt for distance
    elif metric == "w1": # 1-D Wasserstein-1 distance
        observed_probs_list = [observed_probs[outcome] for outcome in range(num_bins)]
        target_probs_list = [target_probs[outcome] for outcome in range(num_bins)]
        observed_cdf = np.cumsum(observed_probs_list)
        target_cdf = np.cumsum(target_probs_list)
        distance = sum(abs(target_cdf[i] - observed_cdf[i]) for i in range(num_bins)) # Wasserstein uses CDFs for distance
    
    
    #### STEP 4: Return Distance Calculated ####
    return distance

print(distribution_distance({0: 0, 1: 150, 2:400, 3:200, 4:250}, target_distribution="exponential", metric="w1", lamda=2))