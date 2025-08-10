import numpy as np
from scipy.stats import norm, chisquare, ttest_1samp, chi2, shapiro
import math
from src.quantum_walk import hadamard_walk_probs

def distribution_distance(result, target_distribution="hadamard_walk", metric="tvd", input_type="counts", **target_distribution_params):
    """
    Calculates the distance between an observed distribution of Quantum Galton Board results and a target distribution.

    Args:
        result (dict): A dictionary where keys are bin numbers and values are the frequency.
        target_distribution (str): The target distribution type, either "gaussian", "exponential", "binomial", "hadamard_walk" or "laplace".
        metric (str): The distance metric to use:
            "tvd" for total variation distance,
            "kl" for Kullback-Leibler divergence,
            "js" for Jensen-Shannon distance,
            "w1" for 1-D Wasserstein-1 (or Earth Mover's) distance,
            "hel" for Hellinger distance,
            "bcd" for Bhattacharyya Distance,
            "chi" for Chi-Squared Statistic (NOT a distance metric, should be paired with chi-squared test).
        **target_distribution_params: Additional parameters for the target distribution, namely:
            "p" for Gaussian and Binomial (n is inferred)
            "lamda" for Exponential (dw spelt wrong as is keyword I'm not that dumb)
            none for Chi-squared
            "b" and "mu" for Laplace (note b = scale = 1/lambda)
            "n_dec", or number of decisions for Hadamard Quantum Walk

    Returns:
        float: The calculated distance between the observed and target distributions.
    """


    #### STEP 1: Calculate Observed Probabilities ####
    if input_type == "counts":
        observed_probs = {outcome: count / sum(result.values()) for outcome, count in result.items()} # get observed probability of each outcome
    elif input_type == "probs":
        observed_probs = result
    else:
        raise Exception("input_type must be 'counts' or 'probs'")

    #### STEP 2: Calculate Target Distribution Probabilities ####
    num_bins = len(result)  # number of outcomes
    num_layers = num_bins - 1  # bins = n+1, so layers is bins-1
    target_probs = {}

    # This ensures sinusoidal/cosine distributions know the x-values
    x_values = np.array(target_distribution_params.get("x_values", np.arange(num_bins)), dtype=float)


    # Hadamard Walk 
    if target_distribution == "hadamard_walk": 
        n_dec = target_distribution_params["n_dec"]
        probs = hadamard_walk_probs(n_dec) # use fn defined in quantum_walk.py
        for outcome in range(num_bins):
            target_probs[outcome] = probs[outcome]

    # Gaussian (Normal) Distribution
    elif target_distribution == "gaussian":
        p = target_distribution_params["p"]
        if p < 0 or p > 1:
            raise Exception("Invalid p. Must satisfy 0 <= p <= 1.")
        mu = num_layers * p
        sigma = np.sqrt(num_layers * p * (1 - p))
        for outcome in range(num_bins):
            target_probs[outcome] = norm.pdf(outcome, mu, sigma)

    # Exponential Distribution        
    elif target_distribution == "exponential":
        lamda = target_distribution_params["lamda"]
        if lamda <= 0:
            raise Exception("Invalid lambda value. Must satisfy lamda > 0.")
        for outcome in range(num_bins):
            target_probs[outcome] = lamda * np.exp(-lamda * outcome)

    # Binomial Distribution        
    elif target_distribution == "binomial":
        p = target_distribution_params["p"]
        if p < 0 or p > 1:
            raise Exception("Invalid p. Must satisfy 0 <= p <= 1.")
        for outcome in range(num_bins): 
            target_probs[outcome] = math.comb(num_layers, outcome) * (p ** outcome) * ((1 - p) ** (num_layers - outcome)) 

    # Laplace Distribution
    elif target_distribution == "laplace":
        b = target_distribution_params["b"]
        if b <= 0:
            raise Exception("Invalid b. Must satisfy b > 0.")
        mu = target_distribution_params["mu"]
        for outcome in range(num_bins):
            target_probs[outcome] = (1/(2*b)) * np.exp(-abs(outcome - mu) / b)

    # Sinusoidal (Ramsey-like fringes)
    elif target_distribution == "sinusoidal":
        A = target_distribution_params.get("A", 0.5)
        offset = target_distribution_params.get("offset", 0.5)
        freq = target_distribution_params.get("freq", 1.0)
        phase_shift = target_distribution_params.get("phase_shift", 0.0)
        y_vals = offset + A * np.sin(freq * x_values + phase_shift)
        y_vals = np.clip(y_vals, 0, None)
        y_vals /= np.sum(y_vals)
        target_probs = {outcome: y_vals[i] for i, outcome in enumerate(range(num_bins))}

    # Cosine Squared (MZI intensity profile)
    elif target_distribution == "cosine_squared":
        freq = target_distribution_params.get("freq", 0.5)
        phase_shift = target_distribution_params.get("phase_shift", np.pi/2)
        y_vals = np.cos(freq * x_values + phase_shift) ** 2
        y_vals /= np.sum(y_vals)
        target_probs = {outcome: y_vals[i] for i, outcome in enumerate(range(num_bins))}

    # Double Frequency (Michelson-like)
    elif target_distribution == "double_frequency":
        freq = target_distribution_params.get("freq", 2.0)
        phase_shift = target_distribution_params.get("phase_shift", 0.0)
        y_vals = np.cos(freq * x_values + phase_shift) ** 2
        y_vals /= np.sum(y_vals)
        target_probs = {outcome: y_vals[i] for i, outcome in enumerate(range(num_bins))}

    else:
        raise Exception("Target Distribution provided is not recognised or supported. Check docstring.")
        
    # normalising target probabilities:

    total_pdf = sum(target_probs.values()) # get sum of pdf values
    for outcome in range(num_bins):
        target_probs[outcome] /= total_pdf # normalise so probs sum to 1 for comparability


    #### STEP 3: Compute Distance Metric ####

    # Total Variation Distance
    if metric == "tvd": 
        distance = 0.5 * sum(abs(observed_probs[outcome] - target_probs[outcome]) for outcome in range(num_bins)) # calculate TVD

    # Kullback-Leibler Divergence
    elif metric == "kl":
        distance = sum(observed_probs[outcome] * np.log(observed_probs[outcome] / target_probs[outcome]) 
                        if observed_probs[outcome] > 0 else 0 # controls for observation of 0 in given bin
                        for outcome in range(num_bins))

    # Jensen-Shannon Distance    
    elif metric == "js": 
        M = {outcome: 0.5 * (observed_probs[outcome] + target_probs[outcome]) for outcome in range(num_bins)}
        P1 = sum(observed_probs[outcome] * np.log(observed_probs[outcome] / M[outcome]) 
                if observed_probs[outcome] > 0 else 0 # controls for observation of 0 in given bin
                for outcome in range(num_bins))
        P2 = sum(target_probs[outcome] * np.log(target_probs[outcome] / M[outcome]) 
                if target_probs[outcome] > 0 else 0 # controls for observation of 0 in given bin
                for outcome in range(num_bins))
        distance = np.sqrt(0.5 * (P1 + P2)) # JS = 0.5 * (KL(P1 || M) + KL(P2 || M)), sqrt for distance

    # 1-D Wasserstein-1 Distance
    elif metric == "w1": 
        observed_probs_list = [observed_probs[outcome] for outcome in range(num_bins)]
        target_probs_list = [target_probs[outcome] for outcome in range(num_bins)]
        observed_cdf = np.cumsum(observed_probs_list)
        target_cdf = np.cumsum(target_probs_list)
        distance = sum(abs(target_cdf[i] - observed_cdf[i]) for i in range(num_bins)) # Wasserstein uses CDFs for distance

    # Hellinger Distance
    elif metric == "hel": 
        distance = 1/np.sqrt(2) * np.sqrt(sum((np.sqrt(target_probs[outcome]) 
                        - np.sqrt(observed_probs[outcome]))**2 for outcome in range(num_bins))) 
        
    # Bhattacharyya Distance    
    elif metric == "bcd": 
        distance = -1 * np.log(sum(np.sqrt(observed_probs[outcome] * target_probs[outcome]) 
                        for outcome in range(num_bins))) # -logBC for distance metric
        
    # Chi-Squared Statistic
    elif metric == "chi":
        observed_counts_list = [observed_probs[outcome] * sum(result.values()) for outcome in range(num_bins)]
        target_counts_list = [target_probs[outcome] * sum(result.values()) for outcome in range(num_bins)]
        distance = chisquare(observed_counts_list, target_counts_list).statistic

    else:
        raise Exception("Distance Metric provided is not recognised or supported. Check docstring.")
    
    
    #### STEP 4: Return Distance Calculated ####
    return distance

def significance_test(distances_list, test_type="t", significance_level=0.05, **test_params):
    """
    For a list of distances calculated above, performs a one-tailed t-test to see if the average distance from the 
    expected distribution is greater than a chosen acceptable threshold to a statistically significant degree, given 
    a significance level.

    To account for stochastic uncertainty, can call distribution_distance() with QGB outputs, and average the distance metric. 
    For N runs: D_bar = 1/N * sum(D)
    Can also compute std error of the distance metric values.
    For N runs: s_D = sqrt(1/(N-1) * sum(D - D_bar)**2)
    Assuming the number of runs is high enough for the Central Limit Theorem to apply (ideally N > 30), we can then perform a one-tailed
    t-test to determine if the average distance we observe (which controls for stochastic uncertainty through averaging over many runs)
    is statistically significant in terms of being greater than the user-defined acceptable threshold. In order to check the normality
    assumption, the Shapiro-Wilk test is performed and checked using a p-value of 0.05; if the distances are found to not be normally
    distributed, the t-test will continue but a warning will be thrown.
    H_0: mu_D <= threshold
    H_1: mu_D > threshold
    t = (D_bar - threshold) / (s_D / sqrt(N))
    Uses scipy.stats.ttest_1samp, which takes a list of observations and the expected mean under H_0 (set to threshold here), 
    with N-1 degrees of freedom (handled by scipy), to estimate the p-value, and compares to the chosen significance level.

    Can also use chi-squared statistic in a chi-squared test. Note that the combined chi-squared test is extremely harsh, and even the perfect simulator with shots
    will almost always fail it!

    Args:
        distances_list (list): A list of N distance metrics as calculated by distribution_distance()
        test_type (str): Specifies the type of test to perform:
            "t" by default to perform a one-tailed t-test. This has the above characteristics.
            "chi" to perform a chi-squared test (should be paired only with a distances_list containing
                chi-squared statistics, result will be nonsensical otherwise).
        significance_level (float): significance level for t- or chi-squared test, with which to
            assess p-value.
        **test_params: additional parameters for significance tests, namely:
            threshold (float): acceptable threshold for distance metric to be under for one-tailed t-test 
            n (int): number of layers in Quantum Galton Board (used in chi-squared test only, may be omitted for t-test).

    Returns:
        result_str (str): a string giving the p-value and test result.
    """

    # Robustness Check #
    if significance_level < 0 or significance_level > 1:
        raise Exception("Invalid significance level. Must be between 0 and 1.")

    #### T-test ####
    if test_type == "t":
        shapiro_p = shapiro(distances_list)[1]
        if shapiro_p < 0.05:
            print("Distance Metrics are insufficiently Normally distributed so t-test is unreliable. Consider performing more QGB runs.")
        else:
            print("Normality assumption is valid.")
        threshold = test_params["threshold"]
        if threshold < 0 or threshold > 1:
            raise Exception("Invalid threshold. Must be between 0 and 1.")
        mean = sum(distances_list) / len(distances_list)
        print(f"Mean Distance: {mean}")
        print(f"Standard Deviation: {np.sqrt(1/(len(distances_list)-1) * sum([(D - mean)**2 for D in distances_list]))}")
        results = ttest_1samp(a=distances_list, popmean=threshold, alternative="greater") 
        statistic = results.statistic
        p_value = results.pvalue
        reject_null = p_value <= significance_level
        if reject_null:
            conclusion = "Given that the p-value is <= the significance level, we reject the Null Hypothesis that " \
                "the population mean of the distances calculated is within the acceptable threshold. Thus, we conclude that the distance between the observed " \
                "and target distributions is statistically significant at this significance level."
        else:
            conclusion = "Given that the p-value is > the significance level, we accept the Null Hypothesis that " \
                "the population mean of the distances calculated is within the acceptable threshold. Thus, we conclude that the distance between the observed " \
                "and target distributions is not statistically significant at this significance level."
            
    
    #### Chi-Squared Test ####
    elif test_type == "chi":
        # Note: we cannot average over multiple chi-squared statistics as the result will not be chi-squared distributed, but rather
        # a scaled chi-squared, or gamma distribution. We can, however, directly add them. If N chi-squared statistics have d degrees
        # of freedom each, the result of the addition is chi-squared distributed with degrees of freedom N x d.
        # We opt for direct addition over Fisher's combined statistic as we have access to raw statistics and p-values.
        statistic = sum(distances_list) # as above
        print(f"Joint Chi-Squared Statistic: {statistic}")
        n = test_params["n"]
        dof = len(distances_list) * n # dof = N x d, each chi-square has dof of n as dof = bins - 1 = n + 1 - 1 = n
        print(f"Joint DoF: {dof}")
        p_value = chi2.sf(statistic, dof) # using the survival function to compute P(chi-square >= X**2)
        reject_null = p_value <= significance_level
        if reject_null:
            conclusion = "Given that the p-value is <= the significance level, we reject the Null Hypothesis that " \
                "the observed distribution follows the target distribution. Thus, we conclude that the distance between the observed " \
                "and target distributions is statistically significant at this significance level."
        else:
            conclusion = "Given that the p-value is > the significance level, we accept the Null Hypothesis that " \
                "the observed distribution follows the target distribution. Thus, we conclude that the distance between the observed " \
                "and target distributions is not statistically significant at this significance level."
            
    
    else:
        raise Exception("Test Type provided is not recognised or supported. Check docstring.")


    #### Construct result_str ####
    result_str = f"Test Statistic: {statistic}\nP-Value: {p_value}\nSignificance Level: {significance_level}\n" 
    result_str += conclusion

    #### Return result_str ####
    return result_str
