import numpy as np
from scipy.stats import norm, chisquare, ttest_1samp, chi2
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
            "w1" for 1-D Wasserstein-1 (or Earth Mover's) distance,
            "hel" for Hellinger distance,
            "bcd" for Bhattacharyya Distance,
            "chi" for Chi-Squared Statistic (NOT a distance metric, should be paired with chi-squared test).
        **target_distribution_params: Additional parameters for the target distribution, namely:
            "p" for Gaussian and Binomial (n is inferred)
            "lamda" for Exponential (dw spelt wrong as is keyword I'm not that dumb)
            none for Hadamard Quantum Walk (this is the default, or binomial with p=0.5).

    Returns:
        float: The calculated distance between the observed and target distributions.
    """


    #### STEP 1: Calculate Observed Probabilities ####
    observed_probs = {outcome: count / sum(result.values()) for outcome, count in result.items()} # get observed probability of each outcome


    #### STEP 2: Calculate Target Distribution Probabilities ####
    num_bins = len(result) # number of outcomes
    num_layers = num_bins - 1 # bins = n+1, so layers is bins-1
    target_probs = {}

    # calculating target distribution probabilities based on parameters (i.e. discretising):

    # Hadamard Walk (i.e. Binomial with p = 0.5)
    if target_distribution == "hadamard_walk": # Hadamard walk is a special case where p = 0.5 so use the binomial distribution
        p = 0.5
        for outcome in range(num_bins):
            target_probs[outcome] = math.comb(num_layers, outcome) * (p ** outcome) * ((1 - p) ** (num_layers - outcome))

    # Gaussian (Normal) Distribution      
    elif target_distribution == "gaussian":
        p = target_distribution_params["p"]
        mu = num_layers * p # Mean is np with p = 0.5
        sigma = np.sqrt(num_layers * p * (1 - p)) # Std dev is root(np(1-p))
        for outcome in range(num_bins):
            target_probs[outcome] = norm.pdf(outcome, mu, sigma) # get pdf for each outcome (discretising)

    # Exponential Distribution        
    elif target_distribution == "exponential":
        lamda = target_distribution_params["lamda"]
        for outcome in range(num_bins):
            target_probs[outcome] = lamda * np.exp(-1 * lamda * outcome) # pdf for exp dist. is lambda * exp(-lambda * x)

    # Binomial Distribution        
    elif target_distribution == "binomial":
        p = target_distribution_params["p"]
        for outcome in range(num_bins): 
            target_probs[outcome] = math.comb(num_layers, outcome) * (p ** outcome) * ((1 - p) ** (num_layers - outcome)) 
    
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
    
    
    #### STEP 4: Return Distance Calculated ####
    return distance

def significance_test(distances_list, test_type="t", significance_level=0.05, n=None):
    """
    For a list of distances calculated above, performs a t-test to see if the average distance from the 
    expected distribution is statistically significant to the given significance level.

    To account for stochastic uncertainty, can call distribution_distance() with QGB outputs, and average the distance metric. 
    For N runs: D_bar = 1/N * sum(D)
    Can also compute std error of the distance metric values.
    For N runs: s_D = sqrt(1/(N-1) * sum(D - D_bar)**2)
    Assuming the number of runs is high enough for the Central Limit Theorem to apply (ideally N > 30), we can then perform a t-test
    to determine if the average distance we observe (which controls for stochastic uncertainty through averaging over many runs)
    is statistically significant.
    H_0: mu_D = 0
    H_1: mu_D != 0
    t = (D_bar - 0) / (s_D / sqrt(N))
    Uses scipy.stats.ttest_1samp, which takes a list of observations and the expected mean under H_0 (0 here), 
    with N-1 degrees of freedom (handled by scipy), to estimate the p-value, and compares to the chosen significance level.

    Can also use chi-squared statistic in a chi-squared test.

    Args:
        distances_list (list): A list of N distance metrics as calculated by distribution_distance()
        test_type (str): Specifies the type of test to perform:
            "t" by default to perform a t-test. This has the above characteristics.
            "chi" to perform a chi-squared test (should be paired only with a distances_list containing
                chi-squared statistics, result will be nonsensical otherwise).
        significance_level (float): significance level for t- or chi-squared test, with which to
            assess p-value.
        n (int): number of layers in Quantum Galton Board (used in chi-squared test only, may be omitted for t-test).

    Returns:
        result_str (str): a string giving the p-value and test result.
    """

    #### T-test ####
    if test_type == "t":
        results = ttest_1samp(a=distances_list, popmean=0) # under null hypothesis, we assume population mean of distances is 0
        statistic = results.statistic
        p_value = results.pvalue
        reject_null = p_value <= significance_level
        if reject_null:
            conclusion = "Given that the p-value is <= the significance level, we reject the Null Hypothesis that" \
                "the population mean of the distances calculated is 0. Thus, we conclude that the distance between the observed" \
                "and target distributions is statistically significant at this significance level."
        else:
            conclusion = "Given that the p-value is > the significance level, we accept the Null Hypothesis that" \
                "the population mean of the distances calculated is 0. Thus, we conclude that the distance between the observed" \
                "and target distributions is not statistically significant at this significance level."
            
    
    #### Chi-Squared Test ####
    elif test_type == "chi":
        # Note: we cannot average over multiple chi-squared statistics as the result will not be chi-squared distributed, but rather
        # a scaled chi-squared, or gamma distribution. We can, however, directly add them. If N chi-squared statistics have d degrees
        # of freedom each, the result of the addition is chi-squared distributed with degrees of freedom N x d.
        # We opt for direct addition over Fisher's combined statistic as we have access to raw statistics and p-values.
        statistic = sum(distances_list) # as above
        dof = len(distances_list) * n # dof = N x d, each chi-square has dof of n as dof = bins - 1 = n + 1 - 1 = n
        p_value = chi2.sf(statistic, dof) # using the survival function to compute P(chi-square >= X**2)
        reject_null = p_value <= significance_level
        if reject_null:
            conclusion = "Given that the p-value is <= the significance level, we reject the Null Hypothesis that" \
                "the observed distribution follows the target distribution. Thus, we conclude that the distance between the observed" \
                "and target distributions is statistically significant at this significance level."
        else:
            conclusion = "Given that the p-value is > the significance level, we accept the Null Hypothesis that" \
                "the observed distribution follows the target distribution. Thus, we conclude that the distance between the observed" \
                "and target distributions is not statistically significant at this significance level."


    #### Construct result_str ####
    result_str = f"Test Statistic: {statistic}\nP-Value: {p_value}\nSignificance Level: {significance_level}\n" 
    result_str += conclusion

    #### Return result_str ####
    return result_str
