# Iterative-Dichotomiser-3
ID3 Algorithm function with Entropy and Information Gain measures for categorical data. The function returns: 1) The decision tree rules. 2) The total number of rules.

* Xdata = Dataset Attributes
* ydata = Dataset Target
* pre_pruning = "none" or "chi_2". If "chi_2" is selected then a pre-pruning method based on Fisher Exact test for 2x2 contigency tables with samples < 10,000. Otherwise a Chi Squared test is performed instead.
