# Iterative-Dichotomiser-3
ID3 Algorithm function with Entropy and Information Gain measures for categorical data. The function returns: 1) The decision tree rules. 2) The total number of rules.

* Xdata = Dataset Attributes
* ydata = Dataset Target
* pre_pruning = "none" or "chi_2". If "chi_2" is selected then a pre-pruning method based on a Chi Squared test is performed. If the table is in the 2x2 format and has less than 10,000 examples then a Fisher Exact test is performed instead.
* Finnaly a prediction function - prediction_dt_id3( ) - is also included.
