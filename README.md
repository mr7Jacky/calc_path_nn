# calc_path_nn

This project developed methods to calculate the importances for neural network pruning. This project is based on the repo [shrinkbench](https://gitlab.bucknell.edu/tjs030/shrinkbench-research)

Two methods includes are
1. Bellman Equation: Select the most weighted path for each class iteratively until reaches the threshold and remove the rest weights
3. Cumulative Weights: Calculate the weights on each layer cumulatively with regularization and pruning based on these weights.

