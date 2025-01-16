# Solving the Bin Packing Problem (https://www.wikiwand.com/en/Bin_packing_problem) with first fit algorithm
from collections import defaultdict


def first_fit_decreasing(weights: list, max_weight: int) -> list:
    """
    An Implementation of "First Fit Decreasing Algorithm" for the "Bin Packing Problem"
    :param weights: A List of weights of items to fit
    :param max_weight: Maximum weight a Bin can hold
    :return: A list of lists, each list containing elements in that bin
    """
    bins, current_key, weights = defaultdict(list), 1, sorted(weights, reverse=True)

    for weight in weights:
        found = False
        for key in bins.keys():
            if sum(bins[key]) + weight > max_weight:
                found = False
                continue

            found = True
            bins[key].append(weight)
            break

        # No existing bin was able to hold the item OR no bins exist yet.
        if not found:
            bins[current_key].append(weight)
            current_key += 1

    return list(bins.values())


# Test example
weights = [4, 8, 1, 4, 2, 1]
max_weight = 10
result = first_fit_decreasing(weights, max_weight)
print(result)