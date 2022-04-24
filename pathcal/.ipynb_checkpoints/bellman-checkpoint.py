import numpy as np
from .layers import * 

def importances_to_weights(importances):
    weights = []
    for layer in importances.keys():
        if isLinear(layer):
            weights += [importances[layer]['weight']]
    return weights

def mask_list_to_dict(importances, mask):
    ret_mask = {}
    i = 0
    for layer in importances.keys():
        if isLinear(layer):
            ret_mask[layer] = mask[i]
            i += 1
    return ret_mask

def longest_new_wrapper(importances, fraction):
    # Convert to list
    weights = importances_to_weights(importances)
    # Number of layers
    NUM_LAYER = len(weights)
    # Max length of weight
    MAX_LEN = max([l.shape[0] for l in weights])
    # Store used arcs
    used = []
    # Store the longest weight upon current level and node
    longest = np.ones((NUM_LAYER+1,MAX_LEN)) * -1
    longest[NUM_LAYER,:] = 1
    # Store the longes path upon current level and node
    longest_path = longest.tolist()
    longest_path[NUM_LAYER] = [[(NUM_LAYER,i,0)] for i in range(MAX_LEN)]
    # Mask of the weight
    mask = []
    mask_count = 0
    mask_size = 0
    for l in weights:
        mask += [np.zeros_like(l)]
        mask_size += l.flatten().shape[0]
    for i in range(weights[0].shape[-1]):
        m, path = longest_new(weights, 0, i, new=True)
        for L, v, i in path[:-1]:
            #Check percentage pruned
            if mask_count/mask_size < fraction:
                mask[L][i,v] = 1
                mask_count += 1
        # print(f'Longest path for node {i} is {m:.2f} with path {path}')
    return mask_list_to_dict(importances, mask)

def longest_new(layers, L, v, new):
    if longest[L,v] != -1 and new == False:
        # if there is max value calculated and not new path required
        # we return the value and path from history
        return longest[L,v], longest_path[L][v]
    weights = layers[L][:,v]
    m = -1
    arc = -1
    path = []
    if L == len(layers) - 1 and new:
        # if reach top and still need new
        # fetch the max unused arcs
        for i in range(len(weights)):
            # we need to ensure the current arc is unused and 
            # check for the max weight in unused arcs.
            if (L,v,i) not in used and weights[i] > m:
                m = weights[i]
                arc = (L,v,i)
                path = [(L,v,i)]
        # if nothing is found 
        # we return -1 on weights to make sure this arc is not used
        if m == -1: return m, path
    elif new:
        for i in range(len(weights)):
            # picking a new arc in this layer
            if (L,v,i) not in used:
                cur_m, cur_path = longest_new(layers, L+1, i, new=False)
            # picking a new arc in later layer
            else: 
                cur_m, cur_path = longest_new(layers, L+1, i, new=True)
            cur = weights[i] * cur_m
            if cur > m:
                m = cur
                arc = (L,v,i)
                path = [(L,v,i)]
                p = cur_path
    else:
        # Iterater through all arcs
        for i in range(len(weights)):
            cur_m, cur_path = longest_new(layers, L+1, i, new=False)
            cur = weights[i] * cur_m
            if cur > m:
                m = cur
                arc = (L,v,i)
                path = [(L,v,i)]
                p = cur_path
    # Calculate the path to current level
    path += p
    # Add the arc to used
    used.append(arc)
    # There are some cases where m is not the max value return
    # We do not replace the max value in longest
    if m > longest[L,v]:
        longest[L,v] = m
        longest_path[L][v] = path
    return m, path