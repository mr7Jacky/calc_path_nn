import torch

"""
===============================================================
====              Calculate Paths Using Weights            ====
===============================================================
"""
def cal_linear_paths(prev_paths, weights, threshold=0):
    next_path = torch.tile(prev_paths, (weights.size()[0],)).reshape(weights.size())
    next_path[weights.abs() <= threshold] = 0
    next_path = next_path.sum(axis=-1)
    return next_path

def cal_conv_paths(prev_paths, weights, threshold=0):
    # Calculate paths at current layer
    cur_path = weights.reshape(len(weights),-1)
    cur_path = (cur_path.abs() <= threshold).sum(axis=-1)
    # Calculate sum of all previous paths
    if prev_paths == None:
        total_prev_paths = 1
    else:
        total_prev_paths = prev_paths.sum()
    # Calculate next paths
    next_path = cur_path * total_prev_paths
    return next_path
def expend_paths(prev_paths, weights):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    in_shape = weights.shape[-1]
    path_mat = torch.zeros((in_shape // len(prev_paths) , len(prev_paths))).to(device)
    path_mat[:,:] = prev_paths
    return path_mat.T.flatten()
"""
===============================================================
====              Calculate Flows Using Weights            ====
===============================================================
"""
def cal_linear_flows(prev_flows, weights, threshold=0):
    """
    This method is used for calculate flows for linear layer.
    param:
    prev_flows: the outputing flows from previous layer 
    weights: the weights connecting previous layer to current layer
    threshold: not currently used in this function
    """
    # duplicate and reshape to match the current weights shape for easier calculation
    next_flows = torch.tile(prev_flows, (weights.size()[0],)).reshape(weights.size()) * weights
    #next_flows[weights.abs() <= threshold] = 0
    # Sum over all the flows connect to current node
    next_flows = next_flows.sum(axis=-1)
    # print(next_flows)
    return next_flows

def cal_conv_flows(prev_flows, weights, threshold=0):
    """
    This method is used for calculate flows for conv layer.
    param:
    prev_flows: the outputing flows from previous layer 
    weights: the weights connecting previous layer to current layer
    threshold: not currently used in this function
    """
    # Calculate flowss at current layer
    cur_flows = weights.reshape(len(weights),-1)
    cur_flows = (cur_flows).sum(axis=-1)
    # Calculate sum of all previous flowss
    if prev_flows == None:
        total_prev_flows = 1
    else:
        total_prev_flows = prev_flows.sum()
    # Calculate next flowss
    next_flows = cur_flows * total_prev_flows
    return next_flows

def expend_flows(prev_flows, weights):
    """
    This method is used for reshape the flows when chaning from conv layer to linear layer
    Here we duplicate the flows from conv to match the input shape of linear layer
    param:
    prev_flows: the outputing flows from previous layer 
    weights: the weights connecting previous layer to current layer
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    in_shape = weights.shape[-1]
    flow_mat = torch.zeros((in_shape // len(prev_flows) , len(prev_flows))).to(device)
    flow_mat[:,:] = prev_flows
    return flow_mat.T.flatten()

def cal_linear_zeros(weights, threshold=0):
    return (weights.abs() <= threshold).sum()
"""
===============================================================
====              Calculate weighted paths                 ====
===============================================================
"""
def cal_linear_paths_weighted(prev_paths, weights, threshold=None, pos=False):
    if pos:
        weights = weights.abs()
    weights = weights.detach()
    next_path = torch.tile(prev_paths, (weights.size()[0],)).reshape(weights.size())
    if threshold is not None:
        weights[weights.abs() <= threshold] = 0
    next_path *= weights
    next_path = next_path.sum(axis=-1)
    return next_path

def cal_conv_paths_weighted(prev_paths, weights, threshold=None, pos=False):
    if pos:
        weights = weights.abs()
    # Calculate paths at current layer
    cur_path = weights.detach().reshape(len(weights),-1)
    cur_path = cur_path.sum(axis=-1)
    # Calculate sum of all previous paths
    if prev_paths == None:
        total_prev_paths = 1
    else: 
        total_prev_paths = prev_paths.sum()
    # Calculate next paths
    next_path = cur_path * total_prev_paths
    return next_path

def cal_linear_neurons(prev_paths, weights, threshold=0):
    next_path = torch.tile(prev_paths, (weights.size()[0],)).reshape(weights.size())
    next_path[weights.abs() <= threshold] = 0
    next_path = next_path.sum(axis=-1)
    return next_path, (next_path == 0).sum().item(), weights.sum(axis=-1)[next_path == 0].tolist()

def cal_conv_neurons(prev_paths, weights, threshold=0):
    # Calculate paths at current layer
    cur_path = weights.reshape(len(weights),-1)
    zero_n =  weights.reshape(len(weights),-1).sum(axis=-1)
    cur_path = (cur_path.abs() <= threshold).sum(axis=-1)
    # Calculate sum of all previous paths
    if prev_paths == None:
        total_prev_paths = 1
    else: 
        total_prev_paths = prev_paths.sum()
    # Calculate next paths
    next_path = cur_path * total_prev_paths
    return next_path, (next_path == 0).sum().item(), zero_n[next_path == 0].tolist()