"""Magnitude based pruning

Implements pruning strategy that prunes as follows

For a given tensor, sort by absolute value and
keep only the _fraction_ with highest magnitudes
so that overall desired compression is achieved
"""

import numpy as np
from ..pathcal.calculator import *
from ..pathcal import bellman
from ..pruning import (LayerPruning,
                       VisionPruning,
                       GradientMixin,
                       ActivationMixin)
from .utils import (fraction_threshold,
                    fraction_mask,
                    map_importances,
                    flatten_importances,
                    importance_masks,
                    activation_importance)


class GlobalMagWeight(VisionPruning):

    """
    Flow and magnitude based pruning
    """
    def model_masks(self):
        importances = map_importances(np.abs, self.params())
        importances = calc_flows(importances) # com
        flat_importances = flatten_importances(importances)
        threshold = fraction_threshold(flat_importances, self.fraction)
        masks = importance_masks(importances, threshold)
        total_size = 0
        total_prune = 0
        for layer in list(masks.keys()):
            weights = masks[layer]['weight'].flatten()
            im = importances[layer]['weight'].flatten()
            total_prune += (weights == 0).sum()
            total_size += len(weights)
            print(im.min(), im.max())
            print(f'layer: {layer}\n Pruned {(weights == 0).sum()/weights.shape[0]*100:.2f}%')
        # print(masks)
        print(f'Pruned {total_prune/total_size*100:.2f}%')
        return masks
    
    """
    Bellman's equations
    """
#     def model_masks(self):
#         importances = map_importances(np.abs, self.params())
        
#         # Bottom up algorithm
#         # masks = bellman.longest_botup_run(importances, self.fraction)
#         # Top down new
#         masks = bellman.longest_new_run(importances, self.fraction)
#         for layer in list(masks.keys()):
#             weights = masks[layer]['weight'].flatten()
#             im = importances[layer]['weight'].flatten()
#             print(im.min(), im.max())
#             print(f'layer: {layer}\n Pruned {(weights == 0).sum()/weights.shape[0]*100:.2f}%')
#         return masks
        


class LayerMagWeight(LayerPruning, VisionPruning):

    def layer_masks(self, module):
        params = self.module_params(module)
        importances = {param: np.abs(value) for param, value in params.items()}
        masks = {param: fraction_mask(importances[param], self.fraction)
                 for param, value in params.items() if value is not None}
        return masks


class GlobalMagGrad(GradientMixin, VisionPruning):

    def model_masks(self):
        params = self.params()
        grads = self.param_gradients()
        importances = {mod:
                       {p: np.abs(params[mod][p]*grads[mod][p])
                        for p in mod_params}
                       for mod, mod_params in params.items()}
        flat_importances = flatten_importances(importances)
        threshold = fraction_threshold(flat_importances, self.fraction)
        masks = importance_masks(importances, threshold)
        return masks


class LayerMagGrad(GradientMixin, LayerPruning, VisionPruning):

    def layer_masks(self, module):
        params = self.module_params(module)
        grads = self.module_param_gradients(module)
        importances = {param: np.abs(value*grads[param]) for param, value in params.items()}
        masks = {param: fraction_mask(importances[param], self.fraction)
                 for param, value in params.items() if value is not None}
        return masks


class GlobalMagAct(ActivationMixin, VisionPruning):

    def model_masks(self):
        params = self.params()
        activations = self.activations()
        # [0] is input activation
        importances = {mod:
                       {p: np.abs(activation_importance(params[mod][p], activations[mod][0]))
                        for p in mod_params}
                       for mod, mod_params in params.items()}
        flat_importances = flatten_importances(importances)
        threshold = fraction_threshold(flat_importances, self.fraction)
        masks = importance_masks(importances, threshold)
        return masks


class LayerMagAct(ActivationMixin, LayerPruning, VisionPruning):

    def layer_masks(self, module):
        params = self.module_params(module)
        input_act, _ = self.module_activations(module)
        importances = {param: np.abs(activation_importance(value, input_act))
                       for param, value in params.items()}
        masks = {param: fraction_mask(importances[param], self.fraction)
                 for param, value in params.items() if value is not None}
        return masks
