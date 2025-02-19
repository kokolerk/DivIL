import torch.nn as nn
from torch import Tensor

from torch_geometric.typing import OptTensor
from torch_geometric.nn.conv import MessagePassing

# copy from https://pytorch-geometric.readthedocs.io/en/2.0.4/_modules/torch_geometric/nn/models/explainer.html
def set_masks(mask: Tensor, model: nn.Module):
    """Apply mask to every graph layer in the model."""

    # Loop over layers and set masks on MessagePassing layers:
    for module in model.modules():
        if isinstance(module, MessagePassing):
            module._explain = True
            module._edge_mask = mask


def clear_masks(model: nn.Module):
    """Clear all masks from the model."""
    for module in model.modules():
        if isinstance(module, MessagePassing):
            module._explain = False
            module._edge_mask = None
    return module


# def set_masks(mask: Tensor, model: nn.Module):
#         for module in model.modules():
#             if isinstance(module, MessagePassing):
#                 module._explain = True
#                 module._edge_mask = mask
#                 module._apply_sigmoid = False 
#                 # module.__explain__ = True
#                 # module.__edge_mask__ = mask
                

# def clear_masks(model: nn.Module):
#     for module in model.modules():
#         if isinstance(module, MessagePassing):
#             module._explain = False
#             module._edge_mask = None
#             module._apply_sigmoid = True
#             # module.__explain__ = False
#             # module.__edge_mask__ = None