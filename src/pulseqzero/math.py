import torch
import numpy as np


class Ceil(torch.autograd.Function):
    @staticmethod
    def forward(x):
        return torch.ceil(x)
    
    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class Floor(torch.autograd.Function):
    @staticmethod
    def forward(x):
        return torch.floor(x)
    
    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class Round(torch.autograd.Function):
    @staticmethod
    def forward(x):
        return torch.round(x)
    
    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def ceil(x: torch.Tensor) -> torch.Tensor:
    """Differentiable version of torch.ceil.
    For gradient calculation, this mimicks the identity function."""
    try:
        return Ceil.apply(x)
    except TypeError:
        return torch.as_tensor(np.ceil(x))


def floor(x: torch.Tensor) -> torch.Tensor:
    """Differentiable version of torch.floor.
    For gradient calculation, this mimicks the identity function."""
    try:
        return Floor.apply(x)
    except TypeError:
        return torch.as_tensor(np.floor(x))


def round(x: torch.Tensor) -> torch.Tensor:
    """Differentiable version of torch.round.
    For gradient calculation, this mimicks the identity function."""
    try:
        return Round.apply(x)
    except TypeError:
        return torch.as_tensor(np.round(x))
