import torch


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
    return Ceil.apply(x)


def floor(x: torch.Tensor) -> torch.Tensor:
    """Differentiable version of torch.floor.
    For gradient calculation, this mimicks the identity function."""
    return Floor.apply(x)


def round(x: torch.Tensor) -> torch.Tensor:
    """Differentiable version of torch.round.
    For gradient calculation, this mimicks the identity function."""
    return Round.apply(x)
