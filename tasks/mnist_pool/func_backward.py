import torch
import torch.nn as nn
import torch.nn.functional as F


class AutogradFunction(torch.autograd.Function):
    backward_fn = None

    @staticmethod
    def forward(ctx, x, kernel_size=2):
        # Save inputs for backward pass
        ctx.save_for_backward(x)
        # Apply max pooling
        x = F.max_pool2d(x, kernel_size=kernel_size)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved inputs
        x = ctx.saved_tensors

        # Use the class-level backward function
        grad_input = AutogradFunction.backward_fn(x, grad_output)
        return grad_input, None


def forward_fn(
    x: torch.Tensor,
    kernel_size: int = 2,
) -> torch.Tensor:
    """Implements a max pooling layer with kernel size 2:

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
        kernel_size (int): Kernel size of the max pooling layer

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, in_channels, height // kernel_size, width // kernel_size)
    """
    # Apply max pooling
    x = F.max_pool2d(x, kernel_size=kernel_size)
    return x


class Model(nn.Module):
    """
    Simple model that performs a max pooling layer.
    """

    def __init__(
        self,
        kernel_size: int = 2,
    ):
        """
        Initializes the max pooling layer.

        Args:
            kernel_size (int): Kernel size of the max pooling layer
        """
        super(Model, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor, fn=forward_fn) -> torch.Tensor:
        """
        Forward pass that calls forward_fn.
        """
        return fn(x, self.kernel_size)


def get_inputs(
    batch_size: int = 64,
    in_channels: int = 64,
    height: int = 28,
    width: int = 28,
):
    x = torch.randn(batch_size, in_channels, height, width, requires_grad=True)
    return [x]


input_names = ["x"]
