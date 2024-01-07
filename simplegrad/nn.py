"""Neural network module."""

import random

from simplegrad import tensor


class Neuron:
    """Neuron class"""

    def __init__(self, n_inputs: int) -> None:
        self.weights = [tensor.Tensor(random.uniform(-1, 1)) for _ in range(n_inputs)]
        self.bias = tensor.Tensor(0.0)

    def __call__(self, inputs: list[float]):
        out = sum([w * x for w, x in zip(self.weights, inputs)]) + self.bias
        act = out.tanh()
        return act

    def parameters(self):
        """Return parameters of neuron.

        Returns:
            List of parameters
        """
        return self.weights + [self.bias]

    def reset_gradients(self):
        """Reset gradients to 0."""
        for p in self.parameters():
            p.grad = 0.0


class Layer:
    """Layer class."""

    def __init__(self, n_inputs: int, n_units: int) -> None:
        self.neurons = [Neuron(n_inputs) for _ in range(n_units)]

    def __call__(self, inputs: list[float]):
        out = [n(inputs) for n in self.neurons]
        return out if len(out) > 1 else out[0]

    def parameters(self):
        """Return parameters of layer.

        Returns:
            List of parameters
        """
        return [
            parameter for neuron in self.neurons for parameter in neuron.parameters()
        ]

    def reset_gradients(self):
        """Reset gradients to 0."""
        for p in self.parameters():
            p.grad = 0.0


class MLP:
    """Multi layer Perceptron class."""

    def __init__(self, n_inputs: int, n_units: list[int]) -> None:
        input_shapes = [n_inputs] + n_units
        self.layers = [
            Layer(input_shapes[i], input_shapes[i + 1]) for i in range(len(n_units))
        ]

    def __call__(self, inputs: list[float]):
        for layer in self.layers:
            out = layer(inputs)
        return out

    def parameters(self):
        """Return parameters of mlp.

        Returns:
            List of parameters
        """
        return [parameter for layer in self.layers for parameter in layer.parameters()]

    def reset_gradients(self):
        """Reset gradients to 0."""
        for p in self.parameters():
            p.grad = 0.0
