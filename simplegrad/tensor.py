"""Tensor module."""

from __future__ import annotations

import math


class Tensor:
    """Tensor class to store values and gradients."""

    def __init__(self, value: float) -> None:
        self.value = value
        self._children: set[Tensor] = set()
        self.grad = 0.0
        self._backward = lambda: None

    def __repr__(self) -> str:
        return f"Tensor(value={self.value})"

    def __add__(self, operand: Tensor | int | float) -> Tensor:
        operand = (
            Tensor(value=operand) if isinstance(operand, (float, int)) else operand
        )
        out = Tensor(self.value + operand.value)
        out._children.update({self, operand})

        def _backward():
            self.grad += out.grad
            operand.grad += out.grad

        self._backward = _backward
        return out

    def __radd__(self, operand: Tensor | int | float) -> Tensor:
        return self + operand

    def __neg__(self) -> Tensor:
        return -1 * self

    def __sub__(self, operand: Tensor | int | float) -> Tensor:
        return self + (-operand)

    def __rsub__(self, operand: Tensor | int | float) -> Tensor:
        return operand + (-self)

    def __mul__(self, operand: Tensor | int | float) -> Tensor:
        operand = (
            Tensor(value=operand) if isinstance(operand, (float, int)) else operand
        )
        out = Tensor(self.value * operand.value)
        out._children.update({self, operand})

        def _backward():
            self.grad += operand.value * out.grad
            operand.grad += self.value * out.grad

        self._backward = _backward
        return out

    def __rmul__(self, operand: Tensor | int | float) -> Tensor:
        return self * operand

    def __pow__(self, power: int | float) -> Tensor:
        out = Tensor(value=self.value**power)
        out._children.update({self})

        def _backward():
            self.grad += (power * self.value ** (power - 1)) * out.grad

        self._backward = _backward
        return out

    def __truediv__(self, operand: Tensor | int | float) -> Tensor:
        return self * operand**-1

    def exp(self) -> Tensor:
        """Apply exponentation function."""

        value = math.exp(self.value)
        out = Tensor(value)
        out._children.update({self})

        def _backward():
            self.grad += value * out.grad

        self._backward = _backward
        return out

    def tanh(self) -> Tensor:
        """Apply tanh activation function."""

        value = math.exp(2 * self.value)
        value = (value - 1) / (value + 1)
        out = Tensor(value)
        out._children.update({self})

        def _backward():
            self.grad += (1 - value**2) * out.grad

        self._backward = _backward
        return out

    def _topological_sort(self):
        topological_order = []
        visited = set()

        def _sort(node):
            if node not in visited:
                visited.add(node)
                for child in node._children:
                    _sort(child)
                topological_order.append(node)

        _sort(self)
        return topological_order

    def backward(self) -> None:
        """Compute backward pass on current Tensor."""

        self.grad = 1.0
        for node in reversed(self._topological_sort()):
            node._backward()
