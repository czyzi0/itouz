import numpy as np


class Model:

    def __init__(self, layers, loss):
        self.layers = layers
        self.loss = loss

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def train(self, x_train, y_train, x_val, y_val, lr, momentum, epochs):
        loss = None
        for epoch in range(epochs):
            print(f'Epoch: {epoch}')
            for x_batch, y_batch in zip(x_train, y_train):
                loss_ = self._run_batch(x_batch, y_batch, lr, momentum)
                loss = loss_ if loss is None else 0.5 * loss_ + 0.5 * loss
                print(f'\rLoss: {loss}', end='')
            print()

    def _run_batch(self, x_train, y_train, lr, momentum):
        # Forward pass
        x = x_train
        for layer in self.layers:
            x = layer.forward(x)
        y = x

        # Compute loss and first gradient
        loss = self.loss.forward(y, y_train)
        grad = self.loss.backward()

        # Backpropagation
        for layer in reversed(self.layers):
            grad, deltas = layer.backward(grad)
            layer.optimize(deltas, lr, momentum)

        return loss


class Linear:

    def __init__(self, in_dim, out_dim):
        self.W = np.random.rand(out_dim, in_dim)
        self.b = np.random.rand(out_dim)

        self._prev_dW = np.zeros_like(self.W)
        self._prev_db = np.zeros_like(self.b)

    def __str__(self):
        return f'Linear        {self.n_params(): <12}'

    def forward(self, x):
        self._prev_x = x
        return np.dot(x, self.W.T) + self.b

    def backward(self, grad):
        # Calculate dW
        if grad.ndim == 2:
            dW = np.dot(self._prev_x.T, grad).T
        elif grad.ndim == 3:
            dW = sum(np.dot(self._prev_x[:,i,:].T, grad[:,i,:]).T for i in range(grad.shape[1]))
            dW = dW / grad.shape[1]
        else:
            raise NotImplementedError

        db = np.mean(grad, axis=tuple(range(grad.ndim -1)))
        grad = np.dot(grad, self.W)

        return grad, (dW, db)

    def optimize(self, deltas, lr, momentum):
        dW, db = deltas

        dW = lr * dW + momentum * self._prev_dW
        db = lr * db + momentum * self._prev_db

        self.W = self.W - dW
        self.b = self.b - db

        self._prev_dW = dW
        self._prev_db = db

    def n_params(self):
        return np.prod(self.W.shape) + np.prod(self.b.shape)


class RNN:

    def __init__(self):
        pass

    def __str__(self):
        return f'RNN            todo'

    def forward(self, x):
        pass

    def backward(self, grad):
        pass

    def optimize(self):
        pass

    def n_params(self):
        pass


class ReLU:

    def __init__(self):
        pass

    def __str__(self):
        return f'ReLU     {self.n_params(): <12}'

    def forward(self, x):
        self._prev_x = np.copy(x)
        return np.clip(x, 0, None)

    def backward(self, grad):
        grad = np.where(self._prev_x > 0, grad, 0)
        return grad, tuple()

    def optimize(self, deltas, lr, momentum):
        pass

    def n_params(self):
        return 0


class Softmax:

    def __init__(self):
        pass

    def __str__(self):
        return f'Softmax      {self.n_params(): <12}'

    def forward(self, x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        self._prev_output = e_x / np.sum(e_x, axis=-1, keepdims=True)
        return self._prev_output

    def backward(self, grad):
        grad = self._prev_output * (grad - np.sum(grad * self._prev_output, axis=-1, keepdims=True))
        return grad, tuple()

    def optimize(self, deltas, lr, momentum):
        pass

    def n_params(self):
        return 0


class MSELoss:

    def __init__(self):
        pass

    def __str__(self):
        return 'MSELoss'

    def forward(self, input_, target):
        self._prev_input = input_
        self._prev_target = target
        return np.mean(np.power(input_ - target, 2))

    def backward(self):
        return 2 * (self._prev_input - self._prev_target)


class CrossEntropyLoss:

    def __init__(self):
        pass

    def __str__(self):
        return 'CrossEntropyLoss'

    def forward(self, input_, target):
        self._prev_input = np.clip(input_, 1e-8, None)
        self._prev_target = target
        return np.mean(np.where(target == 1, -np.log(self._prev_input), 0))

    def backward(self):
        return np.where(self._prev_target == 1, -1 / self._prev_input, 0)
