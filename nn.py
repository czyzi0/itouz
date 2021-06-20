import numpy as np


class Model:

    def __init__(self, layers, loss):
        self.layers = layers
        self.loss = loss

    def __str__(self):


    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def train(self, x_train, y_train, x_val, y_val, lr, momentum, epochs):
        loss = 0.0
        for epoch in range(epochs):
            print(f'Epoch: {epoch} - ', end='')
            for x_batch, y_batch in zip(x_train, y_train):
                loss += self._run_batch(x_batch, y_batch, lr, momentum)
            loss = loss / len(x_train)
            print(f'Loss: {loss:.5f}')
            loss = 0.0


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


##############################################################################
# LAYERS
##############################################################################

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

        db = np.mean(grad, axis=tuple(range(grad.ndim - 1)))
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

    def __init__(self, in_dim, out_dim):
        self.Wx = np.random.rand(out_dim, in_dim) * np.sqrt(1 / (in_dim + out_dim))
        self.Wh = np.random.rand(out_dim, out_dim) * np.sqrt(1 / (out_dim + out_dim))
        self.b = np.random.rand(out_dim) * np.sqrt(1 / out_dim)

        self._out_dim = out_dim

        self._prev_dWx = np.zeros_like(self.Wx)
        self._prev_dWh = np.zeros_like(self.Wh)
        self._prev_db = np.zeros_like(self.b)

    def __str__(self):
        return f'RNN       {self.n_params(): <12}'

    def forward(self, x):
        self._prev_x = x

        batch_size, seq_len, _ = x.shape

        self._prev_output = [np.zeros((batch_size, self._out_dim))]
        for i in range(seq_len):
            # Find hidden state from previous timestep
            prev_h = self._prev_output[-1]
            # Forward pass through single cell, TODO: Add tanh
            self._prev_output.append(np.tanh(
                np.dot(x[:,i,:], self.Wx.T) + np.dot(prev_h, self.Wh.T) + self.b))

        self._prev_output = np.stack(self._prev_output, axis=1)
        return self._prev_output[:,1:,:]

    def backward(self, grad):
        batch_size, seq_len, _ = grad.shape

        dWx = np.zeros_like(self.Wx)
        dWh = np.zeros_like(self.Wh)
        db = np.zeros_like(self.b)

        # Gradients with regard to the input
        grad_x = []
        # Output gradient from previous step (no previous step)
        grad_h = np.zeros((batch_size, self._out_dim))

        for i in reversed(range(seq_len)):
            # Add gradient from output
            grad_h = grad_h + grad[:,i,:]

            # Backpropagate through tanh
            grad_tanh = grad_h * (1 - self._prev_output[:,i+1,:]**2)

            # Gradients for parameters
            dWx = dWx + np.dot(self._prev_x[:,i,:].T, grad_tanh).T
            dWh = dWh + np.dot(self._prev_output[:,i,:].T, grad_tanh).T
            db = db + np.mean(grad_tanh, axis=0)

            # Calculate gradient with regard to the input
            grad_x.append(np.dot(grad_tanh, self.Wx))
            # Calculate gradient for next step
            grad_h = np.dot(grad_tanh, self.Wh)

        # Normalize parameter gradients
        dWx = dWx / seq_len
        sWh = dWh / seq_len
        db = db / seq_len

        # Prepare gradient with regard to the input
        grad_x = np.stack(list(reversed(grad_x)), axis=1)

        return grad_x, (dWx, dWh, db)

    def optimize(self, deltas, lr, momentum):
        dWx, dWh, db = deltas

        dWx = lr * dWx + momentum * self._prev_dWx
        dWh = lr * dWh + momentum * self._prev_dWh
        db = lr * db + momentum * self._prev_db

        self.Wx = self.Wx - dWx
        self.Wh = self.Wh - dWh
        self.b = self.b - db

        self._prev_dWx = dWx
        self._prev_dWh = dWh
        self._prev_db = db

    def n_params(self):
        return np.prod(self.Wx.shape) + np.prod(self.Wh.shape) + np.prod(self.b.shape)


##############################################################################
# ACTIVATIONS
##############################################################################

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


##############################################################################
# LOSSES
##############################################################################

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
