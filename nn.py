"""Module with simple neural network library implemented for the project."""

import time

import numpy as np

from utils import my_print


class Model:
    """Class that represents neural network and handles all high level
    operations on it.

    Args:
        layers (List[_layer_]): List of layers to use in the network.
        loss (_loss_): Loss to use in the neural network training.
        train_metrics (List[_metric_]): List of metrics to evaluate training
            set on during the training.
        val_metrics (List[_metrics_]): List of metrics to evaluate validation
            set on during the training.

    """

    def __init__(self, layers, loss, train_metrics, val_metrics):
        self.layers = layers
        self.loss = loss
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics

    def __str__(self):
        str_ = (
            f'----------------Model----------------\n'
            f'layer                   # of params  \n'
            f'-------------------------------------\n')

        for layer in self.layers:
            str_ += f'{layer}\n'
        str_ += f'{self.loss}\n'

        str_ += (
            f'-------------------------------------\n'
            f'TOTAL                   {self.n_params()}\n')

        return str_

    def predict(self, x):
        """Pass data through the network and do the prediction.

        Args:
            x (np.array): Data in the form of numpy array in the shape defined
                by used layers.

        Returns:
            np.array: Prediction with the shape defined by used layers.

        """
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def train(self, x_train, y_train, x_val, y_val, lr, momentum, epochs):
        """Train the neural network.

        Args:
            x_train (List[np.array]): Training data inputs in the form of list
                of batches with shape defined by used layers.
            y_train (List[np.array]): Training data targets in the form of list
                of batches with shape defined by used layers.
            x_val (List[np.array]): Validation data inputs in the form of list
                of batches with shape defined by used layers.
            y_val (List[np.array]): Validation data targets in the form of list
                of batches with shape defined by used layers.
            lr (float): Learning rate.
            momentum (float): Momentum.
            epochs (int): Number of epochs to train the model for.

        """
        loss = 0.0
        for epoch in range(1, epochs + 1):
            start = time.time()
            my_print(f'Epoch: {epoch}', end='', flush=True)

            # Training
            for x, y_true in zip(x_train, y_train):
                loss_, y_pred = self._run_batch(x, y_true, lr=lr, momentum=momentum)
                loss += loss_
                for metric in self.train_metrics:
                    metric.log(y_true, y_pred)
            loss = loss / len(x_train)
            my_print(f' - Loss: {loss:.5f}', end='')
            loss = 0.0

            for metric in self.train_metrics:
                my_print(f' - {metric}: {metric.calc():.4f}', end='')

            # Validation
            for x, y_true in zip(x_val, y_val):
                loss_, y_pred = self._run_batch(x, y_true, optimize=False)
                loss += loss_
                for metric in self.val_metrics:
                    metric.log(y_true, y_pred)
            loss = loss / len(x_val)
            my_print(f' - ValLoss: {loss:.5f}', end='')
            loss = 0.0

            for metric in self.val_metrics:
                my_print(f' - Val{metric}: {metric.calc():.4f}', end='')

            # Shuffle the training data
            data = list(zip(x_train, y_train))
            np.random.shuffle(data)
            x_train, y_train = zip(*data)

            minutes, seconds = divmod(time.time() - start, 60)
            my_print(f' - Duration: {int(minutes)}m {int(seconds)}s', end='')

            my_print()

    def _run_batch(self, x, y, lr=None, momentum=None, optimize=True):
        """Private method that runs the training on the single batch."""
        # Forward pass
        for layer in self.layers:
            x = layer.forward(x)
        # Compute loss
        loss = self.loss.forward(x, y)

        if not optimize:
            return loss, x

        # Compute first gradient
        grad = self.loss.backward()
        # Backpropagation
        for layer in reversed(self.layers):
            grad, deltas = layer.backward(grad)
            layer.optimize(deltas, lr, momentum)

        return loss, x

    def n_params(self):
        """Get total number of parameters in the neural network."""
        return sum(l.n_params() for l in self.layers)


##############################################################################
# METRICS
##############################################################################

class Accuracy:
    """Accuracy metric."""

    def __init__(self):
        self._n_correct = 0
        self._n_total = 0

    def __str__(self):
        return 'Accuracy'

    def log(self, y_true, y_pred):
        """Log single batch of results.

        Args:
            y_true (np.array): Targets in the form of the numpy array with one
                of two shapes: (batch_size, seq_len, n_classes) or
                (batch_size, n_classes).
            y_pred (np.array): Predictions in the form of the numpy array with
                one of two shapes: (batch_size, seq_len, n_classes) or
                (batch_size, n_classes).

        """
        y_true = y_true.reshape(-1, y_true.shape[-1])
        y_pred = y_pred.reshape(-1, y_pred.shape[-1])

        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)

        self._n_correct += np.sum(y_true == y_pred)
        self._n_total += y_true.shape[0]

    def calc(self):
        """Calculate the metric for all logged values and reset state.

        Returns:
            float: Total accuracy for all logged results.

        """
        acc = self._n_correct / self._n_total
        self._n_correct = 0
        self._n_total = 0
        return acc 


##############################################################################
# LAYERS
##############################################################################

class Linear:
    """Linear (fully-connected) layer implementation.

    Args:
        in_dim (int): Size of the input.
        out_dim (int): Size of the output.

    """

    def __init__(self, in_dim, out_dim):
        self.W = np.random.rand(out_dim, in_dim)
        self.b = np.random.rand(out_dim)

        self._in_dim = in_dim
        self._out_dim = out_dim

        self._prev_dW = np.zeros_like(self.W)
        self._prev_db = np.zeros_like(self.b)

    def __str__(self):
        layer = f'Linear({self._in_dim}, {self._out_dim})'
        return f'{layer: <24}{self.n_params()}'

    def forward(self, x):
        """Forward pass

        Args:
            x (np.array): Input data in the form of numpy array with one of
                two shapes (batch_size, in_dim) or
                (batch_size, seq_len, in_dim).

        Returns:
            np.array: Output in the form of numpy array with one of two shapes,
                depending on the input (batch_size, out_dim) or
                (batch_size, seq_len, out_dim).

        """
        self._prev_x = x
        return np.dot(x, self.W.T) + self.b

    def backward(self, grad):
        """Backward pass

        Args:
            grad (np,array): Gradients with regard to layer's output. With
                shape (batch_size, out_dim) or (batch_size, seq_len, out_dim).

        Returns:
            Tuple[np.array, Tuple[np.array, np.array]]: Gradients with regard
                to the layer's input with proper shape, and gradients with
                regard to the layer's parameters.

        """
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
        """Update layer's parameters.

        Args:
            deltas: Gradients with regard to the parameters as returned by
                `backward` method.
            lr (float): Learning rate.
            momentum (flaot): Momentum.

        """
        dW, db = deltas

        dW = lr * dW + momentum * self._prev_dW
        db = lr * db + momentum * self._prev_db

        self.W = self.W - dW
        self.b = self.b - db

        self._prev_dW = dW
        self._prev_db = db

    def n_params(self):
        """Get number of parameters in the layer."""
        return np.prod(self.W.shape) + np.prod(self.b.shape)


class RNN:
    """Standard recurrent layer implementation.

    Args:
        in_dim (int): Size of the input.
        out_dim (int): Size of the output.

    """

    def __init__(self, in_dim, out_dim):
        self.Wx = np.random.rand(out_dim, in_dim) * np.sqrt(1 / (in_dim + out_dim))
        self.Wh = np.random.rand(out_dim, out_dim) * np.sqrt(1 / (out_dim + out_dim))
        self.b = np.random.rand(out_dim) * np.sqrt(1 / out_dim)

        self._in_dim = in_dim
        self._out_dim = out_dim

        self._prev_dWx = np.zeros_like(self.Wx)
        self._prev_dWh = np.zeros_like(self.Wh)
        self._prev_db = np.zeros_like(self.b)

    def __str__(self):
        layer = f'RNN({self._in_dim}, {self._out_dim})'
        return f'{layer: <24}{self.n_params()}'

    def forward(self, x):
        """Forward pass

        Args:
            x (np.array): Input data in the form of numpy array with shape
                (batch_size, seq_len, in_dim).

        Returns:
            np.array: Output in the form of numpy array with shape (batch_size,
                seq_len, out_dim).

        """
        self._prev_x = x

        batch_size, seq_len, _ = x.shape

        self._prev_output = [np.zeros((batch_size, self._out_dim))]
        for i in range(seq_len):
            # Find hidden state from previous timestep
            prev_h = self._prev_output[-1]
            # Forward pass through single cell
            self._prev_output.append(np.tanh(
                np.dot(x[:,i,:], self.Wx.T) + np.dot(prev_h, self.Wh.T) + self.b))

        self._prev_output = np.stack(self._prev_output, axis=1)
        return self._prev_output[:,1:,:]

    def backward(self, grad):
        """Backward pass

        Args:
            grad (np,array): Gradients with regard to layer's output. With
                shape (batch_size, seq_len, out_dim).

        Returns:
            Tuple[np.array, Tuple[np.array, np.array, np.array]]: Gradients
                with regard to the layer's input with proper shape, and
                gradients with regard to the layer's parameters.

        """
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
            db = db + np.sum(grad_tanh, axis=0)

            # Calculate gradient with regard to the input
            grad_x.append(np.dot(grad_tanh, self.Wx))
            # Calculate gradient for next step
            grad_h = np.dot(grad_tanh, self.Wh)

        # Normalize parameter gradients
        #dWx = dWx / seq_len
        #sWh = dWh / seq_len
        #db = db / seq_len

        # Prepare gradient with regard to the input
        grad_x = np.stack(list(reversed(grad_x)), axis=1)

        return grad_x, (dWx, dWh, db)

    def optimize(self, deltas, lr, momentum):
        """Update layer's parameters.

        Args:
            deltas: Gradients with regard to the parameters as returned by
                `backward` method.
            lr (float): Learning rate.
            momentum (flaot): Momentum.

        """
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
        """Get number of parameters in the layer."""
        return np.prod(self.Wx.shape) + np.prod(self.Wh.shape) + np.prod(self.b.shape)


##############################################################################
# ACTIVATIONS
##############################################################################

class ReLU:
    """ReLU activation function implementation."""

    def __init__(self):
        pass

    def __str__(self):
        return f'ReLU()                  {self.n_params()}'

    def forward(self, x):
        """Forward pass

        Args:
            x (np.array): Input data in the form of numpy array with one of
                two shapes (batch_size, dim) or (batch_size, seq_len, dim).

        Returns:
            np.array: Output in the form of numpy array with one of two shapes,
                depending on the input (batch_size, dim) or
                (batch_size, seq_len, dim).

        """
        self._prev_x = np.copy(x)
        return np.clip(x, 0, None)

    def backward(self, grad):
        """Backward pass

        Args:
            grad (np,array): Gradients with regard to layer's output. With
                shape (batch_size, dim) or (batch_size, seq_len, dim).

        Returns:
            Tuple[np.array, Tuple[]]: Gradients with regard to the layer's
                input with proper shape, and empty tuple for compatibilty.

        """
        grad = np.where(self._prev_x > 0, grad, 0)
        return grad, tuple()

    def optimize(self, deltas, lr, momentum):
        """Stub for layer without parameters."""
        pass

    def n_params(self):
        """Get number of parameters in the layer."""
        return 0


class Softmax:
    """Softmax activation function implementation."""

    def __init__(self):
        pass

    def __str__(self):
        return f'Softmax()               {self.n_params()}'

    def forward(self, x):
        """Forward pass

        Args:
            x (np.array): Input data in the form of numpy array with one of
                two shapes (batch_size, dim) or (batch_size, seq_len, dim).

        Returns:
            np.array: Output in the form of numpy array with one of two shapes,
                depending on the input (batch_size, dim) or
                (batch_size, seq_len, dim).

        """
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        self._prev_output = e_x / np.sum(e_x, axis=-1, keepdims=True)
        return self._prev_output

    def backward(self, grad):
        """Backward pass

        Args:
            grad (np,array): Gradients with regard to layer's output. With
                shape (batch_size, dim) or (batch_size, seq_len, dim).

        Returns:
            Tuple[np.array, Tuple[]]: Gradients with regard to the layer's
                input with proper shape, and empty tuple for compatibilty.

        """
        grad = self._prev_output * (grad - np.sum(grad * self._prev_output, axis=-1, keepdims=True))
        return grad, tuple()

    def optimize(self, deltas, lr, momentum):
        """Stub for layer without parameters."""
        pass

    def n_params(self):
        """Get number of parameters in the layer."""
        return 0


##############################################################################
# LOSSES
##############################################################################

class MSELoss:
    """Mean squared error loss implementation."""

    def __init__(self):
        pass

    def __str__(self):
        return 'MSELoss()'

    def forward(self, input_, target):
        """Forward pass

        Args:
            input_ (np.array): Predictions in the form of numpy array with one
                of two shapes (batch_size, dim) or (batch_size, seq_len, dim).
            target (np.array): Targets in the form of numpy array with shapes
                corresponding to `input_`.

        Returns:
            float: Calculated loss.

        """
        self._prev_input = input_
        self._prev_target = target
        return np.mean(np.power(input_ - target, 2))

    def backward(self):
        """Calculate gradient from last forward pass."""
        return 2 * (self._prev_input - self._prev_target)


class CrossEntropyLoss:
    """Cross entropy loss implementation."""

    def __init__(self):
        pass

    def __str__(self):
        return 'CrossEntropyLoss()'

    def forward(self, input_, target):
        """Forward pass

        Args:
            input_ (np.array): Predictions in the form of numpy array with one
                of two shapes (batch_size, dim) or (batch_size, seq_len, dim).
            target (np.array): Targets in the form of numpy array with shapes
                corresponding to `input_`.

        Returns:
            float: Calculated loss.

        """
        self._prev_input = np.clip(input_, 1e-8, None)
        self._prev_target = target
        return (
            np.sum(np.where(target == 1, -np.log(self._prev_input), 0))
            / np.prod(input_.shape[:-1]))

    def backward(self):
        """Calculate gradient from last forward pass."""
        return np.where(self._prev_target == 1, -1 / self._prev_input, 0)
