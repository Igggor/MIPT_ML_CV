from builtins import object
import os
import numpy as np

from .layers import *
from .layer_utils import *

class FullyConnectedNet(object):
    """Class for a multi-layer fully connected neural network.

    Network contains an arbitrary number of hidden layers, ReLU nonlinearities,
    and a softmax loss function. This will also implement dropout and batch/layer
    normalization as options. For a network with L layers, the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional and the {...} block is
    repeated L - 1 times.

    Learnable parameters are stored in the self.params dictionary and will be learned
    using the Solver class.
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout_keep_ratio=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout_keep_ratio: Scalar between 0 and 1 giving dropout strength.
            If dropout_keep_ratio=1 then the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
            are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
            initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
            this datatype. float32 is faster but less accurate, so you should use
            float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers.
            This will make the dropout layers deteriminstic so we can gradient check the model.
        """
        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        
        # Собираем все размеры слоев
        layer_dims = [input_dim] + hidden_dims + [num_classes]
        
        # Инициализируем веса и смещения для каждого слоя
        for i in range(1, self.num_layers + 1):
            # Веса: W1, W2, ...
            self.params[f'W{i}'] = weight_scale * np.random.randn(
                layer_dims[i-1], layer_dims[i]
            )
            # Смещения: b1, b2, ...
            self.params[f'b{i}'] = np.zeros(layer_dims[i])
        
        # Если используем batch normalization, добавляем gamma и beta
        if self.normalization == "batchnorm":
            for i in range(1, self.num_layers):
                # gamma для каждого скрытого слоя
                self.params[f'gamma{i}'] = np.ones(hidden_dims[i-1])
                # beta для каждого скрытого слоя
                self.params[f'beta{i}'] = np.zeros(hidden_dims[i-1])
        
        # Если используем layer normalization, добавляем gamma и beta
        if self.normalization == "layernorm":
            for i in range(1, self.num_layers):
                self.params[f'gamma{i}'] = np.ones(hidden_dims[i-1])
                self.params[f'beta{i}'] = np.zeros(hidden_dims[i-1])
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype.
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """Compute loss and gradient for the fully connected net.
        
        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
            scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
            names to gradients of the loss with respect to those parameters.
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        
        # Решейпим входные данные
        X_reshaped = X.reshape(X.shape[0], -1)
        
        # Прямой проход для каждого слоя
        cache_list = []
        current_out = X_reshaped
        
        for i in range(1, self.num_layers + 1):
            W = self.params[f'W{i}']
            b = self.params[f'b{i}']
            
            # Affine слой
            current_out, affine_cache = affine_forward(current_out, W, b)
            cache_list.append(('affine', affine_cache))
            
            # Если не последний слой, добавляем нормализацию, ReLU и dropout
            if i < self.num_layers:
                # Batch Normalization
                if self.normalization == "batchnorm":
                    gamma = self.params[f'gamma{i}']
                    beta = self.params[f'beta{i}']
                    current_out, bn_cache = batchnorm_forward(current_out, gamma, beta, self.bn_params[i-1])
                    cache_list.append(('batchnorm', bn_cache))
                
                # Layer Normalization
                if self.normalization == "layernorm":
                    gamma = self.params[f'gamma{i}']
                    beta = self.params[f'beta{i}']
                    current_out, ln_cache = layernorm_forward(current_out, gamma, beta, self.bn_params[i-1])
                    cache_list.append(('layernorm', ln_cache))
                
                # ReLU
                current_out, relu_cache = relu_forward(current_out)
                cache_list.append(('relu', relu_cache))
                
                # Dropout
                if self.use_dropout:
                    current_out, dropout_cache = dropout_forward(current_out, self.dropout_param)
                    cache_list.append(('dropout', dropout_cache))
        
        scores = current_out
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early.
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the   #
        # scale and shift parameters.                                              #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        
        # Вычисляем softmax loss
        loss, dout = softmax_loss(scores, y)
        
        # Добавляем L2 регуляризацию
        for i in range(1, self.num_layers + 1):
            loss += 0.5 * self.reg * np.sum(self.params[f'W{i}']**2)
        
        # Обратный проход
        # Идем с конца к началу
        for i in range(self.num_layers, 0, -1):
            layer_type, cache = cache_list.pop()
            
            if layer_type == 'affine':
                dout, dW, db = affine_backward(dout, cache)
                dW += self.reg * self.params[f'W{i}']
                grads[f'W{i}'] = dW
                grads[f'b{i}'] = db
            elif layer_type == 'relu':
                dout = relu_backward(dout, cache)
            elif layer_type == 'dropout':
                dout = dropout_backward(dout, cache)
            elif layer_type == 'batchnorm':
                dout, dgamma, dbeta = batchnorm_backward(dout, cache)
                grads[f'gamma{i}'] = dgamma
                grads[f'beta{i}'] = dbeta
            elif layer_type == 'layernorm':
                dout, dgamma, dbeta = layernorm_backward(dout, cache)
                grads[f'gamma{i}'] = dgamma
                grads[f'beta{i}'] = dbeta
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class TwoLayerNet(object):
    """
    Двухслойная полносвязная нейронная сеть с нелинейностью ReLU и функцией потерь softmax, 
    использующая модульную структуру слоев. Мы предполагаем размерность входных данных D, 
    размерность скрытого слоя H и выполняем классификацию по C классам.

Архитектура должна быть полносвязный слой - reLU - полносвязный слой - softmax.

Обратите внимание, что этот класс не реализует градиентный спуск; вместо этого он
будет взаимодействовать с отдельным объектом Solver, который отвечает за выполнение
оптимизации.

Обучаемые параметры модели хранятся в словаре
self.params, который сопоставляет имена параметров с массивами numpy.
    """

    def __init__(
        self,
        input_dim=3 * 32 * 32,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
    ):
        """
        Инициализация.

        Входы:
        - input_dim: Целое число, указывающее размер входных данных
- hidden_dim: Целое число, указывающее размер скрытого слоя
- num_classes: Целое число, указывающее количество классов для классификации
- weight_scale: Скаляр, указывающий стандартное отклонение для случайной инициализации весов.
- reg: Скаляр, указывающий силу L2-регуляризации.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
       # TODO: Инициализировать веса и смещения двухслойной сети. Веса 
      # должны быть инициализированы гауссовым распределением с центром в точке 0,0 и
      # стандартным отклонением, равным weight_scale, а смещения должны быть
      # инициализированы нулем. Все веса и смещения должны храниться в
      # словаре self.params, при этом веса первого слоя
      # и смещения будут использовать ключи 'W1' и 'b1', а веса и смещения второго слоя 
      # будут использовать ключи 'W2' и 'b2'.
      
        ############################################################################
        self.params = {
            'W1': weight_scale * np.random.randn(input_dim, hidden_dim),
            'b1': np.zeros(hidden_dim),
            'W2': weight_scale * np.random.randn(hidden_dim, num_classes),
            'b2': np.zeros(num_classes)
        }
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def loss(self, X, y=None):
        """
        Вычисляет функцию потерь и градиент для мини-батча данных.
Входные данные:
- X: Массив входных данных формы (N, d_1, ..., d_k)
- y: Массив меток формы (N, ..., d_k). y[i] — метка для X[i].

Возвращает:
Если y равно None, то выполняется прямой проход модели во время тестирования и возвращается:
- scores: Массив формы (N, C), содержащий оценки классификации, где
scores[i, c] — оценка классификации для X[i] и класса c.
Если y не равно None, то выполняется прямой и обратный проходы во время обучения и
возвращается кортеж из:
- loss: Скалярное значение, определяющее функцию потерь
- grads: Словарь с теми же ключами, что и self.params, сопоставляющий имена параметров
с градиентами функции потерь относительно этих параметров.
        """
        scores = None
        ############################################################################
        # TODO: Реализовать прямой проход для двухслойной сети, вычисляя 
        # оценки классов для X и сохраняя их в переменной scores. 
        ############################################################################
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        
        # Решейпим входные данные
        X_reshaped = X.reshape(X.shape[0], -1)
        
        # Прямой проход: affine - relu - affine
        hidden_out, hidden_cache = affine_forward(X_reshaped, W1, b1)
        relu_out, relu_cache = relu_forward(hidden_out)
        scores, score_cache = affine_forward(relu_out, W2, b2)
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Реализовать обратный проход для двухслойной сети. Сохранить функцию потерь
        # в переменной loss, а градиенты — в словаре grads. Вычислить потери данных
        # с помощью softmax и убедиться, что grads[k] содержит градиенты для
        # self.params[k]. Не забудьте добавить L2-регуляризацию!
        # #
        # ПРИМЕЧАНИЕ: Чтобы убедиться, что ваша реализация соответствует нашей и вы проходите #
        # автоматизированные тесты, убедитесь, что ваша L2-регуляризация включает множитель #
        # равный 0,5 для упрощения выражения для градиента. #
        ############################################################################
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        
        # Softmax loss
        loss, dout = softmax_loss(scores, y)
        
        # L2 регуляризация
        loss += 0.5 * self.reg * (np.sum(W1**2) + np.sum(W2**2))
        
        # Обратный проход
        # Второй affine слой
        drelu, dW2, db2 = affine_backward(dout, score_cache)
        dW2 += self.reg * W2
        
        # ReLU
        dhidden = relu_backward(drelu, relu_cache)
        
        # Первый affine слой
        dX, dW1, db1 = affine_backward(dhidden, hidden_cache)
        dW1 += self.reg * W1
        
        # Сохраняем градиенты
        grads['W1'] = dW1
        grads['b1'] = db1
        grads['W2'] = dW2
        grads['b2'] = db2
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

    def save(self, fname):
      """Save model parameters."""
      fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
      params = self.params
      np.save(fpath, params)
      print(fname, "saved.")
    
    def load(self, fname):
      """Load model parameters."""
      fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
      if not os.path.exists(fpath):
        print(fname, "not available.")
        return False
      else:
        params = np.load(fpath, allow_pickle=True).item()
        self.params = params
        print(fname, "loaded.")
        return True

