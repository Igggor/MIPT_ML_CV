from builtins import range
import numpy as np

def affine_forward(x, w, b):
    """
    Вычисляет прямой проход для аффинного (полносвязного) слоя.
    Входные данные x имеют форму (N, d_1, ..., d_k) и содержат мини-пакет из N примеров, 
    где каждый пример x[i] имеет форму (d_1, ..., d_k). Мы преобразуем каждый входной вектор 
    в вектор размерности D = d_1 * ... * d_k, и затем преобразуем его в выходной вектор размерности M.

Входные данные:
- x: Массив numpy, содержащий входные данные, формы (N, d_1, ..., d_k)
- w: Массив numpy весов, формы (D, M)
- b: Массив numpy смещений (bias), формы (M,)

    Возвращает кортеж:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Реализуйте обратный проход полносвязного слоя. Положите результат в out. You   #
    # Потребуется решейпить input в строку
    ###########################################################################

    x_reshaped = x.reshape(x.shape[0], -1)
    out = x_reshaped @ w + b

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Вычисляет обратный проход для полносвязного слоя.
    Входные данные:
    - dout: Производная от исходного слоя, форма (N, M)
    - cache: 
        - x: Входные данные, форма (N, d_1, ..., d_k)
        - w: Веса, форма (D, M)
        - b: Смещения, форма (M,)

Возвращает кортеж:
    - dx: Градиент относительно x, форма (N, d1, ..., d_k)
    - dw: Градиент относительно w, форма (D, M)
    - db: Градиент относительно b, форма (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO:  Реализуйте обратный проход полносвязного слоя.         
    ###########################################################################
    
    x_reshaped = x.reshape(x.shape[0], -1)
    dx = dout @ w.T
    dx = dx.reshape(x.shape)
    
    dw = x_reshaped.T @ dout
    
    db = np.sum(dout, axis=0)
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Вычисляет прямой проход для слоя выпрямленных линейных блоков (ReLU).

    Входные данные:
    - x: Входные данные любой формы
    Возвращает кортеж:
    - out: Выходные данные той же формы, что и x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Реализуйте RELU на прямом проходе.                                  #
    ###########################################################################
    out = np.maximum(0, x)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Вычисляет обратный проход для слоя ReLU.
    Входные данные:
    - dout: Производные от исходного слоя любой формы
    - cache: Входные данные x той же формы, что и dout
    Возвращает:
    - dx: Градиент относительно x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Реализуйте RELU на обраьном проходе
    ###########################################################################
    dx = dout * (x > 0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def softmax_loss(x, y):
    """
    Вычисляет функцию потерь и градиент для классификации с использованием функции softmax.
    Входные данные:
    - x: Входные данные формы (N, C), где x[i, j] — оценка для j-го класса для i-го входного значения.
    - y: Вектор меток формы (N,), где y[i] — метка для x[i] и 0 <= y[i] < C
    Возвращает кортеж:
    - loss: Скаляр, задающий функцию потерь
    - dx: Градиент функции потерь относительно x
    """
    loss, dx = None, None

    ###########################################################################
    # YOUR CODE
    ###########################################################################
    
    x_shifted = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x_shifted)
    
    sum_exp = np.sum(exp_x, axis=1, keepdims=True)
    
    probs = exp_x / sum_exp
    
    probs = np.clip(probs, 1e-10, 1.0)
    
    N = x.shape[0]
    loss = -np.sum(np.log(probs[np.arange(N), y])) / N
    
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx

def batchnorm_forward(x, gamma, beta, bn_param):
    """Прямой проход для batch-нормализации.

Во время обучения среднее значение выборки и (нескорректированная) дисперсия выборки вычисляются
на основе статистики батчей и используются для нормализации входящих данных.
Во время обучения мы также поддерживаем экспоненциально убывающее скользящее среднее
среднего значения и дисперсии каждого признака, и эти средние значения используются для нормализации
данных во время тестирования.

На каждом шаге времени мы обновляем скользящие средние значения для среднего значения и дисперсии, используя
экспоненциальное затухание на основе параметра момента:
running_mean = momentum * running_mean + (1 - momentum) * sample_mean
running_var = momentum * running_var + (1 - momentum) * sample_var

Обратите внимание, что в статье о пакетной нормализации предлагается другое поведение во время тестирования:
они вычисляют среднее значение выборки и дисперсию для каждого признака, используя
большое количество обучающих изображений, а не скользящее среднее. Для
этой реализации мы выбрали использование скользящих средних, поскольку
они не требуют дополнительного этапа оценки; В реализации пакетной нормализации torch7 также используются скользящие средние.

Входные данные:

- x: Данные формы (N, D)
- gamma: Параметр масштаба формы (D,)
- beta: Параметр сдвига формы (D,)
- bn_param: Словарь со следующими ключами:
- mode: 'train' или 'test'; обязательно
- eps: Константа для обеспечения численной стабильности
- momentum: Константа для скользящего среднего / дисперсии.
- running_mean: Массив формы (D,), дающий скользящее среднее признаков
- running_var: Массив формы (D,), дающий скользящую дисперсию признаков

Возвращает кортеж:

- out: формы (N, D)

- cache: Кортеж значений, необходимых в обратном проходе
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
        #######################################################################
        # TODO: 
        #######################################################################
        sample_mean = np.mean(x, axis=0)
        sample_var = np.var(x, axis=0)
        
        x_centered = x - sample_mean
        std = np.sqrt(sample_var + eps)
        x_normalized = x_centered / std
        
        out = gamma * x_normalized + beta
        
        cache = (x, x_normalized, sample_mean, sample_var, std, gamma, beta, eps)
        
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: 
        #######################################################################
        x_centered = x - running_mean
        std = np.sqrt(running_var + eps)
        x_normalized = x_centered / std
        
        out = gamma * x_normalized + beta
        
        cache = (x, x_normalized, running_mean, running_var, std, gamma, beta, eps)
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """Обратный проход для пакетной нормализации.

Для этой реализации вам следует составить граф вычислений для
пакетной нормализации и распространить градиенты в обратном направлении .

Входные данные:
- dout: Производные от исходных узлов, форма (N, D)
- cache: Переменная промежуточных значений из batchnorm_forward.

Возвращает кортеж:
- dx: Градиент относительно входных данных x, форма (N, D)
- dgamma: Градиент относительно параметра масштабирования gamma, форма (D,)
- dbeta: Градиент относительно параметра сдвига beta, форма (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: 
    ###########################################################################
    x, x_normalized, sample_mean, sample_var, std, gamma, beta, eps = cache
    N = x.shape[0]
    
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * x_normalized, axis=0)    
    dx_normalized = dout * gamma
    
    dvar = np.sum(dx_normalized * (x - sample_mean) * (-0.5) * (sample_var + eps)**(-1.5), axis=0)
    dmean = np.sum(-dx_normalized / std, axis=0) + dvar * np.mean(-2 * (x - sample_mean), axis=0)
    dx = dx_normalized / std + dvar * 2 * (x - sample_mean) / N + dmean / N
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):

    """Для этой реализации вам следует вычислить производные для обратного прохода пакетной
    нормализации в виде простого выражения.
    Дополнительные подсказки см. в Jupyter Notebook.

    Примечание: Эта реализация должна ожидать получения той же переменной кэша,
    что и batchnorm_backward,
    но может не использовать все значения из кэша.

    Входы/выходы: Те же, что и batchnorm_backward"""
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: 
    ###########################################################################
    # 
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """Прямой проход для нормализации.

Как во время обучения, так и во время тестирования входящие данные нормализуются по точкам данных,
прежде чем масштабироваться с помощью параметров гамма и бета, идентичных параметрам пакетной нормализации.

Обратите внимание, что в отличие от пакетной нормализации, поведение во время обучения и тестирования для
нормализации слоя идентично, и нам не нужно отслеживать скользящие средние
любого рода.

Входные данные:
- x: Данные формы (N, D)
- gamma: Параметр масштабирования формы (D,)
- beta: Параметр сдвига формы (D,)
- ln_param: Словарь со следующими ключами:
- eps: Константа для обеспечения числовой стабильности

Возвращает кортеж:
- out: формы (N, D)
- cache: Кортеж значений, необходимых для обратного прохода
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: 
    ###########################################################################
    N = x.shape[0]
    
    mean = np.mean(x, axis=1, keepdims=True)
    var = np.var(x, axis=1, keepdims=True)
    
    x_centered = x - mean
    std = np.sqrt(var + eps)
    x_normalized = x_centered / std
    
    out = gamma * x_normalized + beta
    cache = (x, x_normalized, mean, var, std, gamma, beta, eps)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """Обратный проход для нормализации.

В этой реализации можно полагаться на уже проделанную работу
для пакетной нормализации.

Входные данные:
- dout: Производные от исходного слоя, форма (N, D)
- cache: Переменная промежуточных значений из layernorm_forward.
Возвращает кортеж:
- dx: Градиент относительно входных данных x, форма (N, D)
- dgamma: Градиент относительно параметра масштабирования gamma, форма (D,)
- dbeta: Градиент относительно параметра сдвига beta, форма (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: 
    ###########################################################################\n    # Распаковываем кэш\n    x, x_normalized, mean, var, std, gamma, beta, eps = cache\n    N = x.shape[1]  # количество признаков\n    \n    # dbeta и dgamma\n    dbeta = np.sum(dout, axis=1, keepdims=True).flatten()\n    dgamma = np.sum(dout * x_normalized, axis=1, keepdims=True).flatten()\n    \n    # dx_normalized\n    dx_normalized = dout * gamma\n    \n    # dvar\n    dvar = np.sum(dx_normalized * (x - mean) * (-0.5) * (var + eps)**(-1.5), axis=1, keepdims=True)\n    \n    # dmean\n    dmean = np.sum(-dx_normalized / std, axis=1, keepdims=True) + dvar * np.mean(-2 * (x - mean), axis=1, keepdims=True)\n    \n    # dx\n    dx = dx_normalized / std + dvar * 2 * (x - mean) / N + dmean / N\n    \n    ###########################################################################\n    #                             END OF YOUR CODE                            #\n    ###########################################################################\n    return dx, dgamma, dbeta(x, dropout_param):
    """Прямой проход Dropout

Обратите внимание, что это отличается от стандартной версии Dropout.
Здесь p — вероятность сохранения выходного сигнала нейрона, в отличие от
вероятности отбрасывания выходного сигнала нейрона.

Входные данные:
- x: Входные данные любой формы
- dropout_param: Словарь со следующими ключами:
- p: Параметр Dropout. Мы сохраняем каждый выходной сигнал нейрона с вероятностью p.
- mode: 'test' или 'train'. Если режим train, то выполняем Dropout;
если режим test, то просто возвращаем входные данные.
- seed: Начальное значение для генератора случайных чисел. Передача начального значения делает эту
функцию детерминированной, что необходимо для проверки градиента, но не
в реальных сетях.

Выходные данные:
- out: Массив той же формы, что и x.
- cache: кортеж (dropout_param, mask). В режиме обучения mask — это маска Dropout,
которая использовалась для умножения входных данных; в тестовом режиме mask — None.
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        #######################################################################
        # TODO: 
        #######################################################################
        mask = (np.random.rand(*x.shape) < p) / p
        out = x * mask
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: 
        #######################################################################
        out = x
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """Обратный проход дропаута.

Входные данные:
- dout: производные от вышестоящего потока 
- cache: (dropout_param, mask) из dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        #######################################################################
        # TODO: 
        #######################################################################
        dx = dout * mask
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == "test":
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """ реализация прямого прохода для сверточного слоя.

    Входные данные состоят из N точек данных, каждая c C каналами, высотой H и
    шириной W. Мы свертываем каждый вход F различными фильтрами, где каждый фильтр
    охватывает все C каналов и имеет высоту HH и ширину WW.

    Входные данные:
    - x: Входные данные формы (N, C, H, W)
    - w: Веса фильтров формы (F, C, HH, WW)
    - b: Смещение формы (F,)
    - conv_param: Словарь со следующими ключами:
    - 'stride': Количество пикселей между соседними рецептивными полями в
    горизонтальном и вертикальном направлениях.
    - 'pad': Количество пикселей, которые будут использоваться для заполнения входных данных нулями.
    Во время заполнения нули 'pad' должны располагаться симметрично (т.е. одинаково с обеих сторон)
    вдоль осей высоты и ширины входных данных. Будьте осторожны, чтобы не изменять исходные
    входные данные x напрямую.

    Возвращает кортеж:
    - out: Выходные данные формы (N, F, H', W'), где H' и W' задаются формулами:
        H' = 1 + (H + 2 * pad - HH) / stride
        W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Напишите реализацию свертки прямого прохода.                         #
    # Hint: можно использовать np.pad для паддинга.                      #
    ###########################################################################
    
    stride = conv_param['stride']
    pad = conv_param['pad']
    
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant', constant_values=0)
    
    H_out = 1 + (H + 2 * pad - HH) // stride
    W_out = 1 + (W + 2 * pad - WW) // stride
    
    out = np.zeros((N, F, H_out, W_out))
    
    for i in range(N):
        for f in range(F):
            for oh in range(H_out):
                for ow in range(W_out):
                    h_start = oh * stride
                    w_start = ow * stride
                    x_patch = x_padded[i, :, h_start:h_start+HH, w_start:w_start+WW]                    
                    out[i, f, oh, ow] = np.sum(x_patch * w[f]) + b[f]
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """ реализация обратного прохода для сверточного слоя.
    Входные данные:
    - dout: Производные от исходного слоя.
    - cache: Кортеж (x, w, b, conv_param), как в conv_forward_naive

    Возвращает кортеж:
    - dx: Градиент относительно x
    - dw: Градиент относительно w
    - db: Градиент относительно b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Реализация светки обратного прохода.                        #
    ###########################################################################
    
    x, w, b, conv_param = cache
    stride = conv_param['stride']
    pad = conv_param['pad']
    
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant', constant_values=0)
    
    H_out = 1 + (H + 2 * pad - HH) // stride
    W_out = 1 + (W + 2 * pad - WW) // stride
    
    dx_padded = np.zeros_like(x_padded)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)
    
    for i in range(N):
        for f in range(F):
            db[f] += np.sum(dout[i, f])
            for oh in range(H_out):
                for ow in range(W_out):
                    h_start = oh * stride
                    w_start = ow * stride
                    
                    x_patch = x_padded[i, :, h_start:h_start+HH, w_start:w_start+WW]
                    dw[f] += x_patch * dout[i, f, oh, ow]
                    
                    dx_padded[i, :, h_start:h_start+HH, w_start:w_start+WW] += w[f] * dout[i, f, oh, ow]
    
    dx = dx_padded[:, :, pad:pad+H, pad:pad+W]
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """реализация прямого прохода для слоя max-pooling.

    Входные данные:
    - x: Входные данные, форма (N, C, H, W)
    - pool_param: словарь со следующими ключами:
        - 'pool_height': Высота каждой области пулинга
        - 'pool_width': Ширина каждой области пулинга
    - 'stride': Расстояние между соседними областями пулинга

    Заполнение не требуется, например, можно предположить:
    - (H - pool_height) % stride == 0
    - (W - pool_width) % stride == 0

    Возвращает кортеж:
    - out: Выходные данные, форма (N, C, H', W'), где H' и W' задаются формулами:
    H' = 1 + (H - pool_height) / stride
    W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    
    N, C, H, W = x.shape
    
    H_out = 1 + (H - pool_height) // stride
    W_out = 1 + (W - pool_width) // stride
    
    out = np.zeros((N, C, H_out, W_out))
    
    for i in range(N):
        for c in range(C):
            for oh in range(H_out):
                for ow in range(W_out):
                    h_start = oh * stride
                    w_start = ow * stride
                    
                    x_patch = x[i, c, h_start:h_start+pool_height, w_start:w_start+pool_width]
                    
                    out[i, c, oh, ow] = np.max(x_patch)
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    реализация обратного прохода для max-pooling.
    Входные данные:
    - dout: Производные от исходного слоя.
    - cache: Кортеж (x, pool_param), как в прямом проходе

    Возвращает кортеж:
    - dx: Градиент относительно x
    
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    
    x, pool_param = cache
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    
    N, C, H, W = x.shape
    _, _, H_out, W_out = dout.shape
    
    dx = np.zeros_like(x)
    
    for i in range(N):
        for c in range(C):
            for oh in range(H_out):
                for ow in range(W_out):
                    h_start = oh * stride
                    w_start = ow * stride
                    
                    x_patch = x[i, c, h_start:h_start+pool_height, w_start:w_start+pool_width]
                    max_idx = np.unravel_index(np.argmax(x_patch), x_patch.shape)
                    dx[i, c, h_start + max_idx[0], w_start + max_idx[1]] += dout[i, c, oh, ow]
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """Вычисляет прямой проход для пространственной пакетной нормализации.

Входные данные:
- x: Входные данные формы (N, C, H, W)
- gamma: Параметр масштаба формы (C,)
- beta: Параметр сдвига формы (C,)
- bn_param: Словарь со следующими ключами:
- mode: 'train' или 'test'; обязательно
- eps: Константа для обеспечения численной стабильности
- momentum: Константа для скользящего среднего / дисперсии. momentum=0 означает, что
старая информация полностью отбрасывается на каждом шаге времени, а
momentum=1 означает, что новая информация никогда не включается.
Значение по умолчанию momentum=0.9 должно хорошо работать в большинстве ситуаций.
- running_mean: Массив формы (D,), содержащий скользящее среднее значений признаков
- running_var: Массив формы (D,), содержащий скользящую дисперсию значений признаков

Возвращает кортеж:
- out: Выходные данные формы (N, C, H, W)
- cache: Значения, необходимые для обратного прохода
    """
    out, cache = None, None

    ###########################################################################
    # TODO: 
    ###########################################################################
    mode = bn_param.get("mode", "train")
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)
    
    N, C, H, W = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(C, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(C, dtype=x.dtype))
    
    if mode == "train":
        x_reshaped = x.transpose(0, 2, 3, 1).reshape(N * H * W, C)
        sample_mean = np.mean(x_reshaped, axis=0)
        sample_var = np.var(x_reshaped, axis=0)
        
        x_centered = x - sample_mean.reshape(1, C, 1, 1)
        std = np.sqrt(sample_var + eps)
        x_normalized = x_centered / std.reshape(1, C, 1, 1)
        
        out = gamma.reshape(1, C, 1, 1) * x_normalized + beta.reshape(1, C, 1, 1)
        cache = (x, x_normalized, sample_mean, sample_var, std, gamma, beta, eps)
        
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var
        
        bn_param["running_mean"] = running_mean
        bn_param["running_var"] = running_var
    elif mode == "test":
        x_centered = x - running_mean.reshape(1, C, 1, 1)
        std = np.sqrt(running_var + eps)
        x_normalized = x_centered / std.reshape(1, C, 1, 1)
        
        out = gamma.reshape(1, C, 1, 1) * x_normalized + beta.reshape(1, C, 1, 1)
        cache = (x, x_normalized, running_mean, running_var, std, gamma, beta, eps)
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """Вычисляет обратный проход для пространственной пакетной нормализации.

    Входные данные:
    - dout: Производные от исходного алгоритма, формы (N, C, H, W)
    - cache: Значения из прямого прохода

    Возвращает кортеж:
    - dx: Градиент относительно входных данных, формы (N, C, H, W)
    - dgamma: Градиент относительно параметра масштаба, формы (C,)
    - dbeta: Градиент относительно параметра сдвига, формы (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: 
    ###########################################################################
    x, x_normalized, sample_mean, sample_var, std, gamma, beta, eps = cache
    N, C, H, W = x.shape
    
    dbeta = np.sum(dout, axis=(0, 2, 3))
    dgamma = np.sum(dout * x_normalized, axis=(0, 2, 3))
    
    x_centered = x - sample_mean.reshape(1, C, 1, 1)
    dx_normalized = dout * gamma.reshape(1, C, 1, 1)
    
    dvar = np.sum(dx_normalized * x_centered * (-0.5) * (sample_var + eps)**(-1.5), axis=(0, 2, 3))
    dmean = np.sum(-dx_normalized / std.reshape(1, C, 1, 1), axis=(0, 2, 3)) + dvar * np.mean(-2 * x_centered, axis=(0, 2, 3))
    dx = dx_normalized / std.reshape(1, C, 1, 1) + dvar.reshape(1, C, 1, 1) * 2 * x_centered / (N * H * W) + dmean.reshape(1, C, 1, 1) / (N * H * W)
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """Вычисляет прямой проход для групповой нормализации.

В отличие от послойной нормализации, групповая нормализация разбивает каждую запись в данных на G
смежных частей, которые затем нормализуются независимо. Затем к данным применяются сдвиг и масштабирование для каждой характеристики, аналогично пакетной нормализации и послойной
нормализации.

Входные данные:
- x: Входные данные формы (N, C, H, W)
- gamma: Параметр масштабирования формы (1, C, 1, 1)
- beta: Параметр сдвига формы (1, C, 1, 1)
- G: Целочисленное число групп для разделения, должно быть делителем C
- gn_param: Словарь со следующими ключами:
- eps: Константа для обеспечения числовой стабильности

Возвращает кортеж:
- out: Выходные данные формы (N, C, H, W)
- cache: Значения, необходимые для обратного прохода
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: 
    ###########################################################################
    N, C, H, W = x.shape
    
    group_size = C // G
    x_reshaped = x.reshape(N, G, group_size, H, W)
    
    mean = np.mean(x_reshaped, axis=(2, 3, 4), keepdims=True)
    var = np.var(x_reshaped, axis=(2, 3, 4), keepdims=True)
    
    x_centered = x_reshaped - mean
    std = np.sqrt(var + eps)
    x_normalized = x_centered / std
    
    gamma_reshaped = gamma.reshape(1, G, group_size, 1, 1)
    beta_reshaped = beta.reshape(1, G, group_size, 1, 1)
    
    out = gamma_reshaped * x_normalized + beta_reshaped
    
    out = out.reshape(N, C, H, W)
    
    cache = (x, x_reshaped, x_normalized, mean, var, std, gamma, beta, eps, G)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """Вычисляет обратный проход для групповой нормализации.
Входные данные:
- dout: Производные от исходных данных, форма (N, C, H, W)
- cache: Значения из прямого прохода

Возвращает кортеж:
- dx: Градиент относительно входных данных, форма (N, C, H, W)
- dgamma: Градиент относительно параметра масштаба, форма (1, C, 1, 1)
- dbeta: Градиент относительно параметра сдвига, форма (1, C, 1, 1)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: 
    ###########################################################################
    x, x_reshaped, x_normalized, mean, var, std, gamma, beta, eps, G = cache
    N, C, H, W = x.shape
    group_size = C // G
    
    dout_reshaped = dout.reshape(N, G, group_size, H, W)
    
    dbeta = np.sum(dout, axis=(0, 2, 3), keepdims=True)
    dgamma = np.sum(dout * x_normalized.reshape(N, C, H, W), axis=(0, 2, 3), keepdims=True)
    
    gamma_reshaped = gamma.reshape(1, G, group_size, 1, 1)
    
    dx_normalized = dout_reshaped * gamma_reshaped
    
    x_centered = x_reshaped - mean
    dvar = np.sum(dx_normalized * x_centered * (-0.5) * (var + eps)**(-1.5), axis=(2, 3, 4), keepdims=True)
    
    dmean = np.sum(-dx_normalized / std, axis=(2, 3, 4), keepdims=True) + dvar * np.mean(-2 * x_centered, axis=(2, 3, 4), keepdims=True)
    
    dx_reshaped = dx_normalized / std + dvar * 2 * x_centered / (group_size * H * W) + dmean / (group_size * H * W)
    
    dx = dx_reshaped.reshape(N, C, H, W)
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta
