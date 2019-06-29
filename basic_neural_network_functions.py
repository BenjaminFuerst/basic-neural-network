import numpy as np


def initialize_parameters(layer_dims):
    """
    Implements initialization of network layers
    
    Receives:
    layer_dims -- python array containing the dimensions of each layer in the network (counting input layer)
    
    Returns:
    parameters -- python dictionary containing the parameters W1, b1, ..., WL, bL
                    Wl -- weight matrix, numpy array with shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector, numpy array with shape (layer_dims[l], 1)
    """
    
    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)
    
    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(1/layer_dims[l-1])
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))
        
    return parameters


def sigmoid(Z, use_cache=True):
    """
    Implements the sigmoid function
    
    Receives:
    Z -- output of the linear function, numpy array with shape (size of current layer, number of examples)
    use_cache -- if true, caches Z
    
    Returns:
    A -- output of sigmoid(Z), numpy array with shape (size of current layer, number of examples)
    cache -- caches Z, useful during backpropagation
    """
    
    A = 1/(1+np.exp(-Z))
    
    if use_cache:
        cache = Z
        return A, cache
    else:
        return A


def relu(Z, use_cache=True):
    """
    Implements the ReLU function
    
    Receives:
    Z -- output of the linear function, numpy array with shape (size of current layer, number of examples)
    use_cache -- if true, caches Z
    
    Returns:
    A -- output of ReLU(Z), numpy array with shape (size of current layer, number of examples)
    cache -- caches Z, useful during backpropagation
    """
    
    A = np.maximum(0, Z)
    
    if use_cache:
        cache = Z
        return A, cache
    else:
        return A


def linear_forward(A_prev, W, b, use_cache=True):
    """
    Implements the linear part of forward propagation
    
    Receives:
    A_prev -- activations from previous layer (or input layer), numpy array with shape (size of previous layer, number of examples)
    W -- weight matrix, numpy array with shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array with shape (size of current layer, 1)
    use_cache -- if true, caches A_prev, W and b
    
    Returns:
    Z -- output of the linear function, numpy array with shape (size of current layer, number of examples)
    cache -- caches A_prev, W and b, useful during backpropagation
    """
    
    Z = np.dot(W, A_prev) + b
    
    if use_cache:
        cache = (A_prev, W, b)
        return Z, cache
    else:
        return Z


def linear_activation_forward(A_prev, W, b, activation, use_cache=True):
    """
    Implements the activation part of forward propagation
    
    Receives:
    A_prev -- activations from previous layer (or input layer), numpy array with shape (size of previous layer, number of examples)
    W -- weight matrix, numpy array with shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array with shape (size of current layer, 1)
    activation -- the activation to be used in this layer, can be "sigmoid" or "relu"
    use_cache -- if true, caches linear_cache and activation_cache
    
    Returns:
    A -- output of the activation function, numpy array with shape (size of current layer, number of examples)
    cache -- caches linear_cache and activation_cache, useful during backpropagation
    """
    
    if use_cache:
        Z, linear_cache = linear_forward(A_prev, W, b, use_cache=True)
        if activation == "sigmoid":
            A, activation_cache = sigmoid(Z, use_cache=True)
        elif activation == "relu":
            A, activation_cache = relu(Z, use_cache=True)
        cache = (linear_cache, activation_cache)
        return A, cache
    else:
        Z = linear_forward(A_prev, W, b, use_cache=False)
        if activation == "sigmoid":
            A = sigmoid(Z, use_cache=False)
        elif activation == "relu":
            A = relu(Z, use_cache=False)
        return A


def forward_propagation(X, parameters, use_cache=True):
    """
    Implements forward propagation
    
    Receives:
    X -- data, numpy array with shape (size of input layer, number of examples)
    parameters -- python dictionary containing parameters of the model
    use_cache -- if true, caches every cache of linear_activation_forward()
    
    Returns:
    AL -- last output of the activation function
    caches -- python array containing every cache of linear_activation_forward(), useful during backpropagation
    """
    
    L = len(parameters) // 2
    A = X
    
    if use_cache:
        caches = []
        for l in range(1, L):
            A_prev = A
            A, cache = linear_activation_forward(A_prev, parameters["W" + str(l)], parameters["b" + str(l)], "relu", use_cache=True)
            caches.append(cache)
        AL, cache = linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], "sigmoid", use_cache=True)
        caches.append(cache)
        return AL, caches
    else:
        for l in range(1, L):
            A_prev = A
            A = linear_activation_forward(A_prev, parameters["W" + str(l)], parameters["b" + str(l)], "relu", use_cache=False)
        AL = linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], "sigmoid", use_cache=False)
        return AL


def compute_cost(AL, Y):
    """
    Implements the cost function
    
    Receives:
    AL -- probability vector corresponding to label predictions, numpy array with shape (output size, number of examples)
    Y -- true label vector, numpy array with shape (output size, number of examples)
    
    Returns:
    cost -- cross-entropy cost, scalar
    """
    
    m = AL.shape[1]
    
    cost = -1/m * np.sum(Y*np.log(AL) + (1-Y)*np.log(1-AL))
    cost = np.squeeze(cost)
    
    return cost


def sigmoid_backward(dA, cache):
    """
    Implements backward propagation for a sigmoid unit
    
    Receives:
    dA -- post-activation gradient, numpy array with shape (size of current layer, number of examples)
    cache -- Z, stored from linear_activation_forward() (of current layer l)
    
    Returns:
    dZ -- gradient of the cost with respect to Z, numpy array with shape (size of current layer, number of examples)
    """
    
    Z = cache
    s = sigmoid(Z, use_cache=False)
    dZ = dA * s * (1-s)
    
    return dZ


def relu_backward(dA, cache):
    """
    Implements backward propagation for a ReLU unit
    
    Receives:
    dA -- post-activation gradient, numpy array with shape (size of current layer, number of examples)
    cache -- Z, stored from linear_activation_forward() (of current layer l)
    
    Returns:
    dZ -- gradient of the cost with respect to Z, numpy array with shape (size of current layer, number of examples)
    """
    
    Z = cache
    Z[Z <= 0] = 0
    Z[Z > 0] = 1
    dZ = dA * Z
    
    return dZ


def linear_backward(dZ, cache):
    """
    Implements the linear part of backward propagation
    
    Receives:
    dZ -- gradient of the cost with respect to Z, numpy array with shape (size of current layer, number of examples)
    cache -- tuple of values (A_prev, W, b), stored from linear_forward() (of current layer l)
    
    Returns:
    dA_prev -- gradient of the cost with respect to the activation (of previous layer l-1), numpy array with same shape as A_prev
    dW -- gradient of the cost with respect to W (of current layer l), numpy array with same shape as W
    db -- gradient of the cost with respect to b (of current layer l), numpy array with same shape as b
    """
    
    A_prev, W, b = cache
    m = A_prev.shape[1]
    
    dW = 1/m * np.dot(dZ, A_prev.T)
    db = 1/m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    
    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    """
    Implements the activation part of backward propagation
    
    Receives:
    dA -- post-activation gradient, numpy array with shape (size of current layer, number of examples)
    cache -- tuple of values (linear_cache, activation_cache), stored from linear_activation_forward() (of current layer l)
    activation -- the activation to be used in this layer, can be "sigmoid" or "relu"
    
    Returns:
    dA_prev -- gradient of the cost with respect to the activation (of previous layer l-1), numpy array with same shape as A_prev
    dW -- gradient of the cost with respect to W (of current layer l), numpy array with same shape as W
    db -- gradient of the cost with respect to b (of current layer l), numpy array with same shape as b
    """
    
    linear_cache, activation_cache = cache
    
    
    if activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    elif activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, db


def backward_propagation(AL, Y, caches):
    """
    Implements backward propagation
    
    Receives:
    AL -- probability vector corresponding to label predictions, numpy array with shape (output size, number of examples)
    Y -- true label vector, numpy array with shape (output size, number of examples)
    caches -- python array containing every cache of linear_activation_forward()
    
    Returns:
    grads -- python dictionary containing gradients
    """
    
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    
    dAL = -(np.divide(Y, AL) - np.divide(1-Y, 1-AL))
    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation="sigmoid")
    
    for l in range(L-1, 0, -1):
        current_cache = caches[l-1]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l)], current_cache, activation="relu")
        grads["dA" + str(l-1)] = dA_prev_temp
        grads["dW" + str(l)] = dW_temp
        grads["db" + str(l)] = db_temp
    
    return grads


def update_parameters(parameters, grads, learning_rate):
    """
    Updates parameters using gradient descent
    
    Receives:
    parameters -- python dictionary containing parameters
    grads -- python dictionary containing gradients
    
    Returns:
    parameters -- python dictionary containing updated parameters
    """
    
    L = len(parameters) // 2
    
    for l in range(1, L+1):
        parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * grads["db" + str(l)]
    
    return parameters


def predict(X, Y, parameters):
    """
    Implements predicting for testing
    
    Receives:
    X -- data, numpy array with shape (size of input layer, number of examples)
    Y -- true label vector, numpy array with shape (output size, number of examples)
    parameters -- python dictionary containing parameters
    
    Returns:
    p -- predictions for X
    """
    
    L = len(parameters) // 2
    m = X.shape[1]
    
    p = np.zeros((1, m))
    probabilities = forward_propagation(X, parameters, use_cache=False)
    
    for i in range(m):
        if probabilities[0,i] > 0.5:
            p[0,i] = 1
    
    print("Accuracy: " + str(np.sum(p == Y)/m))
    
    return p
