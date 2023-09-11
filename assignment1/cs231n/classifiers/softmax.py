from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    num_class = W.shape[1]
    for i in range(num_train):
      scores = X[i].dot(W) # (C, )
      y_exp = np.exp(scores - scores.max())
      softmax = y_exp / y_exp.sum() # (C, ) 
      loss -= np.log(softmax[y[i]])
      softmax[y[i]] -= 1
      dW += np.outer(X[i], softmax) # (D, 1) @ (1, C) -> (D, C)

    loss = loss / num_train + reg * (W**2).sum()
    dW = dW / num_train + 2 * reg * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # - W: A numpy array of shape (D, C) containing weights.
    # - X: A numpy array of shape (N, D) containing a minibatch of data.
    # - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    #   that X[i] has label c, where 0 <= c < C.
    # - reg: (float) regularization strength
    
    # Number of training samples
    N = X.shape[0]

    # Compute the loss
    scores = X @ W # (N, C)
    y_norm = scores - scores.max(axis=1, keepdims=True)
    y_exp = np.exp(y_norm)
    y_pred = y_exp / y_exp.sum(axis=1, keepdims=True) # (N, C)
    loss -= np.log(y_pred[range(N), y]).sum() / N + reg * (W**2).sum()

    # dW = 2 * reg * W
    # dy_pred = np.zeros_like(y_pred)
    # dy_pred[range(n), y] -= 1 / (y[range(n)] * n)
    # dy_exp = (1 / y_exp.sum(axis=1, keepdims=True)) * (1 - y_exp) * dy_pred
    # dy_norm = np.exp(y_norm) * dy_exp
    # dscores = np.ones_like(scores)
    # dscores[range(n), scores.argmax(axis=1)] = 0
    # dscores = dscores * dy_norm

    # Compute the gradient dW
    y_pred[range(N), y] -= 1
    dW = X.T @ y_pred / N + 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
