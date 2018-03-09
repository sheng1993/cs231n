import numpy as np
from random import shuffle
import math

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
  # (N, D) x (D, C) => (N, C)
  scores = X.dot(W)
  # Numerical stability
  scores -= np.max(scores)

  num_train = X.shape[0]
  num_classes = W.shape[1]

  for i in range(num_train):
    sum_j = 0.0
    for j in range(num_classes):
      sum_j += math.exp(scores[i, j])

    # softmax function
    # softmax(k) = e(scores[k]) / sum_j(scores[j])
    f = lambda k: math.exp(scores[i, k]) / sum_j    
    prob_i = f(y[i])

    # Cross-entropy loss: -log(softmax)
    l_i = -1.0 * math.log(prob_i)
    loss += l_i

    for j in range(num_classes):
      prob_j = f(j)
      dW[:, j] += (prob_j - (j == y[i])) * X[i]


  # Regularization
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)

  dW /= num_train
  dW += reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

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
  # (N, D) x (D, C) => (N, C)
  scores = X.dot(W)
  # Numerical stability
  scores -= np.max(scores, keepdims=True)

  num_train = X.shape[0]

  sum_scores = np.sum(np.exp(scores), axis=1, keepdims=True)
  p = np.exp(scores)/sum_scores

  loss = np.sum(-np.log(p[np.arange(num_train), y]))

  ind = np.zeros_like(p)
  ind[np.arange(num_train), y] = 1
  dW = X.T.dot(p - ind)

  #Regularization
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)

  dW /= num_train
  dW += reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

