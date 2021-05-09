import numpy as np


def sigmoid(x):
  return 1 / (1 + np.exp(-x))


def softmax(xs):
  # Applies the Softmax Function to the input array.
  return np.exp(xs) / sum(np.exp(xs))

