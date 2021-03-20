import numpy as np

def softmax(xs):
  # Applies the Softmax Function to the input array.
  return np.exp(xs) / sum(np.exp(xs))


a = -2.6
b = 2.6
c = np.array([a,b])
s = softmax(c)
print(s)
