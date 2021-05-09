import numpy as np

def softmax(xs):
  # Applies the Softmax Function to the input array.
  return np.exp(xs) / sum(np.exp(xs))


a = -2.6
b = 2.6
c = np.array([a,b])

xt = np.random.randn(3,4)
a_prev = np.random.randn(5,4)
print(xt)
print(a_prev)

concat = np.zeros((5+3,4))
concat[: 5, :] = a_prev
concat[5 :, :] = xt
print(concat)
dit = (dc_next * cct + ot * (1 - np.square(np.tanh(c_next))) * cct * da_next) * it * (1 - it)

dcct = (dc_next * it + ot * (1 - np.square(np.tanh(c_next))) * it * da_next) * (1 - np.square(cct))