import json
import numpy as np
from numpy.random import randn
from encoder import NumpyEncoder
from feature_extraction import createVocabulary, createInputList, sanitizer
import random
from maths import softmax

def softmax(xs):
  # Applies the Softmax Function to the input array.
  return np.exp(xs) / sum(np.exp(xs))

class RNN:
  # A many-to-one Vanilla Recurrent Neural Network.

  def __init__(self, input_size, output_size, hidden_size=64):
    # Weights
    self.Whh = randn(hidden_size, hidden_size) / 1000
    self.Wxh = randn(hidden_size, input_size) / 1000
    self.Why = randn(output_size, hidden_size) / 1000

    # Biases
    self.bh = np.zeros((hidden_size, 1))
    self.by = np.zeros((output_size, 1))

  def forward(self, inputs):
    '''
    Perform a forward pass of the RNN using the given inputs.
    Returns the final output and hidden state.
    - inputs is an array of one hot vectors with shape (input_size, 1).
    '''
    h = np.zeros((self.Whh.shape[0], 1))

    self.last_inputs = inputs
    self.last_hs = {0: h}

    # Perform each step of the RNN
    for i, x in enumerate(inputs):
      h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)
      self.last_hs[i + 1] = h

    # Compute the output
    y = self.Why @ h + self.by

    return y, h

  def backprop(self, d_y, learn_rate=2e-2):
    '''
    Perform a backward pass of the RNN.
    - d_y (dL/dy) has shape (output_size, 1).
    - learn_rate is a float.
    '''
    n = len(self.last_inputs)

    # Calculate dL/dWhy and dL/dby.
    d_Why = d_y @ self.last_hs[n].T
    d_by = d_y

    # Initialize dL/dWhh, dL/dWxh, and dL/dbh to zero.
    d_Whh = np.zeros(self.Whh.shape)
    d_Wxh = np.zeros(self.Wxh.shape)
    d_bh = np.zeros(self.bh.shape)

    # Calculate dL/dh for the last h.
    # dL/dh = dL/dy * dy/dh
    d_h = self.Why.T @ d_y

    # Backpropagate through time.
    for t in reversed(range(n)):
      # An intermediate value: dL/dh * (1 - h^2)
      temp = ((1 - self.last_hs[t + 1] ** 2) * d_h)

      # dL/db = dL/dh * (1 - h^2)
      d_bh += temp

      # dL/dWhh = dL/dh * (1 - h^2) * h_{t-1}
      d_Whh += temp @ self.last_hs[t].T

      # dL/dWxh = dL/dh * (1 - h^2) * x
      d_Wxh += temp @ self.last_inputs[t].T

      # Next dL/dh = dL/dh * (1 - h^2) * Whh
      d_h = self.Whh @ temp

    # Clip to prevent exploding gradients.
    for d in [d_Wxh, d_Whh, d_Why, d_bh, d_by]:
      np.clip(d, -1, 1, out=d)

    # Update weights and biases using gradient descent.
    self.Whh -= learn_rate * d_Whh
    self.Wxh -= learn_rate * d_Wxh
    self.Why -= learn_rate * d_Why
    self.bh -= learn_rate * d_bh
    self.by -= learn_rate * d_by

  def saveModel(self):
    WB_dict = {'Wxh': self.Wxh, 'Whh': self.Whh, 'Why': self.Why,
               'bh': self.bh, 'by': self.by}

    dumped = json.dumps(WB_dict, cls=NumpyEncoder)
    with open("model_rnn.json", "w") as fp:
      json.dump(dumped, fp)

    fp.close()

  def loadModel(self):
    with open('model_rnn.json', 'r') as f:
      jsonData = json.load(f)
    WB_dict = json.loads(jsonData)
    f.close()
    return WB_dict
# End class RNN

def rnnProcessData(rnn, data, backprop=True):
  '''
  Returns the RNN's loss and accuracy for the given data.
  - data is a dictionary mapping text to True or False.
  - backprop determines if the backward phase should be run.
  '''
  items = list(data.items())
  random.shuffle(items)

  loss = 0
  num_correct = 0

  for x, y in items:
    inputs = createInputList(x, vocab_size, word_to_idx)
    target = int(y[0])

    # Forward
    out, _ = rnn.forward(inputs)
    probs = softmax(out)

    # Calculate loss / accuracy
    loss -= np.log(probs[target])
    num_correct += int(np.argmax(probs) == target)

    if backprop:
      # Build dL/dy
      d_L_d_y = probs
      d_L_d_y[target] -= 1

      # Backward
      rnn.backprop(d_L_d_y)

  return loss / len(data), num_correct / len(data)


def rnnTrain(rnn, train_data, numLoop):
  '''
  Train data and save model after training
  :param train_data:
  :param rnn:
  :param numLoop: number of training loop
  :return:
  '''
  # Training loop
  for epoch in range(numLoop):
    train_loss, train_acc = rnnProcessData(rnn, train_data)

    if epoch % 100 == 99:
      print('--- Epoch %d' % (epoch + 1))
      print('Train:\tLoss %.3f | Accuracy: %.3f' % (train_loss, train_acc))

      # test_loss, test_acc = rnnProcessData(rnn, test_data, backprop=False)
      # print('Test:\tLoss %.3f | Accuracy: %.3f' % (test_loss, test_acc))

  # Save model
  rnn.saveModel()


def get_train_data():
  path = '/Users/nguyenphuc/Documents/Python/SocialMediaBullyDetect/social_media_cyberbullying_detection/RNN_source/social_media_cyberbullying_detection/datasets/MMHS/'
  fn = 'train_data_text_labels.json'

  f = open(path + fn, 'r')
  data = json.load(f)  # size is 1002
  f.close()
  return data


train_data = get_train_data()

vocab_size, word_to_idx = createVocabulary(train_data)

rnn = RNN(vocab_size, 6)
rnnTrain(rnn, train_data, 400)


