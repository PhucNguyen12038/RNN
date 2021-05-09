import numpy as np
from maths import sigmoid, softmax
from numpy.random import randn
from data import train_data, test_data
import random
from encoder import NumpyEncoder
import json
from feature_extraction import createVocabulary, createInput3D


class LSTM:

  def __init__(self, input_size, output_size, hidden_size=16):
    '''
    :param input_size: x
    :param output_size: y
    :param hidden_size: h
    Wf -- Weight matrix of the forget gate, numpy array of shape (h, h + x)
    bf -- Bias of the forget gate, numpy array of shape (h, 1)
    Wi -- Weight matrix of the update gate, numpy array of shape (h, h + x)
    bi -- Bias of the update gate, numpy array of shape (h, 1)
    Wc -- Weight matrix of the first "tanh", numpy array of shape (h, h + x)
    bc --  Bias of the first "tanh", numpy array of shape (h, 1)
    Wo -- Weight matrix of the output gate, numpy array of shape (h, h + x)
    bo --  Bias of the output gate, numpy array of shape (h, 1)
    Wy -- Weight matrix relating the hidden-state to the output, numpy array of shape (y, h)
    by -- Bias relating the hidden-state to the output, numpy array of shape (y, 1)
    '''
    self.Wf = randn(hidden_size, hidden_size + input_size)
    self.Wi = randn(hidden_size, hidden_size + input_size)
    self.Wc = randn(hidden_size, hidden_size + input_size)
    self.Wo = randn(hidden_size, hidden_size + input_size)
    self.Wy = randn(output_size, hidden_size)

    self.bf = np.zeros((hidden_size, 1))
    self.bi = np.zeros((hidden_size, 1))
    self.bc = np.zeros((hidden_size, 1))
    self.bo = np.zeros((hidden_size, 1))
    self.by = np.zeros((output_size, 1))

    self.a0 = randn(hidden_size, 1)

  def lstm_cell_forward(self, xt, a_prev, c_prev):
    '''
    Arguments:
    xt -- your input data at timestep "t", numpy array of shape (n_x, m).
    a_prev -- Hidden state at timestep "t-1", numpy array of shape (n_a, m)
    c_prev -- Memory state at timestep "t-1", numpy array of shape (n_a, m)
    parameters -- python dictionary containing:
                        Wf -- Weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                        bf -- Bias of the forget gate, numpy array of shape (n_a, 1)
                        Wi -- Weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                        bi -- Bias of the update gate, numpy array of shape (n_a, 1)
                        Wc -- Weight matrix of the first "tanh", numpy array of shape (n_a, n_a + n_x)
                        bc --  Bias of the first "tanh", numpy array of shape (n_a, 1)
                        Wo -- Weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                        bo --  Bias of the output gate, numpy array of shape (n_a, 1)
                        Wy -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)

    Returns:
    a_next -- next hidden state, of shape (n_a, m)
    c_next -- next memory state, of shape (n_a, m)
    yt_pred -- prediction at timestep "t", numpy array of shape (n_y, m)
    cache -- tuple of values needed for the backward pass, contains (a_next, c_next, a_prev, c_prev, xt, parameters)

    Note: ft/it/ot stand for the forget/update/output gates, cct stands for the candidate value (c tilde),
          c stands for the memory value
    '''

    # Retrieve dimensions from shapes of xt and Wy
    n_x, m = xt.shape
    n_y, n_a = self.Wy.shape

    # Concatenate a_prev and xt (≈3 lines)
    concat = np.zeros([n_x + n_a, m])
    concat[: n_a, :] = a_prev
    concat[n_a:, :] = xt

    # Compute values for ft, it, cct, c_next, ot, a_next using the formulas given figure (4) (≈6 lines)
    ft = sigmoid(np.dot(self.Wf, concat) + self.bf)
    it = sigmoid(np.dot(self.Wi, concat) + self.bi)
    cct = np.tanh(np.dot(self.Wc, concat) + self.bc)
    c_next = ft * c_prev + it * cct
    ot = sigmoid(np.dot(self.Wo, concat) + self.bo)
    a_next = ot * np.tanh(c_next)

    # Compute prediction of the LSTM cell (≈1 line)
    yt_pred = softmax(np.dot(self.Wy, a_next) + self.by)

    # store values needed for backward propagation in cache
    cache = (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt)

    return a_next, c_next, yt_pred, cache

  def lstm_forward(self, x):
    '''
    Arguments:
    x -- Input data for every time-step, of shape (n_x, m, T_x).
    a0 -- Initial hidden state, of shape (n_a, m)
    parameters -- python dictionary containing:
                        Wf -- Weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                        bf -- Bias of the forget gate, numpy array of shape (n_a, 1)
                        Wi -- Weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                        bi -- Bias of the update gate, numpy array of shape (n_a, 1)
                        Wc -- Weight matrix of the first "tanh", numpy array of shape (n_a, n_a + n_x)
                        bc -- Bias of the first "tanh", numpy array of shape (n_a, 1)
                        Wo -- Weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                        bo -- Bias of the output gate, numpy array of shape (n_a, 1)
                        Wy -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)

    Returns:
    a -- Hidden states for every time-step, numpy array of shape (n_a, m, T_x)
    y -- Predictions for every time-step, numpy array of shape (n_y, m, T_x)
    caches -- tuple of values needed for the backward pass, contains (list of all the caches, x)
    '''

    # Initialize "caches", which will track the list of all the caches
    caches = []

    # Retrieve dimensions from shapes of x and parameters['Wy'] (≈2 lines)
    n_x, m, T_x = x.shape
    n_y, n_a = self.Wy.shape

    # initialize "a", "c" and "y" with zeros (≈3 lines)
    a = np.zeros([n_a, m, T_x])
    c = np.zeros([n_a, m, T_x])
    y = np.zeros([n_y, m, T_x])

    # Initialize a_next and c_next (≈2 lines)
    a_next = self.a0
    c_next = np.zeros([n_a, m])

    # loop over all time-steps
    for t in range(T_x):
      # Update next hidden state, next memory state, compute the prediction, get the cache (≈1 line)
      a_next, c_next, yt, cache = self.lstm_cell_forward(x[:, :, t], a_next, c_next)
      # Save the value of the new "next" hidden state in a (≈1 line)
      a[:, :, t] = a_next
      # Save the value of the prediction in y (≈1 line)
      y[:, :, t] = yt
      # Save the value of the next cell state (≈1 line)
      c[:, :, t] = c_next
      # Append the cache into caches (≈1 line)
      caches.append(cache)

    # store values needed for backward propagation in cache
    caches = (caches, x)

    return a, y, c, caches

  def lstm_cell_backward(self, da_next, dc_next, cache):
    '''
    Implement the backward pass for the LSTM-cell (single time-step).
    Arguments:
    da_next -- Gradients of next hidden state, of shape (n_a, m)
    dc_next -- Gradients of next cell state, of shape (n_a, m)
    cache -- cache storing information from the forward pass
    Returns:
    gradients -- python dictionary containing:
                        dxt -- Gradient of input data at time-step t, of shape (n_x, m)
                        da_prev -- Gradient w.r.t. the previous hidden state, numpy array of shape (n_a, m)
                        dc_prev -- Gradient w.r.t. the previous memory state, of shape (n_a, m, T_x)
                        dWf -- Gradient w.r.t. the weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                        dWi -- Gradient w.r.t. the weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                        dWc -- Gradient w.r.t. the weight matrix of the memory gate, numpy array of shape (n_a, n_a + n_x)
                        dWo -- Gradient w.r.t. the weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                        dbf -- Gradient w.r.t. biases of the forget gate, of shape (n_a, 1)
                        dbi -- Gradient w.r.t. biases of the update gate, of shape (n_a, 1)
                        dbc -- Gradient w.r.t. biases of the memory gate, of shape (n_a, 1)
                        dbo -- Gradient w.r.t. biases of the output gate, of shape (n_a, 1)
    '''

    # Retrieve information from "cache"
    (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt) = cache

    # Retrieve dimensions from xt's and a_next's shape (≈2 lines)
    n_x, m = xt.shape
    n_a, m = a_next.shape

    # Compute gates related derivatives, you can find their values can be found by looking carefully at equations (7) to (10) (≈4 lines)
    dot = da_next * np.tanh(c_next) * ot * (1 - ot)
    dcct = (dc_next * it + ot * (1 - np.square(np.tanh(c_next))) * it * da_next) * (1 - np.square(cct))
    dit = (dc_next * cct + ot * (1 - np.square(np.tanh(c_next))) * cct * da_next) * it * (1 - it)
    dft = (dc_next * c_prev + ot * (1 - np.square(np.tanh(c_next))) * c_prev * da_next) * ft * (1 - ft)

    # Compute parameters related derivatives. Use equations (11)-(14) (≈8 lines)
    concat = np.concatenate((a_prev, xt), axis=0).T
    dWf = np.dot(dft, concat)
    dWi = np.dot(dit, concat)
    dWc = np.dot(dcct, concat)
    dWo = np.dot(dot, concat)
    dbf = np.sum(dft, axis=1, keepdims=True)
    dbi = np.sum(dit, axis=1, keepdims=True)
    dbc = np.sum(dcct, axis=1, keepdims=True)
    dbo = np.sum(dot, axis=1, keepdims=True)

    # Compute derivatives w.r.t previous hidden state, previous memory state and input. Use equations (15)-(17). (≈3 lines)
    da_prev = np.dot(self.Wf[:, :n_a].T, dft) + np.dot(self.Wc[:, :n_a].T, dcct) + np.dot(
      self.Wi[:, :n_a].T, dit) + np.dot(self.Wo[:, :n_a].T, dot)
    dc_prev = dc_next * ft + ot * (1 - np.square(np.tanh(c_next))) * ft * da_next
    dxt = np.dot(self.Wf[:, n_a:].T, dft) + np.dot(self.Wc[:, n_a:].T, dcct) + np.dot(
      self.Wi[:, n_a:].T, dit) + np.dot(self.Wo[:, n_a:].T, dot)

    # Save gradients in dictionary
    gradients = {"dxt": dxt, "da_prev": da_prev, "dc_prev": dc_prev, "dWf": dWf, "dbf": dbf, "dWi": dWi, "dbi": dbi,
                 "dWc": dWc, "dbc": dbc, "dWo": dWo, "dbo": dbo}

    return gradients

  def lstm_backward(self, da, caches, learning_rate):

    '''
    Implement the backward pass for the RNN with LSTM-cell (over a whole sequence).
    Arguments:
    da -- Gradients w.r.t the hidden states, numpy-array of shape (n_a, m, T_x)
    caches -- cache storing information from the forward pass (lstm_forward)
    Returns:
    gradients -- python dictionary containing:
                        dx -- Gradient of inputs, of shape (n_x, m, T_x)
                        da0 -- Gradient w.r.t. the previous hidden state, numpy array of shape (n_a, m)
                        dWf -- Gradient w.r.t. the weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                        dWi -- Gradient w.r.t. the weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                        dWc -- Gradient w.r.t. the weight matrix of the memory gate, numpy array of shape (n_a, n_a + n_x)
                        dWo -- Gradient w.r.t. the weight matrix of the save gate, numpy array of shape (n_a, n_a + n_x)
                        dbf -- Gradient w.r.t. biases of the forget gate, of shape (n_a, 1)
                        dbi -- Gradient w.r.t. biases of the update gate, of shape (n_a, 1)
                        dbc -- Gradient w.r.t. biases of the memory gate, of shape (n_a, 1)
                        dbo -- Gradient w.r.t. biases of the save gate, of shape (n_a, 1)
    '''

    # Retrieve values from the first cache (t=1) of caches.
    (caches, x) = caches
    (a1, c1, a0, c0, f1, i1, cc1, o1, x1) = caches[0]

    # Retrieve dimensions from da's and x1's shapes (≈2 lines)
    n_a, m, T_x = da.shape
    n_x, m = x1.shape

    # initialize the gradients with the right sizes (≈12 lines)
    dx = np.zeros([n_x, m, T_x])
    da0 = np.zeros([n_a, m])
    da_prevt = np.zeros([n_a, m])
    dc_prevt = np.zeros([n_a, m])
    dWf = np.zeros([n_a, n_a + n_x])
    dWi = np.zeros([n_a, n_a + n_x])
    dWc = np.zeros([n_a, n_a + n_x])
    dWo = np.zeros([n_a, n_a + n_x])
    dbf = np.zeros([n_a, 1])
    dbi = np.zeros([n_a, 1])
    dbc = np.zeros([n_a, 1])
    dbo = np.zeros([n_a, 1])

    # loop back over the whole sequence
    for t in reversed(range(T_x)):
      # Compute all gradients using lstm_cell_backward
      gradients = self.lstm_cell_backward(da[:, :, t], dc_prevt, caches[t])
      # Store or add the gradient to the parameters' previous step's gradient
      dx[:, :, t] = gradients['dxt']
      dWf = dWf + gradients['dWf']
      dWi = dWi + gradients['dWi']
      dWc = dWc + gradients['dWc']
      dWo = dWo + gradients['dWo']
      dbf = dbf + gradients['dbf']
      dbi = dbi + gradients['dbi']
      dbc = dbc + gradients['dbc']
      dbo = dbo + gradients['dbo']
    # Set the first activation's gradient to the backpropagated gradient da_prev.
    da0 = gradients['da_prev']

    # Update
    self.Wf -= learning_rate * dWf
    self.Wi -= learning_rate * dWi
    self.Wc -= learning_rate * dWc
    self.Wo -= learning_rate * dWo
    self.bf -= learning_rate * dbf
    self.bi -= learning_rate * dbi
    self.bc -= learning_rate * dbc
    self.bo -= learning_rate * dbo

    self.a0 -= learning_rate * da0

  def saveModel(self):

    # Weight and bias
    WB_dict = {'Wf': self.Wf, 'Wi': self.Wi, 'Wc': self.Wc, 'Wo': self.Wo, 'h0': self.last_hs[0],
               'bf': self.bf, 'bi': self.bi, 'bc': self.bc, 'bo': self.bo, 'c0': self.last_cs[0]  }

    dumped = json.dumps(WB_dict, cls=NumpyEncoder)
    with open("model_lstm.json", "w") as fp:
      json.dump(dumped, fp)

    fp.close()

  def loadModel(self):
    with open('model_lstm.json', 'r') as f:
      jsonData = json.load(f)
    WB_dict = json.loads(jsonData)
    f.close()
    return WB_dict


# def initiateLSTM():
#   lstm = LSTM(vocab_size, 2)
#   lstmTrain(lstm,900)

def lstmProcessData(lstm, data, backprop=True):
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
    inputs = createInput3D(x)
    target = int(y)

    _, n_a = lstm.Wy.shape

    # Forward

    a, y, c, caches = lstm.lstm_forward(inputs)
    probs = y

    # Calculate loss / accuracy
    # Scale probs to pred array2d (2 x 1)
    pred = np.average(probs, axis=2)
    loss -= np.log(pred[target])
    num_correct += int(np.argmax(pred) == target)

    if backprop:
      # Build dL/dy
      d_L_d_y = probs
      d_L_d_y[target] -= 1
      d_L_d_c = 1

      # Backward
      T_x = inputs.shape[2]

      d_L_d_a = np.zeros((n_a, 1, T_x))
      for i in range(T_x):
        d_L_d_a[:,:,i] = np.dot(lstm.Wy.T, d_L_d_y[:,:,i])

      lstm.lstm_backward(d_L_d_a, caches, learning_rate=0.5)


  return loss / len(data), num_correct / len(data)

def lstmTrain(lstm, numLoop):
  '''
  Train data and save model after training
  :param numLoop: number of training loop
  :return:
  '''
  # Training loop
  for epoch in range(numLoop):
    train_loss, train_acc = lstmProcessData(lstm, train_data)

    if epoch % 100 == 99:
      print('--- Epoch %d' % (epoch + 1))
      print('Train:\tLoss %.3f | Accuracy: %.3f' % (train_loss, train_acc))

      test_loss, test_acc = lstmProcessData(lstm, test_data, backprop=False)
      print('Test:\tLoss %.3f | Accuracy: %.3f' % (test_loss, test_acc))

  # Save model

# initiateLSTM()