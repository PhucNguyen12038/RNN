import numpy as np
from numpy.random import randn
from maths import sigmoid
from encoder import NumpyEncoder
import json

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
    self.Wf = randn(hidden_size, hidden_size + input_size) / 1000
    self.Wi = randn(hidden_size, hidden_size + input_size) / 1000
    self.Wc = randn(hidden_size, hidden_size + input_size) / 1000
    self.Wo = randn(hidden_size, hidden_size + input_size) / 1000
    self.Wy = randn(output_size, hidden_size) / 1000

    self.bf = np.zeros((hidden_size, 1))
    self.bi = np.zeros((hidden_size, 1))
    self.bc = np.zeros((hidden_size, 1))
    self.bo = np.zeros((hidden_size, 1))
    self.by = np.zeros((output_size, 1))


  def forward(self, inputs):
    '''
    Perform a forward pass of LSTM using the given inputs.
    :param inputs: is an array of one hot vectors with shape (input_size, 1).
    :return: final output and hidden state.
    '''
    n_y, n_h = self.Wy.shape
    y = np.zeros((n_y, 1))
    h = np.zeros((n_h, 1))
    c = np.zeros((n_h, 1))

    self.last_hs = {0: h}  # save hidden state
    self.last_cs = {0: c}  # save cell state

    self.last_inputs = inputs


    self.caches = {}

    # Perform each step of the LSTM
    for i, x in enumerate(inputs):
      n_x, m = x.shape

      # Concatenate hidden state and input
      concat = np.zeros((n_x + n_h, 1))
      concat[: n_h, :] = h
      concat[n_h :, :] = x

      ft = sigmoid(np.dot(self.Wf, concat) + self.bf)
      it = sigmoid(np.dot(self.Wi, concat) + self.bi)
      cct = np.tanh(np.dot(self.Wc, concat) + self.bc)
      c = ft * c + it * cct
      self.last_cs[i+1] = c
      ot = sigmoid(np.dot(self.Wo, concat) + self.bo)
      h = ot*np.tanh(c)
      self.last_hs[i+1] = h
      cache = (h, c, self.last_hs[i], self.last_cs[i], ft, it, cct, ot, x)
      self.caches[i] = cache
    y = np.dot(self.Wy, h) + self.by
    return y

  def backprop(self, d_y, d_c, learn_rate=2e-2):
    '''
    Perform a backward pass of the LSTM.
    :param d_y: (dL/dy) has shape (output_size, 1).
    :param learn_rate: is a float.
    :return:
    '''
    n = len(self.last_inputs)
    d_Wy = d_y @ self.last_hs[n].T
    d_by = d_y
    d_y_d_h = self.Wy
    n_h, m = self.Wf.shape
    n_x = m - n_h

    dWf = np.zeros(self.Wf.shape)
    dbf = np.zeros(self.bf.shape)
    dWi = np.zeros(self.Wi.shape)
    dbi = np.zeros(self.bi.shape)
    dWo = np.zeros(self.Wo.shape)
    dbo = np.zeros(self.bo.shape)
    dWc = np.zeros(self.Wc.shape)
    dbc = np.zeros(self.bc.shape)

    dh_prev = np.zeros((n_h,1))
    dc_prev = np.zeros((n_h, 1))
    dx = np.zeros((n_x, 1))

    for t in reversed(range(n)):
      (h, c, h_prev, c_prev, ft, it, cct, ot, x) = self.caches[t]
      d_h_d_ot = np.tanh(c) * ot * (1-ot)
      d_L_d_h = d_y_d_h.T @ d_y
      dot = d_L_d_h * d_h_d_ot
      dcct = d_c * it * (1-np.square(cct)) + d_L_d_h * ot * (1-np.square(np.tanh(c))) * it * (1-cct**2)
      dft = ft * (1-ft) * (d_c * c_prev + d_L_d_h * ot * (1-np.square(np.tanh(c))) * c_prev)
      dit = it * (1-it) * (d_c * cct + d_L_d_h * ot * (1-np.square(np.tanh(c))) * cct)

      concate = np.concatenate((h_prev,x), axis=0).T
      dWf += dft @ concate
      dWi += dit @ concate
      dWc += dcct @ concate
      dWo += dot @ concate
      dbf += np.sum(dft, axis=1, keepdims=True)
      dbi += np.sum(dit, axis=1, keepdims=True)
      dbc += np.sum(dcct, axis=1, keepdims=True)
      dbo += np.sum(dot, axis=1, keepdims=True)

      dh_prev += self.Wf[:,: n_h].T @ dft + self.Wi[:,: n_h].T @ dit+ self.Wc[:,: n_h].T @ dcct + self.Wo[:,: n_h].T @ dot
      dc_prev += d_c * ft + d_L_d_h * ot * (1-np.square(np.tanh(c))) * ft
      dx += self.Wf[:,n_h :].T @ dft + self.Wi[:,n_h :].T @ dit+ self.Wc[:,n_h :].T @ dcct + self.Wo[:,n_h :].T @ dot

      # Clip to prevent exploding gradients.
    for d in [dWf, dWi, dWc, dWo, dbf, dbi, dbc, dbo]:
      np.clip(d, -1, 1, out=d)

    self.Wf -= learn_rate * dWf
    self.bf -= learn_rate * dbf
    self.Wi -= learn_rate * dWi
    self.bi -= learn_rate * dbi
    self.Wc -= learn_rate * dWc
    self.bc -= learn_rate * dbc
    self.Wo -= learn_rate * dWo
    self.bo -= learn_rate * dbo

    self.last_hs[0] -= learn_rate * dh_prev
    self.last_cs[0] -= learn_rate * dc_prev
    self.last_inputs -= learn_rate * dx

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