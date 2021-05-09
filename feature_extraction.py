import numpy as np
import json
import re
from contractions import contractions
import codecs
import itertools


# Create the vocabulary.
def createVocabulary(train_data):
  '''
  Create dictionary with unique words and index each word
  :param train_data: python dictionary
  :return:
  '''
  vocab = list(set([w for text in train_data.keys() for w in text.split(' ')]))
  vocab_size = len(vocab)
  print('%d unique words found' % vocab_size)
  # print(vocab)

  # Assign indices to each word.
  word_to_idx = {w: i for i, w in enumerate(vocab)}
  # idx_to_word = {i: w for i, w in enumerate(vocab)}
  return vocab_size, word_to_idx


def createInputList(text, vocab_size, word_to_idx):
  """
  Returns list of 2d arrays of one-hot vectors representing the words in the input text string.
  - text is a string
  - Each one-hot vector has shape (vocab_size, 1)
  """
  inputs = []
  for w in text.split(' '):
    v = np.zeros((vocab_size, 1))
    v[word_to_idx[w]] = 1
    inputs.append(v)
  return inputs


def createInput3D(text, vocab_size, word_to_idx):
  '''
  Returns 3d arrays of one-hot vectors representing the words in the input text string and order.
  - text is a string
  - Each one-hot vector has shape (vocab_size, 1)
  output: (vocab_size, 1, vocab_order)
  '''
  wordCount = len(text.split(' '))
  inputs = np.zeros((vocab_size, 1, wordCount))
  order = 0
  for w in text.split(' '):
    v = np.zeros((vocab_size, 1))
    v[word_to_idx[w]] = 1
    inputs[word_to_idx[w], 0, order] = 1
    order += 1
  return inputs


def generate_data():
  path = '/Users/nguyenphuc/Documents/Python/SocialMediaBullyDetect/social_media_cyberbullying_detection/RNN_source/social_media_cyberbullying_detection/datasets/MMHS/'
  fn = 'MMHS150K_extraction.json'

  f = open(path + fn, 'r')
  data = json.load(f)  # size is 32002
  f.close()
  train_data_no = 300
  val_data_no = 100
  test_data_no = 200
  count = 0
  train_dict = {}
  val_dict = {}
  test_dict = {}

  for k, v in data.items():
    if count < train_data_no:
      train_dict[k] = v
    elif count > train_data_no and count <= train_data_no + val_data_no:
      val_dict[k] = v
    elif count <= train_data_no + val_data_no + test_data_no:
      test_dict[k] = v
    count += 1

  print(count)
  print(len(train_dict.items()))
  print(len(val_dict.items()))
  print(len(test_dict.items()))

  with open(path + 'train_data.json', 'w') as fp:
    json.dump(train_dict, fp)
  fp.close()

  with open(path + 'val_data.json', 'w') as fp:
    json.dump(val_dict, fp)
  fp.close()

  with open(path + 'test_data.json', 'w') as fp:
    json.dump(test_dict, fp)
  fp.close()

# generate_data()

def sanitizer(original_text):
  # text = BeautifulSoup(original_text, "lxml").get_text()
  text = original_text
  # Remove Encodings
  text = re.sub(r'\\\\', r'\\', text)
  text = re.sub(r'\\x\w{2,2}', ' ', text)
  text = re.sub(r'\\u\w{4,4}', ' ', text)
  text = re.sub(r'\\n', '.', text)

  # Whitespace Formatting
  text = text.replace('"', ' ')
  text = text.replace('\\', ' ')
  text = text.replace('_', ' ')
  text = text.replace('-', ' ')
  text = re.sub(' +', ' ', text)

  # Remove Unicode characters
  text = codecs.decode(text, 'unicode-escape')
  text = ''.join([i if ord(i) < 128 else '' for i in text])

  # Remove email addresses
  text = re.sub(r'[\w\-][\w\-\.]+@[\w\-][\w\-\.]+[a-zA-Z]{1,4}', ' ', text)

  # Remove Twitter Usernames
  text = re.sub(r"(\A|\s)@(\w+)+[a-zA-Z0-9_\.]", ' ', text)

  # Remove urls
  text = re.sub(r'\w+:\/\/\S+', ' ', text)

  # Word Standardizing (Ex. Looooolll should be Looll)
  text = ''.join(''.join(s)[:2] for _, s in itertools.groupby(text))

  # Convert words to lower case
  text = text.lower().split()

  # Remove contractions by expansion of words
  text = [contractions[word] if word in contractions else word for word in text]

  # Rejoin words
  text = " ".join(text)

  # Remove non-alphabets
  text = re.sub("[^a-z\s]", " ", text)

  return " ".join(text.split())


def extract_train_data():
  """
  Extracting tweet text and labels from MMHS dataset json
  :return:
  """
  path = '/Users/nguyenphuc/Documents/Python/SocialMediaBullyDetect/social_media_cyberbullying_detection/RNN_source/social_media_cyberbullying_detection/datasets/MMHS/'
  fn = 'train_data.json'

  f = open(path + fn, 'r')
  data = json.load(f)  # size is 22002
  f.close()
  print(len(data.items()))
  train_data = {}
  for k, v in data.items():
    tweet_text = v['tweet_text']
    tweet_text = sanitizer(tweet_text)
    train_data[tweet_text] = v['labels']

  with open(path + 'train_data_text_labels.json', 'w') as fp:
    json.dump(train_data, fp)
  fp.close()
  print("Write file successfully")

# extract_train_data()


def grammarContractions(original_text):
  text = original_text.lower().split()
  reformed = [contractions[word] if word in contractions else word for word in text]

  return " ".join(reformed)

import csv
def extract_train_data_from_csv():
  path = '/Users/nguyenphuc/Documents/Python/SocialMediaBullyDetect/social_media_cyberbullying_detection/RNN_source/social_media_cyberbullying_detection/datasets/twitter/'
  csv_fn = 'clean_dataset.csv'
  train_data = 'train_data.json'
  count = 0
  dict_data = {}

  with open(path + csv_fn, 'r') as file:
    csv_file = csv.DictReader(file)
    for row in csv_file:
      dict_data[dict(row)['Comment']] = dict(row)['Insult']
      count += 1
      if (count == 50):
        break
  file.close()

  with open(path + 'train_data.json', 'w') as fp:
    json.dump(dict_data, fp)
  fp.close()