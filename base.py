# Importing library
import math
import random
import matplotlib.pyplot as plt
import numpy as np

FEATURES_NAMES = ['white_king_column', 'white_king_row', 'white_rook_column',
                  'white_rook_row', 'black_king_column', 'black_king_row']

ENCODED_VALUES = {
  'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8,
  '1': 1, '2': 2, '3': 3, '4': 4, '5': 5,  '6': 6, '7': 7, '8': 8
}

# convert line formatted in a dictionary
def decodeLine(line):
  values = line.split(',')
  data = {}
  data['white_king_column'] = values[0]
  data['white_king_row'] = values[1]
  data['white_rook_column'] = values[2]
  data['white_rook_row'] = values[3]
  data['black_king_column'] = values[4]
  data['black_king_row'] = values[5]
  data['result'] = values[6][:-1]

  return data

# read database
def readData():
  filename = r'Data/krkopt.data'
  f = open(filename, "r")

  dataset = []

  line = f.readline()
  while line != "":
    dataset.append(decodeLine(line))
    line = f.readline()
  
  return dataset

# split dataset into train and test
def splitDataset(dataset, ratio):
  random.shuffle(dataset)

  sz = len(dataset)
  cut = int(sz*ratio)

  train_data = dataset[:cut]
  test_data = dataset[cut:]

  return train_data, test_data

# encode values of rows/columns to [1..8]
def encodeValue(value):
  return ENCODED_VALUES[value]

# get meand and standart deviation of a set of data for a feature
def getMeanAndStdDeviation(dots):
  avg = sum(dots)/ float(len(dots))
  variance = sum([pow(avg - x, 2) for x in dots]) / float(len(dots) - 1)
  dev = math.sqrt(variance)

  return (avg, dev)

# trainning ML with dataset, getting some information
# in this case, we get mean and standart deviation for each result and features
# the returned dictionary will be in this format:
#   [result] = {
#      feature1: (mean, stdDev),
#      feature2: (mean, stdDev),
#      feature3: (mean, stdDev)
#   }
def getInformation(dataset):
  raw_info = {}

  for data in dataset:
    result = data['result']

    if result not in raw_info.keys():
      raw_info[result] = {}
      for name in FEATURES_NAMES:
        raw_info[result][name] = []

    for name in FEATURES_NAMES:
      raw_info[result][name].append(encodeValue(data[name]))
  
  info = {}
  for tag in raw_info.keys():
    info[tag] = {}
    for feature in FEATURES_NAMES:
      info[tag][feature] = getMeanAndStdDeviation(raw_info[tag][feature])
  
  return info

# calculate the probability of x be a part of this gaussian function
def calcGaussianProbability(avg, dev, x):
  if dev == 0.0:
    return 0.0

  expo = math.exp(-math.pow(x - avg, 2)/(2.0 * pow(dev, 2)))
  return expo / float(dev * math.sqrt(2.0 * math.pi))

# for an specific label/class, calc the prob this single data be group of this class
def calcClassProbability(summaries, data):
  prob = 1.0

  for name in FEATURES_NAMES:
    avg, dev = summaries[name]
    value = encodeValue(data[name])

    prob *= calcGaussianProbability(avg, dev, value)
  
  return prob

# predict label for a unique data
def getPredict(info, data):
  candidates = []

  for label, summaries in info.items():
    candidates.append((label, calcClassProbability(summaries, data)))
  
  bestLabel = None
  maxProb = None

  for label, prob in candidates:
    if maxProb == None or maxProb < prob:
      maxProb = prob
      bestLabel = label
  
  return bestLabel

# predict results for an specific dataset
def getPredictions(info, dataset):
  predictions = []
  
  for data in dataset:
    predict = {
      'data': data,
      'label': getPredict(info, data)
    }

    predictions.append(predict)

  hits = 0

  for predict in predictions:
    if predict['data']['result'] == predict['label']:
      hits += 1

  acc = hits / float(len(dataset))

  return predictions, acc

def viewData(dataset):
  histogram = {}

  coordinates = [1, 2, 3, 4, 5, 6, 7, 8]

  for name in FEATURES_NAMES:
    histogram[name] = [0, 0, 0, 0, 0, 0, 0, 0]

  for data in dataset:
    for key, value in data.items():
      if key == 'result':
        continue
      
      histogram[key][encodeValue(def view3D(dataset):
  fig = plt.figure()
  x = np.random.normal(1, 8, 30)
  y = np.random.normal(1, 8, 30)
 
  colour = []

  for data in dataset:
    

  plt.scatter(x, y, s = 50, c = colour, alpha = 0.8)
  plt.colorbar()
  plt.show()value) - 1] += 1
  
  for feature, l in histogram.items():
    title = "Histograma para o atributo: " + feature

    fig, ax = plt.subplots()

    ax.set_title(title)
    ax.set_ylabel('Frequência')
    ax.set_xlabel('Valor no tabuleiro')
    ax.bar(coordinates, l)
    plt.show()

def main():
  # add the data path in your system
  dataset = readData()

  RATIO = 0.7
  train_data, test_data = splitDataset(dataset, RATIO)

  print("Total number of examples are:", len(dataset))
  print("Size of train data:", len(train_data))
  print("Size of test data:", len(test_data))

  dataset_information = getInformation(train_data)
  #viewData(train_data)
  prediction, accuracy = getPredictions(dataset_information, test_data)
  view3D(test_data)

  print("================")
  percentage = round(float(accuracy * 100), 2)
  print("ran test data with", percentage, "of accuracy")

if __name__ == "__main__":
  main()
