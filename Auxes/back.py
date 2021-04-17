import read_MNIST
import numpy as np
import random

def shuffle(self):
      if not len(self.temp):
         return []
      i = random.choice(self.indices)
      j = random.choice(self.indices)
      self.temp[i], self.temp[j] = self.temp[j], self.temp[i]
      return self.temp

def FindAccuracy():
    learning_rate = 1
    number_of_epochs = 20
    batch_size = 10
    counter =0
    
    for i in range (100):
        testset = read_MNIST.train_set[i][0]

        weights = np.random.normal(0, 1, size=(16, 784))
        bias = np.random.normal(0,0,size=(16, 1))
    
        weights2 = np.random.normal(0, 1, size=(16, 16))
        bias2 = np.random.normal(0,0,size=(16, 1))

        weights3 = np.random.normal(0, 1, size=(10, 16))
        bias3 = np.random.normal(0,0,size=(10, 1))

        for i in range (number_of_epochs):
            shuffle(testset)
            for batch in testset:
                grad_W1 = np.random.normal(0,0,size=(16, 784))
                grad_W2 = np.random.normal(0,0,size=(16, 16))
                grad_W3 = np.random.normal(0,0,size=(10, 16))
                grad_b1 = np.random.normal(0,0,size=(16, 1))
                grad_b2 = np.random.normal(0,0,size=(16, 1))
                grad_b3 = np.random.normal(0,0,size=(10, 1))
            
                
           