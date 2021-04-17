import read_MNIST
import matplotlib.pyplot as plt
import numpy as np


def FindAccuracy():
    counter =0
    for i in range (100):
        weights = np.random.normal(0, 1, size=(16, 784))
    
        bias = np.random.normal(0,0,size=(16, 1))
   
        res = np.dot(weights, (read_MNIST.train_set[i][0]))+bias

        # print(read_MNIST.train_set[6][0])
        # print(res)
    
        weights2 = np.random.normal(0, 1, size=(16, 16))
        bias2 = np.random.normal(0,0,size=(16, 1))
        res2 = np.dot(weights2, (res))+bias2

        weights3 = np.random.normal(0, 1, size=(10, 16))
        bias3 = np.random.normal(0,0,size=(10, 1))
        res3 = np.dot(weights3, (res2))+bias3

        # print(res3)
        # print(read_MNIST.train_set[i][1])
        # maxElement = np.amax(res3)
        # 
        result = np.argmax(res3, axis=0)
        
        # AnsmaxElement =  np.amax(read_MNIST.train_set[i][1])
        AnsResult = np.argmax(read_MNIST.train_set[i][1], axis=0)# np.where(read_MNIST.train_set[i][1] == AnsmaxElement)
        # print(result)
        # print(AnsResult)
        if result == AnsResult:
            counter+=1
        
    print(counter/100)



    
def main():
    FindAccuracy()

if __name__ == "__main__":
    main()