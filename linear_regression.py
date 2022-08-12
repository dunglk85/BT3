import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg

class Regressor():
    # init methodd initializes all parameters needed to implement regression
    def __init__(self, learning_rate=1, tol=0.001, checking_after=10, max_iters=1000000):
        self.W = None
        self.__lr = learning_rate
        self.__tol = tol
        self.__check_af = checking_after
        self.__n = None
        self.__m = None
        self.__costs = []
        self.__iterations = []
        self.__max_iters = max_iters
        self.__inner_count = None
        np.random.seed(np.random.randint(100))
    # random initialization of weights and bias
    def __init_w(self):
        self.W = np.random.randn(self.__n)


    # fit the model to the dataset: training process
    def fit(self, X, y):
        self.back_tracking_gradient(X,y)
    # test the model on test data
    def fix_step_gradient(self, X, y):
        self.__m, self.__n = X.shape
        self.__init_w()
        loss = np.dot(X, self.W) - y
        cost = np.sum(np.square(loss)) / (2 * self.__m)
        dW = np.dot(X.T, loss) / self.__m
        self.__costs.append(cost)
        self.__iterations.append(0)
        for i in range(self.__max_iters):
            self.W = self.W - self.__lr * dW
            loss = np.dot(X, self.W) - y
            cost = np.sum(np.square(loss)) / (2 * self.__m)
            dW = np.dot(X.T, loss) / self.__m
            if self.__costs[-1] - cost < self.__tol*cost:
                break
            self.__costs.append(cost)
            self.__iterations.append(i+1)

    def back_tracking_gradient(self,X,y):
        self.__inner_count = 0
        self.__m, self.__n = X.shape
        self.__init_w()
        loss = np.dot(X, self.W) - y
        cost = np.sum(np.square(loss)) / (2 * self.__m)
        dW = np.dot(X.T, loss) / self.__m
        self.__costs.append(cost)
        self.__iterations.append(0)
        for i in range(self.__max_iters):
            t = self.__lr
            self.W = self.W - t * dW
            loss = np.dot(X, self.W) - y
            cost = np.sum(np.square(loss)) / (2 * self.__m)
            dW = np.dot(X.T, loss) / self.__m
            square_norm_dW = np.dot(dW.T,dW)
            while cost > self.__costs[-1] - .01*t * square_norm_dW:
                self.__inner_count += 1
                print(self.__inner_count)
                t = 0.5*t
                self.W = self.W - t * dW
                loss = np.dot(X, self.W) - y
                cost = np.sum(np.square(loss)) / (2 * self.__m)

            if self.__costs[-1] - cost < self.__tol*cost:
                print("inner", self.__inner_count)
                break
            self.__costs.append(cost)
            self.__iterations.append(i + 1)

    def accelerated(self, X, y):
        pass
    def predict(self,X):
        return np.dot(X,self.W)
    # plot the iterations vs cost curves
    def plot(self,figsize=(7,5)):
        plt.figure(figsize=figsize)
        plt.plot(self.__iterations,self.__costs)
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.title("Iterations vs Cost")
        plt.show()
    # calculates the accuracy
    def score(self,X,y):
        return 1-(np.sum(((y-self.predict(X))**2))/np.sum((y-np.mean(y))**2))