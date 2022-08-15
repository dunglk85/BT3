import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg

class Regressor():
    # init methodd initializes all parameters needed to implement regression
    def __init__(self, learning_rate=0.01, tol=0.000001, max_iters=1000000):
        self.W = None
        self.__lr = learning_rate
        self.__tol = tol
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
        self.back_tracking_newton(X,y)
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
            if self.__costs[-1] - cost < self.__tol*cost:
                break
            self.__costs.append(cost)
            self.__iterations.append(i + 1)
            dW = np.dot(X.T, loss) / self.__m

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
            while cost > self.__costs[-1] - 0.5*t * square_norm_dW:
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

    def fix_step_accelerated(self, X, y):
        self.__m, self.__n = X.shape
        self.__init_w()
        loss = np.dot(X, self.W) - y
        cost = np.sum(np.square(loss)) / (2 * self.__m)
        self.__costs.append(cost)
        self.__iterations.append(0)
        pre_W = self.W
        for i in range(self.__max_iters):
            v = self.W + i * (self.W - pre_W)/ (i + 3)
            loss_v = np.dot(X, v) - y
            dV = np.dot(X.T, loss_v) / self.__m
            pre_W = self.W
            self.W = v - self.__lr * dV
            loss = np.dot(X, self.W) - y
            cost = np.sum(np.square(loss)) / (2 * self.__m)
            if self.__costs[-1] - cost < self.__tol * cost:
                break
            self.__costs.append(cost)
            self.__iterations.append(i + 1)

    def back_tracking_accelerated(self, X, y):
        self.__inner_count = 0
        self.__m, self.__n = X.shape
        self.__init_w()
        loss = np.dot(X, self.W) - y
        cost = np.sum(np.square(loss)) / (2 * self.__m)
        self.__costs.append(cost)
        self.__iterations.append(0)
        pre_W = self.W
        t = 1
        for i in range(self.__max_iters):
            v = self.W + i * (self.W - pre_W)/ (i + 3)
            loss_v = np.dot(X, v) - y
            cost_v = np.sum(np.square(loss_v)) / (2 * self.__m)
            dV = np.dot(X.T, loss_v) / self.__m
            pre_W = self.W
            self.W = v - t * dV
            loss = np.dot(X, self.W) - y
            cost = np.sum(np.square(loss)) / (2 * self.__m)
            square_norm = np.dot(dV.T, dV)
            while cost > cost_v - 0.5*t*square_norm:
                t = 0.9*t
                self.W = v - t * dV
                loss = np.dot(X, self.W) - y
                cost = np.sum(np.square(loss)) / (2 * self.__m)
                self.__inner_count += 1
            if self.__costs[-1] - cost < self.__tol * cost:
                print(self.__inner_count)
                break
            self.__costs.append(cost)
            self.__iterations.append(i + 1)

    def fix_step_newton(self, X, y):
        self.__m, self.__n = X.shape
        self.__init_w()
        loss = np.dot(X, self.W) - y
        cost = np.sum(np.square(loss)) / (2 * self.__m)
        dW = np.dot(X.T, loss) / self.__m
        hW = np.dot(X.T, X)/self.__m
        inv_h = linalg.inv(hW)
        self.__costs.append(cost)
        self.__iterations.append(0)
        for i in range(self.__max_iters):
            self.W = self.W - np.dot(inv_h,dW)
            loss = np.dot(X, self.W) - y
            cost = np.sum(np.square(loss)) / (2 * self.__m)
            if self.__costs[-1] - cost < self.__tol*cost:
                break
            self.__costs.append(cost)
            self.__iterations.append(i + 1)
            dW = np.dot(X.T, loss) / self.__m

    def back_tracking_newton(self, X, y):
        self.__m, self.__n = X.shape
        self.__init_w()
        loss = np.dot(X, self.W) - y
        cost = np.sum(np.square(loss)) / (2 * self.__m)
        dW = np.dot(X.T, loss) / self.__m
        hW = np.dot(X.T, X) / self.__m
        inv_h = linalg.inv(hW)
        self.__costs.append(cost)
        self.__iterations.append(0)
        for i in range(self.__max_iters):
            v = - np.dot(inv_h, dW)
            t = 1
            self.W = self.W + t*v
            loss = np.dot(X, self.W) - y
            cost = np.sum(np.square(loss)) / (2 * self.__m)
            squ = np.dot(dW.T, v)
            while cost > self.__costs[-1] + 0.5*t*squ:
                t = 0.4*t
                self.W = self.W + t * v
                loss = np.dot(X, self.W) - y
                cost = np.sum(np.square(loss)) / (2 * self.__m)
                print(cost)
            if self.__costs[-1] - cost < self.__tol * cost:
                break
            self.__costs.append(cost)
            self.__iterations.append(i + 1)
            dW = np.dot(X.T, loss) / self.__m
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