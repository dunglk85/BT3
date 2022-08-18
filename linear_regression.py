import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg

class Regressor():
    def __init__(self, w_init, learning_rate=0.6, tol=1e-4, max_iters=1000000, check_after=10):
        self.W = w_init
        self.lr = learning_rate
        self.tol = tol
        self.check_af = check_after
        self.n = None
        self.m = None
        self.costs = []
        self.iterations = []
        self.max_iters = max_iters
        self.inner_count = None
        self.grad_norm = None
        self.H = None
        self.ATb = None
        self.btb = None

    def cost(self, w):
        return np.dot(np.dot(w.T, self.H), w)/2 - np.dot(w.T, self.ATb) + self.btb/2
    def grad(self, w):
        return np.dot(self.H, w) - self.ATb

    def get_data_info(self, X, y):
        self.m, self.n = X.shape
        self.H = np.dot(X.T, X)/self.m
        self.ATb = np.dot(X.T, y)/self.m
        self.btb = np.dot(y.T,y)/self.m

    def check(self, g):
        self.grad_norm = linalg.norm(g)
        if self.grad_norm < self.tol:
            return True
        return False


    def fit(self, X, y):
        self.back_tracking_newton(X,y)
    # test the model on test data
    def fix_step_gradient(self, X, y):
        self.get_data_info(X, y)
        for i in range(self.max_iters):
            dW = self.grad(self.W)
            if i % self.check_af == 0:
                if self.check(dW):
                    break
            self.W = self.W - self.lr*dW
            cost = self.cost(self.W)
            self.costs.append(cost)
            self.iterations.append(i)

    def back_tracking_gradient(self,X,y):
        self.inner_count = 0
        self.get_data_info(X, y)
        cost = self.cost(self.W)
        self.costs.append(cost)
        self.iterations.append(0)
        for i in range(self.max_iters):
            dW = self.grad(self.W)
            if self.check(dW):
                break
            t = 1
            cost = self.cost(self.W)
            while cost > self.costs[-1] - 0.5*t * self.grad_norm*self.grad_norm and t > 1e-6:
                self.inner_count += 1
                t = 0.5*t
                self.W = self.W - t * dW
                cost = self.cost(self.W)

            self.costs.append(cost)
            self.iterations.append(i + 1)

    def fix_step_accelerated(self, X, y):
        self.get_data_info(X, y)
        cost = self.cost(self.W)
        self.costs.append(cost)
        self.iterations.append(0)
        pre_W = self.W
        for i in range(self.max_iters):
            v = self.W + i * (self.W - pre_W)/(i + 3)
            dV = self.grad(v)
            if i % self.check_af == 0:
                if self.check(dV):
                    break
            pre_W = self.W
            self.W = v - self.lr * dV
            cost = self.cost(self.W)
            self.costs.append(cost)
            self.iterations.append(i + 1)

    def back_tracking_accelerated(self, X, y):
        self.inner_count = 0
        self.get_data_info(X, y)
        cost = self.cost(self.W)
        self.costs.append(cost)
        self.iterations.append(0)
        pre_W = self.W
        for i in range(self.max_iters):
            v = self.W + i * (self.W - pre_W)/ (i + 3)
            dV = self.grad(v)
            if self.check(dV):
                break
            t = 0.65
            pre_W = self.W
            cost = self.cost(self.W)
            while cost > self.costs[-1] - 0.5 * t * self.grad_norm * self.grad_norm and t > 1e-6:
                self.inner_count += 1
                t = 0.5 * t
                self.W = self.W - t * dV
                cost = self.cost(self.W)

            self.costs.append(cost)
            self.iterations.append(i)

    def fix_step_newton(self, X, y):
        self.get_data_info(X, y)
        inv_h = linalg.inv(self.H)
        for i in range(self.max_iters):
            dW = self.grad(self.W)
            if i % self.check_af == 0:
                if self.check(dW):
                    break
            self.W = self.W - np.dot(inv_h, dW)
            cost = self.cost(self.W)
            self.costs.append(cost)
            self.iterations.append(i)


    def back_tracking_newton(self, X, y):
        self.get_data_info(X, y)
        inv_h = linalg.inv(self.H)
        self.inner_count = 0
        cost = self.cost(self.W)
        self.costs.append(cost)
        self.iterations.append(0)
        for i in range(self.max_iters):
            dW = self.grad(self.W)
            if self.check(dW):
                break
            t = 1
            cost = self.cost(self.W)
            while cost > self.costs[-1] - 0.5 * t * self.grad_norm * self.grad_norm and t > 1e-6:
                self.inner_count += 1
                t = 0.5 * t
                self.W = self.W - t*np.dot(inv_h, dW)
                cost = self.cost(self.W)
            self.costs.append(cost)
            self.iterations.append(i+1)


    def predict(self,X):
        return np.dot(X,self.W)
    # plot the iterations vs cost curves
    def plot(self,figsize=(7,5)):
        plt.figure(figsize=figsize)
        plt.plot(self.iterations, self.costs)
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.title("Iterations vs Cost")
        plt.show()
    # calculates the accuracy
    def score(self,X,y):
        return 1-(np.sum(((y-self.predict(X))**2))/np.sum((y-np.mean(y))**2))