import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg

class Regressor():
    def __init__(self, w_init, learning_rate=0.1, tol=1e-6, max_iters=100000, check_stop = True):
        self.W = w_init
        self.lr = learning_rate
        self.tol = tol
        self.check_af = 10
        self.n = None
        self.m = None
        self.costs = []
        self.iterations = []
        self.max_iters = max_iters
        self.inner_count = None
        self.square_norm = None
        self.H = None
        self.ATb = None
        self.btb = None
        self.check_stop = check_stop

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
        norm = linalg.norm(g)
        self.square_norm = norm*norm
        if self.check_stop:
            if norm < self.tol:
                return True
        return False


    def fit(self, X, y, solver='gd'):
        if solver == 'bgd':
            return self.back_tracking_gradient(X,y)
        elif solver == 'acc':
            return self.fix_step_accelerated(X, y)
        elif solver == 'bac':
            return self.back_tracking_accelerated(X,y)
        elif solver == 'nt':
            return self.fix_step_newton(X,y)
        elif solver == 'bnt':
            return self.back_tracking_newton(X,y)
        else:
            return self.fix_step_gradient(X,y)
    # test the model on test data
    def fix_step_gradient(self, X, y):
        self.get_data_info(X, y)
        costs = []
        for i in range(self.max_iters+1):
            cost = self.cost(self.W)
            costs.append(cost)
            dW = self.grad(self.W)
            if i % self.check_af == 0:
                if self.check(dW):
                    break
            self.W = self.W - self.lr*dW.T
        return costs
    def back_tracking_gradient(self,X,y):
        costs = []
        self.inner_count = 0
        self.get_data_info(X, y)
        cost = self.cost(self.W)
        costs.append(cost)
        for i in range(self.max_iters):
            dW = self.grad(self.W)
            if self.check(dW):
                break
            self.W = self.W - self.lr * dW.T
            next_cost = self.cost(self.W)
            t = 1
            while next_cost > costs[-1] - 0.5*t * self.square_norm and t > 1e-8:
                self.inner_count += 1
                t = 0.5*t
                self.W = self.W - t * dW.T
                next_cost = self.cost(self.W)
            costs.append(next_cost)
        return costs
    def fix_step_accelerated(self, X, y):
        costs = []
        self.get_data_info(X, y)
        cost = self.cost(self.W)
        costs.append(cost)
        pre_W = self.W
        for i in range(self.max_iters):
            v = self.W + i * (self.W - pre_W)/(i + 3)
            dV = self.grad(v)
            if i % self.check_af == 0:
                if self.check(dV):
                    break
            pre_W = self.W
            self.W = v - self.lr * dV.T
            cost = self.cost(self.W)
            costs.append(cost)
        return costs
    def back_tracking_accelerated(self, X, y):
        costs = []
        self.inner_count = 0
        self.get_data_info(X, y)
        cost = self.cost(self.W)
        costs.append(cost)
        pre_W = self.W
        t = 1
        for i in range(self.max_iters):
            v = self.W + i * (self.W - pre_W)/ (i + 3)
            dV = self.grad(v)
            if self.check(dV):
                break
            pre_W = self.W
            self.W = self.W - self.lr * dV.T
            next_cost = self.cost(self.W)
            while next_cost > costs[-1] - 0.5 * t * self.square_norm and t > 1e-8:
                self.inner_count += 1
                t = 0.8 * t
                self.W = self.W - t * dV.T
                next_cost = self.cost(self.W)
            costs.append(next_cost)
        return costs
    def fix_step_newton(self, X, y):
        costs = []
        self.get_data_info(X, y)
        inv_h = linalg.inv(self.H)
        for i in range(self.max_iters+1):
            cost = self.cost(self.W)
            costs.append(cost)
            dW = self.grad(self.W)
            if self.check(dW):
                break
            self.W = self.W - np.dot(inv_h, dW)
        return costs

    def back_tracking_newton(self, X, y):
        costs = []
        self.get_data_info(X, y)
        inv_h = linalg.inv(self.H)
        self.inner_count = 0
        cost = self.cost(self.W)
        costs.append(cost)
        for i in range(self.max_iters):
            dW = self.grad(self.W)
            if self.check(dW):
                break
            self.W = self.W - np.dot(inv_h, dW)
            next_cost = self.cost(self.W)
            t = 1
            while next_cost > costs[-1] - 0.5 * t * self.square_norm and t > 1e-8:
                self.inner_count += 1
                t = 0.5 * t
                self.W = self.W - t*np.dot(inv_h, dW)
                next_cost = self.cost(self.W)
            costs.append(next_cost)
        return costs

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