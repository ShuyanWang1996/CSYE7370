import numpy as np
import matplotlib.pyplot as plt

# Citation starts
# Source: https://www.freesion.com/article/5297307805/
class EpsilonGreedy:
    def __init__(self):
        self.epsilon = 0.1
        self.num_arm = 10 
        self.arms = np.random.uniform(0, 1, self.num_arm)
        self.best = np.argmax(self.arms)
        self.T = 50000  
        self.hit = np.zeros(self.T)  
        self.reward = np.zeros(self.num_arm)  
        self.num = np.zeros(self.num_arm) 

    def get_reward(self, i):
        return self.arms[i] + np.random.normal(0, 1)

    def update(self, i):
        self.num[i] += 1
        self.reward[i] = (self.reward[i]*(self.num[i]-1)+self.get_reward(i))/self.num[i]

    def calculate(self):
        for i in range(self.T):
            if np.random.random() > self.epsilon:
                index = np.argmax(self.reward)
            else:
                a = np.argmax(self.reward)
                index = a
                while index == a:
                    index = np.random.randint(0, self.num_arm)
            if index == self.best:
                self.hit[i] = 1
            self.update(index)

    def plot(self):
        # Update starts
        plt.figure()
        plt.title("E-Greedy")
        x = np.array(range(self.T))
        y1 = np.zeros(self.T)
        t = 0
        for i in range(self.T):
            t += self.hit[i]
            y1[i] = t/(i+1)
        y2 = np.ones(self.T)*(1-self.epsilon)
        plt.xlabel("Times of Experiment")
        plt.plot(x, y1, label="One")
        plt.plot(x, y2, label="Frequency of Finding the Best")
        plt.legend(loc="upper left")
        plt.show()
        # Update ends


E = EpsilonGreedy()
E.calculate()
E.plot()

# Citation ends