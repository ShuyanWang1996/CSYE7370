# Thompson Sampling

import numpy as np
import matplotlib.pyplot as plt
import pymc

# Constants in simulation: the number of bandit, the number of experiment times
arm_count = 10
exp_count = 500 * arm_count
# Initialize the probability of each machine
arm_probs = np.random.uniform( low = 0, high = 1, size = arm_count )
# para_alphas = np.zeros(arm_count)
# para_betas = np.zeros(arm_count)
# betas = np.zeros(arm_count)
opt_index = 0
trials = np.zeros(arm_count)
wins = np.zeros(arm_count)
averages = np.zeros(arm_count)

# Initialize the trails and wins
def initialPara():
	for i in range(0,arm_count-1):
		trials[i] = 100
		wins[i] = int(arm_probs[i]*100)

# Citation starts
# Source: https://zhuanlan.zhihu.com/p/36199435
def thompsonExp():
	for i in range(1,exp_count):
		b = pymc.rbeta(1 + wins, 1 + trials - wins)
		item = np.argmax(b)
		trials[item] += 1
		# Rewrite the judement code
		result =  np.random.binomial(n=1, p=arm_probs[item])
		if result == 1:
			wins[item] += 1
# Citation ends

# Draw the figure
def draw():
	plt.figure()
	plt.title("Thompson Sampling")
	x_axis = np.linspace(0,arm_count-1,10)
	y_axis = np.linspace(0,1,10)
	plt.xticks(x_axis)
	plt.yticks(y_axis)
	plt.plot(x_axis,arm_probs,linestyle="--")
	plt.xlabel("Arm Index")
	plt.ylabel("Random Generated Probability")
	opt_x = opt_index
	opt_y = arm_probs[opt_x]
	plt.scatter([opt_x],[opt_y],s=30,color="red")
	plt.annotate("Selected Arm",xy=(opt_x, opt_y),fontsize=8,xycoords='data') 
	plt.show()

# Program Entrance
if __name__ == '__main__':
	initialPara()
	thompsonExp()
	for i in range(0,arm_count-1):
		averages[i] = wins[i] / trials[i]
	opt_index = np.argmax(averages)
	print("Arm - %d is selected to be the recommended one" %opt_index)
	draw()