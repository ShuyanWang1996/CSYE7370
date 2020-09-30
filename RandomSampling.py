# Random Sampling

import numpy as np
import matplotlib.pyplot as plt

# Constants in simulation: the number of bandit, the number of experiment times
arm_count = 10
exp_count = 500
# Initialize the probability of each machine
arm_probs = np.random.uniform( low = 0, high = 1, size = arm_count )
rewards = np.zeros(arm_count)
opt_index = 0

# Each arm tests 1000 times to calculate the average reward
def randomExp():
	for i  in range(0,arm_count-1):
		rewards[i] = np.random.binomial(n=exp_count,p=arm_probs[i]) / exp_count
		print("Arm - %d gets the reward %f with the probability %f" %(i,rewards[i],arm_probs[i]))

# Draw the figure
def draw():
	plt.figure()
	plt.title("Random Sampling")
	x_axis = np.linspace(0,arm_count-1,10)
	plt.xticks(x_axis)
	plt.plot(x_axis,arm_probs,linestyle="--",label="probability")
	plt.plot(x_axis,rewards,linestyle="-",label="reward")
	plt.legend(loc="upper left")
	plt.xlabel("Arm Index")
	# Annotate the selected arm
	opt_x = opt_index
	opt_y = rewards[opt_x]
	plt.scatter([opt_x],[opt_y],s=30,color="red")
	plt.annotate("Selected Arm",xy=(opt_x, opt_y),fontsize=8,xycoords='data') 
	plt.show()


# Program Entrance
if __name__ == '__main__':
	print("Testing each arm for %d times" %exp_count)
	randomExp()
	# Pick the one with the highest reward
	opt_index = np.argmax(rewards)
	print("Arm - %d is selected to be the recommended one" %opt_index)
	draw()