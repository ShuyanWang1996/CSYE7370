# Upper Confidence Bound

import numpy as np
import matplotlib.pyplot as plt

# Constants in simulation: the number of bandit, the number of experiment times
arm_count = 10
exp_count = 500 * arm_count
# Initialize the probability of each machine
arm_probs = np.random.uniform( low = 0, high = 1, size = arm_count )
# Citation starts
# Source: https://zhuanlan.zhihu.com/p/32356077
recorded_rewards = np.zeros(arm_count)
chosen_count =  np.zeros(arm_count)
total_reward = 0
# Citation ends
opt_index = 0

# Calculate Delta after each experiment
# Citation starts
# Source: https://zhuanlan.zhihu.com/p/32356077
def calculate_delta(T, item):
    if chosen_count[item] == 0:
        return 1
    else:
        return np.sqrt(2 * np.log(T) / chosen_count[item])
# Citation ends

# For Nth experiment, pick one machine by Upper Confidence Bound algorithm
# Citation starts
# Source: https://zhuanlan.zhihu.com/p/32356077
def UCB(t, N):
    # upper_bound_probs = recorded_rewards + delta
    upper_bound_probs = [recorded_rewards[item] + calculate_delta(t, item) for item in range(N)]
    # Pick the one with the highest UBC value
    item = np.argmax(upper_bound_probs)
    # Generate the result by np.random.binomial()
    reward = np.random.binomial(n=1, p=arm_probs[item])
    return item, reward
# Citation ends

# Draw the figure
def draw():
	plt.figure()
	plt.title("Upper Confidence Bound")
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
	for i in range(1,exp_count):
		# Citation starts
		# Source: https://zhuanlan.zhihu.com/p/32356077
		item, reward = UCB(i, arm_count)
		total_reward += reward
		recorded_rewards[item] = ((i - 1) * recorded_rewards[item] + reward) / i
		chosen_count[item] += 1
		# Citation ends
	total_reward = total_reward / exp_count
	opt_index = np.argmax(chosen_count)
	print("Arm - %d is selected to be the recommended one" %opt_index)
	draw()

