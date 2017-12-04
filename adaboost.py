'''
Kristina Kolibab
Homework 4
AdaBoost 
'''

import numpy as np
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import matplotlib.pyplot as plt

class Adaboost:

	def __init__(self, digits, target, epoch, classifier):
		self.x = digits
		self.y = target
		self.T = epoch
		self.clf = classifier
		self.m = len(self.y)
		self.alpha = 0
		self.et = [0] * self.T 
		self.ht = [0] * self.T
		self.wt = [0] * self.T
		self.D = [1/self.m] * self.m   
		self.ss = int(self.m*0.5)
		self.indices = [0] * self.ss
		self.TR = [0] * self.T

	def randomSample(self):
		return np.random.choice(self.m, size=self.ss, replace=False, p=self.D)     

	def getError(self, t):
		product = 1
		for i in range(t):
			product *= 2 * np.sqrt(self.et[i] * (1 - self.et[i]))
		return product

	def fixTarget(self):
		temp = []
		for i in range(len(self.y)):
			if self.y[i] % 2 == 0:
				temp.append(1)
			else:
				temp.append(-1)
		return temp

	def plot(self, z, t, i):
		plt.figure(i)
		plt.plot(range(self.T), self.TR, z)
		plt.xlabel("Rounds")
		plt.ylabel("Training Error")
		plt.title(t)

	def plot_both(self, T1, TR1, t):
		plt.figure(3)
		plt.plot(range(T1), TR1, 'bo', range(self.T), self.TR, 'g^')
		plt.xlabel("Rounds")
		plt.ylabel("Training Error")
		plt.title(t)

# M A I N
def main():

	# Load Digits 
	digits, target = datasets.load_digits(return_X_y=True)
	T = 50

	# I could have also made this a "train" function in my class 

	# Decision Tree
	dt = Adaboost(digits, target, T, DecisionTreeClassifier(max_depth=5))
	dt.y = dt.fixTarget() # change targets to +/-1 values
	for t in range(dt.T):
		dt.indices = dt.randomSample()
		dt.indices = np.sort(dt.indices)

		dt.clf.fit(
				[dt.x[i] for i in dt.indices], 
				[dt.y[i] for i in dt.indices],
				sample_weight = [dt.D[i] for i in dt.indices])

		dt.ht[t] = dt.clf.predict(dt.x)
		dt.et[t] = sum(dt.D * (dt.y != dt.ht[t]))
		dt.wt[t] = 0.5*np.log((1 - dt.et[t])/dt.et[t])

		temp = sum([dt.D[j] * np.exp(-dt.wt[t] * dt.y[j] * dt.ht[t][j]) for j in range(dt.m)])
		dt.D = [(dt.D[i] * np.exp(-dt.wt[t] * dt.y[i] * dt.ht[t][i]))/temp for i in range(dt.m)]
    
		# training error
		dt.TR[t] = (dt.getError(t))    
	dt.plot('bo', "Decision Tree Weak Learner", 1)

	# Random Forest
	rf = Adaboost(digits, target, T, RandomForestClassifier\
			(max_depth=5, n_estimators=10, max_features=1))
	rf.y = rf.fixTarget() # change targets to +/-1 values
	for t in range(rf.T):
		rf.indices = rf.randomSample()
		rf.indices = np.sort(rf.indices)

		rf.clf.fit(
				[rf.x[i] for i in rf.indices], 
				[rf.y[i] for i in rf.indices],
				sample_weight = [rf.D[i] for i in rf.indices])

		rf.ht[t] = rf.clf.predict(rf.x)
		rf.et[t] = sum(rf.D * (rf.y != rf.ht[t]))
		rf.wt[t] = 0.5*np.log((1 - rf.et[t])/rf.et[t])

		temp = sum([rf.D[j] * np.exp(-rf.wt[t] * rf.y[j] * rf.ht[t][j]) for j in range(rf.m)])
		rf.D = [(rf.D[i] * np.exp(-rf.wt[t] * rf.y[i] * rf.ht[t][i]))/temp for i in range(rf.m)]

		# training error
		rf.TR[t] = (rf.getError(t))
	rf.plot('g^', "Random Forest Weak Learner", 2)

	# plot both
	rf.plot_both(dt.T, dt.TR, "Decision Tree vs. Random Forest")
	plt.show()
	
if __name__ == "__main__":
	main()




