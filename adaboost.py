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

def fixTarget(t):
	temp = []
	for i in range(len(t)):
		if t[i] % 2 == 0:
			temp.append(1)
		else:
			temp.append(-1)
	return temp

# M A I N
def main():

	# Load Digits 
	digits, target = datasets.load_digits(return_X_y=True)
	T = 50
	target = fixTarget(target) # change 0,1,...,9 values to +/-1 values


	TR = []
	# Decision Tree
	dt = Adaboost(digits, target, T, DecisionTreeClassifier(max_depth=5))
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
		dt.TR.append(dt.getError(t))    
 
	plt.axis([0, dt.T, 0, dt.m])	
	plt.show()
 
	# Random Forest
	ada_2 = Adaboost(digits, target, T, RandomForestClassifier\
			(max_depth=5, n_estimators=10, max_features=1))

if __name__ == "__main__":
	main()




