import random
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

random.seed(1)


d = 1
all_outputs = []

for tries in range(2):
	X_a = []
	Y_a = []

	for i in range(10):
		X_a.append([random.random() * 2])
		Y_a.append(random.random() * 2)

	if tries == 0:
		X = np.array(X_a)
		Y = np.array(Y_a)
	else:
		X_ = np.array(X_a)
		Y_ = np.array(Y_a)
		X_3 = np.array(X_a * 3)
		Y_3 = np.array(Y_a * 3)

X4 = np.concatenate([X,X_3])
Y4 = np.concatenate([Y,Y_3])

X_4 = np.concatenate([X_,X_3])
Y_4 = np.concatenate([Y_,Y_3])

reg = []
reg.append(LinearRegression().fit(X,Y))
reg.append(LinearRegression().fit(X_,Y_))
reg.append(LinearRegression().fit(X4,Y4))
reg.append(LinearRegression().fit(X_4,Y_4))

mn=0
mx=2
x1=np.linspace(mn,mx,500)

for i in range(4):
	y1=reg[i].coef_[0]*x1+reg[i].intercept_
	fig, ax = plt.subplots()
	if i == 0:
		ax.scatter(X,Y,color='red', marker = 'o', label="S (under truth_j)")
		ax.plot(x1,y1,'-r')
	elif i == 1:
		ax.scatter(X_,Y_,color='blue', marker = 'x', label="S' (under s_j)")
		ax.plot(x1,y1,'-b')
	elif i == 2:
		ax.scatter(X,Y,color='red', marker = 'o', label="S (under truth_j)")
		ax.scatter(X_,Y_,color='blue', marker = 'x', label="3 x S' (under truth_j)", linewidth = 3)
		ax.plot(x1,y1,'-r')
	elif i == 3:
		ax.scatter(X_,Y_,color='blue', marker = 'x', label="4 x S' (under s_j)", linewidth = 4)
		ax.plot(x1,y1,'-b')
		
	ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)
	plt.ylim(0,2)
	plt.show()	

