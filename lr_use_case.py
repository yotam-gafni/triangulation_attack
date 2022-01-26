import random
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

random.seed(1)


d = 1
all_outputs = []

X_a = []
Y_a = []
X_b = []
Y_b = []

for i in range(10):
	var = random.random() * 2
	diff = random.random() * 0.5 
	ran = random.random() 
	X_a.append([100 * var])
	Y_a.append(var * (0.6 + ran) + diff)
	var = random.random() * 2
	diff = random.random() * 0.25
	ran = random.random() * 0.6
	X_b.append([100 * var])
	Y_b.append(var * (0.9 + ran) + diff)

X = np.array(X_a)
Y = np.array(Y_a)
X2 = np.array(X_b)
Y2 = np.array(Y_b)
X3 = np.concatenate([X,X2])
Y3 = np.concatenate([Y,Y2])

reg = []
reg.append(LinearRegression().fit(X,Y))
reg.append(LinearRegression().fit(X2,Y2))
reg.append(LinearRegression().fit(X3,Y3))

mn=0
mx=200
x1=np.linspace(mn,mx,500)

fig, ax = plt.subplots()
plt.xlabel("Square meters")
plt.ylabel("House price (in Million NIS)")
ax.scatter(X,Y,color='orange', marker = 'o', label="Yad2")
ax.scatter(X2,Y2,color='green', marker = 'x', label="Madlan")
y1=reg[0].coef_[0]*x1+reg[0].intercept_
ax.plot(x1,y1,'-',color='orange')
y1=reg[1].coef_[0]*x1+reg[1].intercept_
ax.plot(x1,y1,'-g')
y1=reg[2].coef_[0]*x1+reg[2].intercept_
ax.plot(x1,y1,'-b')
ax.legend(loc="upper left", fancybox=True, shadow=True, ncol=5)
plt.show()	
