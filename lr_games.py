import random
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

random.seed(1)


d = 1
all_outputs = []

X_a = []
Y_a = []

for i in range(10):
	X_a.append([random.random() * 2])
	Y_a.append(random.random() * 2)

X = np.array(X_a)
Y = np.array(Y_a)

reg = LinearRegression().fit(X,Y)

X_s = []
Y_s = []
X_u = []
X_l = []
Y_u = []
Y_l = []
for j in range(2):
	var = random.random() * 2
	X_u.append([var])
	X_l.append([var])
	Y_u.append(reg.coef_[0] * var + reg.intercept_ + 0.3)
	Y_l.append(reg.coef_[0] * var + reg.intercept_ + 0.3 * (-1))

X_s = X_u + X_l
Y_s = Y_u + Y_l

X_up = np.array(X_u + X_a)
Y_up = np.array(Y_u + Y_a)
X_oup = np.array(X_u)
Y_oup = np.array(Y_u)

X_low = np.array(X_l + X_a)
Y_low = np.array(Y_l + Y_a)
X_olow = np.array(X_l)
Y_olow = np.array(Y_l)

X_cond = np.array(X_s)
Y_cond = np.array(Y_s)

reg2 = LinearRegression().fit(X_cond, Y_cond)
reg3 = LinearRegression().fit(X_up, Y_up)
reg4 = LinearRegression().fit(X_low, Y_low)


mn=np.min(X)
mx=np.max(X)
x1=np.linspace(mn,mx,500)
y1=reg.coef_[0]*x1+reg.intercept_
y2=reg2.coef_[0]*x1+reg2.intercept_
fig, ax = plt.subplots()
ax.plot(X,Y,'ob', label="Old points")
ax.scatter(X_cond,Y_cond,marker='x', color='red', linewidth=3,label="New factual update")
ax.plot(x1,y1,'-b')
ax.plot(x1,y2,'-b')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)
plt.show()	

y1=reg.coef_[0]*x1+reg.intercept_
y2=reg2.coef_[0]*x1+reg2.intercept_
y3=reg3.coef_[0]*x1+reg3.intercept_
y4=reg4.coef_[0]*x1+reg4.intercept_
fig, ax = plt.subplots()
ax.plot(X,Y,'ob', label="Old points")
ax.scatter(X_oup,Y_oup,marker='x', color='red', linewidth=3, label="with upper factual update")
ax.scatter(X_olow,Y_olow,color='orange', marker='x', linewidth=3, label="with lower factual update")
ax.plot(x1,y1,'-b')
ax.plot(x1,y2,'-b')
ax.plot(x1,y3,'-r')
ax.plot(x1,y4,c='orange')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)
plt.show()	

