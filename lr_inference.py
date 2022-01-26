import random
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

random.seed(5)

d = 1
all_outputs = []

X_l = []
Y_l = []

for i in range(4):
	X_l.append([random.random() * 2])
	Y_l.append(random.random() * 2)

X = np.array(X_l)
Y = np.array(Y_l)

#X = np.array([[0],[0],[2],[2]])
#Y = np.array([-0.5,0.5,1.5,2.5])


#X = np.array([[0], [1],[2]])
#Y = np.dot(X, np.array([1])) + 0
#Y = np.array([[0], [1]])

X_perm = np.copy(X)
Y_perm = np.copy(Y)


X_l = []
Y_l = []
for i in range(4):
	X_l.append([random.random() * 2])
	Y_l.append(random.random() * 2)

X_fact = np.array(X_l)
Y_fact = np.array(Y_l)
X_tot = np.concatenate([X_perm,X_fact])
Y_tot = np.concatenate([Y_perm,Y_fact])
reg = LinearRegression().fit(X_tot, Y_tot)
mn=0
mx=2
x1=np.linspace(mn,mx,500)
y1=reg.coef_[0]*x1+reg.intercept_
fig, ax = plt.subplots()
ax.scatter(X_fact,Y_fact,color='orange', marker='x', linewidth=3, label="Factual points of agent")
ax.scatter(X_perm,Y_perm,color='red', marker='o', label="Underlying points by other agents")
ax.plot(x1,y1,'-r')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)
plt.ylim(0,3)
#plt.show()	



vd = np.array([[]])

X_tri = np.array([[]])
y_tri = np.array([])

for i in range(d+2):
	reg = LinearRegression().fit(X, Y)
	#print(reg.coef_)
	#print(reg.intercept_)
	all_outputs.append([reg.intercept_,reg.coef_[0]])

	if i == 0:
		x = np.array([[0]])
		y = np.array([reg.intercept_ + 1])
	else:
		if i == 1:
			vd = np.array([[y[0] - reg.intercept_,0]])
		else:
			res = y[0]  - reg.coef_[0] - reg.intercept_
			vd = np.concatenate([vd,np.array([[res + z - reg.intercept_,res]])])
		x = np.array([[1]])
		y = np.array([reg.coef_[0] + reg.intercept_ + 1])
		z = reg.intercept_

	x1=np.linspace(mn,mx,500)
	y1=reg.coef_[0]*x1+reg.intercept_
	fig, ax = plt.subplots()
	ax.scatter(X_tri,y_tri,color='blue', marker='v', linewidth=3, label="Triangulation points")
	ax.plot(X_perm,Y_perm,'or', label="Underlying points by other agents")
	ax.plot(x1,y1,'-r')
	ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)
	plt.ylim(0,3)
	#plt.show()	

	if i < d+1:

		X = np.concatenate([X,x])
		Y = np.concatenate([Y,y])
		if i == 0:
			X_tri = np.copy(x)
			y_tri = np.copy(y)
		else:
			X_tri = np.concatenate([X_tri,x])
			y_tri = np.concatenate([y_tri,y])

print(all_outputs)
print("X: {}".format(X))
print("Y: {}".format(Y))


ao = np.array(all_outputs)
rho = np.stack([ao[1]-ao[0],ao[2]-ao[1]])

rho_t = np.matrix.transpose(rho)
vd_t = np.matrix.transpose(vd)


 
print(rho_t)
print(vd_t)


inv_rho = np.linalg.inv(rho_t)
fin_x = np.matmul(vd_t, inv_rho)

print(fin_x)
fin_y = np.matmul(fin_x, ao[0])
print(fin_y)

import pdb
pdb.set_trace()
