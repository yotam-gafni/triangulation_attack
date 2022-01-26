import random
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from copy import copy

random.seed(1)

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

D_mats = [np.zeros([d+1,d+1])]
D_mats[0][0][0] = 1
for i in range(1,d+1):
	D_mats.append(np.zeros([d+1,d+1]))
	D_mats[-1][0][0] = 1
	D_mats[-1][0][i] = 1
	D_mats[-1][i][0] = 1
	D_mats[-1][i][i] = 1

rho_mat = []
	

for i in range(d+2):
	reg = LinearRegression().fit(X, Y)

	l_rho = [reg.intercept_]
	for j in range(d):
		l_rho.append(reg.coef_[j])	
	rho_mat.append(np.array(l_rho))

	if i == 0:
		x = np.zeros([1,d])
		y = np.array([reg.intercept_ + 1])
	elif i < d+1:
		x = np.zeros([1,d])
		x[0][i-1] = 1 
		y = np.array([reg.coef_[i-1] + reg.intercept_ + 1])
		#z = reg.intercept_

	X = np.concatenate([X,x])
	Y = np.concatenate([Y,y])
	if i == 0:
		X_tri = np.copy(x)
		y_tri = np.copy(y)
	else:
		X_tri = np.concatenate([X_tri,x])
		y_tri = np.concatenate([y_tri,y])


for i in range(d+1):
	new_diff = rho_mat[i+1] - rho_mat[i]
	
	sumd = np.zeros([d+1,d+1])
	for j in range(i):
		sumd += D_mats[j]
	v = np.zeros([d+1])
	if i == 0:
		v[0] = rho_mat[i][0] + 1
	else:
		v[0] = rho_mat[i][0] + rho_mat[i][i] + 1
		v[i] = v[0]
	if i == 0:
		rho_diff = copy(new_diff)
		w_mat = v - np.matmul(sumd,new_diff) - np.matmul(D_mats[i],rho_mat[i+1])
		print(w_mat)
	else:
		rho_diff = np.stack([rho_diff,new_diff])
		w_mat = np.stack([w_mat, v - np.matmul(sumd,new_diff) - np.matmul(D_mats[i],rho_mat[i+1])])
		print(w_mat)

rho_t = np.matrix.transpose(rho_diff)
w_t = np.matrix.transpose(w_mat)

print("X: {}".format(X))
print("Y: {}".format(Y))

print("Alg output diffs: {}".format(rho_t))
print("W matrix: {}".format(w_t))


inv_rho = np.linalg.inv(rho_t)
fin_x = np.matmul(w_t, inv_rho)

print("Inferred X^TX: {}".format(fin_x))
fin_y = np.matmul(fin_x, rho_mat[0])
print("Inferred X^Ty: {}".format(fin_y))


