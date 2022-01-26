import random
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from copy import copy

random.seed(10)

d = 1
all_outputs = []

X_l = []
Y_l = []
X_lext = []

for i in range(4):
	x = []
	for j in range(d):
		x.append(random.random() * 2)	
	y = random.random() * 2
	X_l.append(x)
	X_lext.append([1] + x)
	Y_l.append(y)

X = np.array(X_l)
X_ext = np.array(X_lext)
Y = np.array(Y_l)

X_perm = np.copy(X)
Y_perm = np.copy(Y)

X_lo = []
Y_lo = []
X_factex = []
for i in range(4):
	x = []
	for j in range(d):
		x.append(random.random() * 2)	
	y = random.random() * 2
	X_lo.append(x)
	X_factex.append([1] + x)
	Y_lo.append(y)

X_fact = np.array(X_lo)
Y_fact = np.array(Y_lo)
Xfex = np.array(X_factex)
X_tot = np.concatenate([X_perm,X_fact])
Y_tot = np.concatenate([Y_perm,Y_fact])
reg = LinearRegression().fit(X_tot, Y_tot)
res = [reg.intercept_, reg.coef_[0]]
mn=0
mx=2
x1=np.linspace(mn,mx,500)
x2=np.linspace(mn,mx,500)


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

for eps in [1/100,1/10,1]:
	rho_mat = []
	real_rho = []

	def eps_noise():
		return random.random() * eps - eps/2

	for i in range(d+2):
		reg = LinearRegression().fit(X, Y)

		l_rho = [reg.intercept_ + eps_noise()]
		for j in range(d):
			l_rho.append(reg.coef_[j])	
		rl_rho= [reg.intercept_] + l_rho[1:]
		rho_mat.append(np.array(l_rho))
		real_rho.append(np.array(rl_rho))

		if i == 0:
			x = np.zeros([1,d])
			y = np.array([l_rho[0] + 1])
		elif i < d+1:
			x = np.zeros([1,d])
			x[0][i-1] = 1 
			y = np.array([reg.coef_[i-1] + l_rho[0] + 1])
			#z = reg.intercept_

		if i < d+1:
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
		elif i == 1:
			rho_diff = np.stack([rho_diff,new_diff])
			w_mat = np.stack([w_mat, v - np.matmul(sumd,new_diff) - np.matmul(D_mats[i],rho_mat[i+1])])
			print(w_mat)
		else:
			rho_diff = np.concatenate([rho_diff,[new_diff]])
			w_mat = np.concatenate([w_mat, [v - np.matmul(sumd,new_diff) - np.matmul(D_mats[i],rho_mat[i+1])]])

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

	true_x = np.dot(X_ext.T,X_ext)
	true_y = np.dot(X_ext.T,Y_l)

	print("True X^TX: {}".format(true_x))
	print("True X^Ty: {}".format(true_y))


	truer_est = np.dot(np.linalg.inv(np.dot(Xfex.T,Xfex) + np.dot(X_ext.T,X_ext)),np.dot(Xfex.T,Y_fact) + np.dot(X_ext.T,Y_perm))
	true_est = np.dot(np.linalg.inv(true_x + np.dot(Xfex.T,Xfex)),true_y + np.dot(Xfex.T,Y_fact))
	dlr_est = np.dot(np.linalg.inv(fin_x + np.dot(Xfex.T,Xfex)),fin_y + np.dot(Xfex.T,Y_fact))
	print("DLR EST: {}".format(dlr_est))


	if d == 1:
		y1=res[1]*x1+res[0]
		y2=res[1]*x1+res[0] + eps_noise()
		y3= dlr_est[1]*x1 + dlr_est[0]
		fig, ax = plt.subplots()
		plt.ylim(0,3)
		plt.xlim(0,2)
		ax.scatter(X_fact, Y_fact,color='orange', marker='x', linewidth=3, label="Factual points of agent")
		ax.scatter(X_perm,Y_perm,color='red', marker='o', label="Underlying points by other agents")
		ax.plot(x1,y1,'-r')
		ax.plot(x1,y2,'-y')
		ax.plot(x1,y3,'-b')
		ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)
		plt.show()
