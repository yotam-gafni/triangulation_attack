from sklearn.svm import SVC
import random

cis = SVC(kernel='linear')

def rand_array(T):
	arr = []
	for i in range(T):
	        arr += [[random.random() * 1000]] * 2
	return arr, [-1,1] * T

def rand2_array(T):
	arr = []
	classif = []
	for i in range(T):
		arr += [[random.random() * 1000]]
		classif += [random.choice([1,-1])]
	return arr, classif

def opt_goal(points, types, w, b):
	goal = 0
	for i in range(len(points)):
	        goal += max(0, 1 - types[i] * (w * points[i][0] + b))
	fin = 1/2 * w**2 +  goal
	return fin


def valtry(T):
	all_res = []
	critical_k = -1
	for k in range(1,T):
		cis.fit([[0]]*103 + [[100]] * k + [[102]] * k, [-1] * 100 + [1] * 3 + [-1] * k + [1] * k)
		all_res.append(tuple([cis.coef_[0][0],cis.intercept_[0]]))
		if len(all_res) >= 2 and all_res[-1] != all_res[-2] and critical_k == -1:
			critical_k = len(all_res)
			print(critical_k)
		print(k)
		print(all_res[-1])
	return all_res

all_res = valtry(300)
import pdb
pdb.set_trace()
