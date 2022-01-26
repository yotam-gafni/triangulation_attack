from sklearn.svm import SVC
import random

est = SVC(kernel='linear')

for att in range(10):
	ra = []
	for i in range(4):
		ra.append(random.random())	
	est.fit([[ra[0]*1000],[ra[1]*1000],[ra[2]*-1000],[ra[3]*-1000]],[1,-1,1,-1])
	print(est.coef_[0][0])
	print(est.intercept_[0])
