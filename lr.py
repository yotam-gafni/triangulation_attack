import random
import numpy as np

ROUNDS = 10000
EACH = 100
total_arr = [5 * elem for elem in np.random.rand(EACH * ROUNDS)]
alpha = 10
beta = 10
rand_bits_orig = [elem for elem in np.random.beta(a=alpha,b=beta,size=ROUNDS * 10)]
alpha = float(alpha)
rand_bits = [elem-alpha/(alpha+beta) for elem in rand_bits_orig]


curr_beta = 0

diff_est = []
diff_naive = []

for round in range(ROUNDS):
	arr = total_arr[EACH*round:EACH*(round+1)]
	X1 = arr[:5]
	Y1 = arr[5:10]

	X2 = []
	Y2 = []

	Xtrue = []
	Ytrue = []

	#X2.append(arr[10:15])
	#Y2.append(arr[15:20])
	
	X2.append([0 for i in range(5)])
	Y2.append([0 for i in range(5)])

	X2.append([1,0,0,0,0])
	Y2.append([0 for i in range(5)])

	Xtrue = arr[20:25]
	Ytrue = arr[25:30]

	a_true = sum([elem**2 for elem in Xtrue])
	c_true = sum([Xtrue[i]*Ytrue[i] for i in range(5)])

	#X2.append(arr[20:25])
	#Y2.append(arr[25:30])

	curr_rand = EACH * round + 30

	c = [0,0]
	a = [0,0]
	s = [0,0]
	n = [0,0]

	noisy_b = [0,0]
	noisy2_b = [0,0]

	n_val = (sum([X1[i]**2 for i in range(5)]) + c_true)
	b = (sum([X1[i] * Y1[i] for i in range(5)])+a_true)/n_val

	for j in range(2):
		for i in range(5):
			a[j] += X2[j][i]*Y2[j][i]
			c[j] += X2[j][i]**2


		X = np.concatenate((X1,X2[j]))
		Y = np.concatenate((Y1,Y2[j]))

		for i in range(10):
			s[j] += X[i] * Y[i]
			n[j] += X[i]**2


		noisy_b[j] = (s[j] + a[j] + rand_bits[curr_beta])/(n[j] + c[j])
		noisy2_b[j] = b + rand_bits[curr_beta]/n_val 
		curr_beta += 1

	naive_b1 = (noisy2_b[0] + noisy2_b[1]) / 2
	est_n1 = (noisy_b[1]*c[1] - noisy_b[0]*c[0] + a[0] - a[1])/(noisy_b[0]-noisy_b[1]) 
	est_s1 = noisy_b[0] * (est_n1 + c[0]) - a[0]
	est_b1 = (est_s1 + a_true) / (est_n1 + c_true)


	diff_naive.append(abs(naive_b1 - b))
	diff_est.append(abs(est_b1 - b))

print(sum(diff_naive) / len(diff_naive))
print(sum(diff_est) / len(diff_est))




 
