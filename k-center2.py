import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

random.seed(10)


d = 1
all_outputs = []

eps = 1/4
M = 3 * eps

rand_points = []

for round in range(5):
    rand_points.append(random.random() * 4 * eps - 2*eps)

centers1 = [1,10,100]
strings1 = ["1", "10", "100"]
points1 = [-eps, 0, eps, 2]
sp1 = ["$-\epsilon$", "0", "$\epsilon$", "2"]

l = max(rand_points)


#plt.rcParams["figure.figsize"] = [7.00, 3.50]
#plt.rcParams["figure.autolayout"] = True


fig, ax = plt.subplots()

ax.scatter(points1 + rand_points,[1 for p in points1 + rand_points],color='blue', marker='o', label="Points", zorder = 2)
ax.scatter(centers1,[1 for c in centers1],color='red', marker='*', label="Centers", zorder = 2)
for ind in range(len(centers1)):
    ax.text(x=centers1[ind] - 0.003, y = 0.99, s = strings1[ind])

for ind in range(len(points1)):
    ax.text(x=points1[ind] , y = 0.99, s = sp1[ind])

ax.plot(centers1 + points1 + rand_points, [1 for elem in centers1 + points1 + rand_points], c='black', zorder = 3)
textstr = r"$\overbrace{\ \ \ \ \ \ }^{C \geq M - \epsilon}$"
ax.annotate(r"$\}$",fontsize=24,
            xy=(0.27, 0.77), xycoords='figure fraction', transform = ax.transAxes,
            )

ax.text(0.1, 1.01, textstr, fontsize = 20)#, transform=ax.transAxes, fontsize=20, verticalalignment='top')
plt.xscale(value="symlog")

mult = [-1,1]
vals = ["$-2\epsilon$", "$2\epsilon$"]

for i in range(2):
    val = mult[i] * 2 * eps
    plt.axvline(x=val, ymin=0.47, ymax=0.53, color='b', label='axvline - % of full height')
    ax.text(x=val, y = 0.99, s = vals[i])


plt.show()


centers1 = [l,10,100]
strings1 = ["", "10", "100"]
points1 = [-eps, 0, eps, 1, 2]
sp1 = ["$-\epsilon$", "0", "$\epsilon$", "1", "2"]
rand_points.remove(l)

fig, ax = plt.subplots()

ax.scatter(points1 + rand_points,[1 for p in points1 + rand_points],color='blue', marker='o', label="Points", zorder = 2)
ax.scatter(centers1,[1 for c in centers1],color='red', marker='*', label="Centers", zorder = 2)
for ind in range(len(centers1)):
    ax.text(x=centers1[ind] - 0.003, y = 0.99, s = strings1[ind])

for ind in range(len(points1)):
    ax.text(x=points1[ind] , y = 0.99, s = sp1[ind])

ax.plot(centers1 + points1 + rand_points, [1 for elem in centers1 + points1 + rand_points], c='black', zorder = 3)
textstr = r"$\overbrace{\ \ \ \ \ \ }^{C \geq M - \epsilon}$"
ax.annotate(r"$\}$",fontsize=24,
            xy=(0.27, 0.77), xycoords='figure fraction', transform = ax.transAxes,
            )

ax.text(0.1, 1.01, textstr, fontsize = 20)#, transform=ax.transAxes, fontsize=20, verticalalignment='top')
plt.xscale(value="symlog")

mult = [-1,1]
vals = ["$-2\epsilon$", "$2\epsilon$"]

for i in range(2):
    val = mult[i] * 2 * eps
    plt.axvline(x=val, ymin=0.47, ymax=0.53, color='b', label='axvline - % of full height')
    ax.text(x=val, y = 0.99, s = vals[i])


plt.show()
