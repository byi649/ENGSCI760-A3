import random
import matplotlib.pyplot as plt
import math
import numpy as np
from collections import deque

# Input data
with open("ProbA.txt", 'r') as f:
    w = [line.strip() for line in f.readlines()]

n = int(w[0])
w = [0] + [float(x) for x in w[1:]]

with open("Positions.txt", 'r') as f:
    pos = [line.strip() for line in f.readlines()[1:]]

x = [float(tup.split()[1]) for tup in pos]
y = [float(tup.split()[2]) for tup in pos]

# Do steepest-descent
zArray = []
bestzArray = []
h = min(20, int(n/3))
history = [0]*120

# Generate starting solution
s = list(range(1, n+1)) + [0]*(120-n)

# Iterate steepest-descent
zx = sum([w[s[i]]*x[i] for i in range(120)])
zy = sum([w[s[i]]*y[i] for i in range(120)])
z = 5 * abs(zy) + abs(zx)

zArray.append(z)
bestzArray.append(z)

converged = False
iter_count = 0
worsen = float("inf")
worsen_count = 0

while(not converged):
    neighbourhood = []
    iter_count += 1
    for a in range(120):
        for b in range(a + 1, 120):
            if (not (iter_count - history[s[a]] < h) and not (iter_count - history[s[b]] < h)) or history[s[a]] == 0 or history[s[b]] == 0:
                if b != a + 60 and w[s[a]] != w[s[b]]:
                    zy_new = zy - w[s[a]]*y[a] + w[s[b]]*y[a] - w[s[b]]*y[b] + w[s[a]]*y[b]
                    zx_new = zx - w[s[a]]*x[a] + w[s[b]]*x[a] - w[s[b]]*x[b] + w[s[a]]*x[b]
                    z_new = 5 * abs(zy_new) + abs(zx_new)

                    zArray.append(z_new)
                    neighbourhood.append((a, b, z_new, zy_new, zx_new))
                    bestzArray.append(z)

    if iter_count > 1e5 or worsen_count > 2*worsen:
        converged = True

    newSwap = min(neighbourhood, key=lambda t: t[2])

    if newSwap[2] > z:
        worsen = min(worsen, iter_count)

    if newSwap[2] > min(bestzArray):
        worsen_count += 1
    else:
        worsen_count = 0

    (a, b, z, zy, zx) = newSwap
    s[a], s[b] = s[b], s[a]
    history[s[a]] = iter_count
    history[s[b]] = iter_count


totalweight = sum(w)
print("dY =", zy/totalweight)
print("dX =", zx/totalweight)
print("z =", z/totalweight)

plt.scatter(x=range(len(zArray)), y=[math.log(x) for x in zArray], marker='x', alpha=0.5, s=0.2)
plt.plot([math.log(x) for x in bestzArray], 'r')
plt.xlabel("Function evaluation count")
plt.ylabel("Solution quality (log)")
plt.title("Tabu search")
plt.show()

with open("Results.txt", 'w') as f:
    f.write(str(z/totalweight)+"\n")
    f.write("\n".join([str(x) for x in s]))

