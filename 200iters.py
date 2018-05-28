# Benjamin Yi
# byi649
# 925302651

import random
import matplotlib.pyplot as plt
import math
import numpy as np

# Input data
with open("ProbA.txt", 'r') as f:
    w = [line.strip() for line in f.readlines()]

n = int(w[0])
w = [0] + [float(x) for x in w[1:]]

with open("Positions.txt", 'r') as f:
    pos = [line.strip() for line in f.readlines()[1:]]

x = [float(tup.split()[1]) for tup in pos]
y = [float(tup.split()[2]) for tup in pos]

bestZ = float("inf")

for k in range(200):

    # Generate random starting solutions
    s = list(range(1, n+1)) + [0]*(120-n)
    for i in range(119):
        j = random.randint(i+1, 119)
        s[i], s[j] = s[j], s[i]

    # Iterate next-descent
    zx = sum([w[s[i]]*x[i] for i in range(120)])
    zy = sum([w[s[i]]*y[i] for i in range(120)])
    z = 5 * abs(zy) + abs(zx)

    converged = False
    last_swap = (0, 0)

    while(not converged):
        for a in range(120):
            for b in range(a + 1, 120):
                # Stop if the neighbourhood surrounding the last swap has been searched
                if (a, b) == last_swap:
                    converged = True
                else:
                    if b != a + 60 and w[s[a]] != w[s[b]]:
                        zy_new = zy - w[s[a]]*y[a] + w[s[b]]*y[a] - w[s[b]]*y[b] + w[s[a]]*y[b]
                        zx_new = zx - w[s[a]]*x[a] + w[s[b]]*x[a] - w[s[b]]*x[b] + w[s[a]]*x[b]
                        z_new = 5 * abs(zy_new) + abs(zx_new)

                        if z_new < z:
                            # Update and swap
                            z, zy, zx = z_new, zy_new, zx_new
                            s[a], s[b] = s[b], s[a]
                            last_swap = (a, b)

                # In the extremely rare case the shuffle gave us a local minima
                if last_swap == (0, 0) and (a, b) == (119, 120):
                    converged = True

            if converged:
                break

    totalweight = sum(w)
    z = z/totalweight

    if z < bestZ:
        bestZ, bestS = z, s

print(bestZ)

with open("Results.txt", 'w') as f:
    f.write(str(bestZ)+"\n")
    f.write("\n".join([str(x) for x in bestS]))