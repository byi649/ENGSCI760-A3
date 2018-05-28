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

# Do two next-descent on problem A
zArray = []
bestzArray = []

for k in range(2):

    # Generate random starting solutions
    s = list(range(1, n+1)) + [0]*(120-n)
    for i in range(119):
        j = random.randint(i+1, 119)
        s[i], s[j] = s[j], s[i]

    # Iterate next-descent
    zx = sum([w[s[i]]*x[i] for i in range(120)])
    zy = sum([w[s[i]]*y[i] for i in range(120)])
    z = 5 * abs(zy) + abs(zx)

    zArray.append(z)
    bestzArray.append(z)

    converged = False
    last_swap = (119, 119) # TODO: edge case where we converge with no swaps on first iteration
    while(not converged):
        for a in range(120):
            for b in range(a + 1, 120):
                if (a, b) == last_swap:
                    converged = True
                else:
                    if b != a + 60 and w[s[a]] != w[s[b]]:
                        zy_new = zy - w[s[a]]*y[a] + w[s[b]]*y[a] - w[s[b]]*y[b] + w[s[a]]*y[b]
                        zx_new = zx - w[s[a]]*x[a] + w[s[b]]*x[a] - w[s[b]]*x[b] + w[s[a]]*x[b]
                        z_new = 5 * abs(zy_new) + abs(zx_new)

                        zArray.append(z_new)

                        if z_new < z:
                            z = z_new
                            zy = zy_new
                            zx = zx_new
                            s[a], s[b] = s[b], s[a]
                            last_swap = (a, b)

                        bestzArray.append(z)

    bestzArray.append(np.nan) # To create a vertical break between descents
    totalweight = sum(w)
    print("dY =", zy/totalweight)
    print("dX =", zx/totalweight)
    print("z =", z/totalweight)

plt.scatter(x=range(len(zArray)), y=[math.log(x) for x in zArray], marker='x', alpha=0.5, s=0.2)
plt.plot([math.log(x) for x in bestzArray], 'r')
plt.xlabel("Function evaluation count")
plt.ylabel("Solution quality (log)")
plt.title("Next descent local search")
plt.show()

with open("Results.txt", 'w') as f:
    f.write(str(z/totalweight)+"\n")
    f.write("\n".join([str(x) for x in s]))

