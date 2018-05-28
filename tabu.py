# Benjamin Yi
# byi649
# 925302651

import matplotlib.pyplot as plt
import math

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
h = min(20, int(float(n)/3.0))
history = [-float('inf')]*120 # Arbitrary value < -h

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
best_index = float("inf") # Arbitrary value > 0
worsen_count = 0

while(not converged):
    neighbourhood = []
    iter_count += 1
    for a in range(120):
        # Skip tabu swaps
        if (not (iter_count - history[s[a]] < h)):
            for b in range(a + 1, 120):
                # Skip tabu swaps
                if (not (iter_count - history[s[b]] < h)):
                    if b != a + 60 and w[s[a]] != w[s[b]]:
                        zy_new = zy - w[s[a]]*y[a] + w[s[b]]*y[a] - w[s[b]]*y[b] + w[s[a]]*y[b]
                        zx_new = zx - w[s[a]]*x[a] + w[s[b]]*x[a] - w[s[b]]*x[b] + w[s[a]]*x[b]
                        z_new = 5 * abs(zy_new) + abs(zx_new)

                        zArray.append(z_new) # For plot
                        neighbourhood.append((a, b, z_new, zy_new, zx_new))
                        bestzArray.append(z) # For plot

    # Terminate at max iterations or stuck in local minima
    if iter_count > 1e5 or worsen_count > 2*best_index:
        converged = True

    # Best swap minimises z_new
    newSwap = min(neighbourhood, key=lambda t: t[2])

    # If our new solution is worse than the best we know
    if newSwap[2] > min(bestzArray):
        worsen_count += 1
    else:
        # We've found the best solution so far
        worsen_count = 0
        best_index = iter_count

    (a, b, z, zy, zx) = newSwap

    # Keep a record of recent swaps (not including empty containers)
    if s[a] != 0:
        history[s[a]] = iter_count
    if s[b] != 0:
        history[s[b]] = iter_count

    # Swap positions
    s[a], s[b] = s[b], s[a]


totalweight = sum(w)
z = min(bestzArray)/totalweight
print("z =", z)

plt.scatter(x=range(len(zArray)), y=[math.log(x/totalweight) for x in zArray], alpha=0.5, s=0.2)
plt.plot([math.log(x/totalweight) for x in bestzArray], 'r')
plt.xlabel("Function evaluation count")
plt.ylabel("Solution quality (log)")
plt.title("Tabu search")
plt.show()
