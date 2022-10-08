import pickle
import numpy as np
from numpy.random import Generator, PCG64
rng = Generator(PCG64(12345))
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

# n_x0 = 10
# n_l0 = 10

n = 10

# Dump the start perturbations into a file
# xy_min_start = [-0.1, -0.1]
# xy_max_start = [0.05, 0.1]
# lol_start = rng.uniform(low=xy_min_start, high=xy_max_start, size=(n,2))
# start_list_tuples = list(lol_start)

# with open('start_perturbations_crane.pkl', 'wb') as fp:
#     pickle.dump(start_list_tuples, fp)

# # Dump the end perturbations into a file
# xy_min_end = [-0.1, -0.1]
# xy_max_end = [0.1, 0.1]
# lol_end = rng.uniform(low=xy_min_end, high=xy_max_end, size=(n,2))
# end_list_tuples = list(lol_end)

# with open('end_perturbations_crane.pkl', 'wb') as fp:
#     pickle.dump(end_list_tuples, fp)


#### ----- Check perturbation points and their feasibility -----
# Reload the starting list from the file
with open ('start_perturbations_crane.pkl', 'rb') as fp:
    start_list_tuples = pickle.load(fp)
# print(start_list_tuples)

# Reload the end list from file
with open ('end_perturbations_crane.pkl', 'rb') as fp:
    end_list_tuples = pickle.load(fp)
# print(end_list_tuples)

start_point = np.array([0.0, -0.9])
end_point = np.array([0.5, -0.9])

_, ax = plt.subplots()
ax.set_aspect('equal')
rect = Rectangle((0.1, -2), 0.1, 1.3, linewidth=1,
                 edgecolor='r', facecolor='none')
ax.add_patch(rect)

rect_l = Rectangle((-0.1, -1), 0.15, 0.2, linewidth=1,
                 edgecolor='k', facecolor='none')
ax.add_patch(rect_l)
rect_r = Rectangle((0.4, -1), 0.2, 0.2, linewidth=1,
                 edgecolor='k', facecolor='none')
ax.add_patch(rect_r)

ax.plot(start_point[0], start_point[1], color='r', marker='x')
ax.plot(end_point[0], end_point[1], color='r', marker='x')

for i in range(len(start_list_tuples)):
    ax.plot(start_point[0]+start_list_tuples[i][0], start_point[1]+start_list_tuples[i][1], color='b', marker='x')
for i in range(len(end_list_tuples)):
    ax.plot(end_point[0]+end_list_tuples[i][0], end_point[1]+end_list_tuples[i][1], color='b', marker='x')
ax.set_ylim((-1.05, -0.5))
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Overhead Crane, P2P Motion with obstacle')
plt.show()