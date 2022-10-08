import pickle
import numpy as np
import scaraParallel_workspacelimits as wspclim
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from numpy.random import Generator, PCG64
import casadi as cs
rng = Generator(PCG64(12345))

n = 10

# Dump the start perturbations into a file
# xy_min_start = [-0.02, 0]
# xy_max_start = [0.02, 0.045]
# lol_start = rng.uniform(low=xy_min_start, high=xy_max_start, size=(n,2))
# start_list_tuples = list(lol_start)

# with open('start_perturbations_scara.pkl', 'wb') as fp:
#     pickle.dump(start_list_tuples, fp)

# # Dump the end perturbations into a file
# xy_min_end = [0, -0.1]
# xy_max_end = [0.1, 0]
# lol_start = rng.uniform(low=xy_min_end, high=xy_max_end, size=(n,2))
# end_list_tuples = list(lol_start)

# with open('end_perturbations_scara.pkl', 'wb') as fp:
#     pickle.dump(end_list_tuples, fp)

#### ----- Check perturbation points and their feasibility -----
start_point = np.array([0.0, 0.115])
end_point = np.array([0.0, 0.405])

cog = cs.vertcat(0.0, 0.2)
c = 0.02
points = cs.vertcat(cs.horzcat(cog[0]-c/2, cog[1]-c/2),
                    cs.horzcat(cog[0]+c/2, cog[1]-c/2),
                    cs.horzcat(cog[0]+c/2, cog[1]+c/2),
                    cs.horzcat(cog[0]-c/2, cog[1]+c/2))

# Reload the starting list from the file
with open ('start_perturbations_scara.pkl', 'rb') as fp:
    start_list_tuples = pickle.load(fp)
print(start_list_tuples)

# Reload the end list from file
with open ('end_perturbations_scara.pkl', 'rb') as fp:
    end_list_tuples = pickle.load(fp)
print(end_list_tuples)

fig, ax = plt.subplots()
X_wrksp, Y_wrksp = wspclim.workspace_boundary()
ax.set_aspect('equal')
ax.plot(X_wrksp, Y_wrksp)
rect = Rectangle(tuple(np.array(points[0, :]).squeeze(
)), 0.02, 0.02, linewidth=1, edgecolor='r', facecolor='none')
ax.add_patch(rect)
ax.plot(start_point[0], start_point[1], color='r', marker='x')
for i in range(len(start_list_tuples)):
    ax.plot(start_point[0]+start_list_tuples[i][0], start_point[1]+start_list_tuples[i][1], color='b', marker='x')
ax.plot(end_point[0], end_point[1], color='r', marker='x')
for i in range(len(end_list_tuples)):
    ax.plot(end_point[0]+end_list_tuples[i][0], end_point[1]+end_list_tuples[i][1], color='b', marker='x')
ax.set_xlabel('$x_1$-axis')
ax.set_ylabel('$x_2$-axis')
ax.set_title('P2P-motion of scara robot')
plt.legend()
plt.grid(alpha=0.5)
plt.tight_layout()
plt.show()