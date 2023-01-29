import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from datetime import datetime
import seaborn as sns

def latexify():
    params = {#'backend': 'ps',
            'axes.labelsize': 10,
            'axes.titlesize': 10,
            'legend.fontsize': 10,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'text.usetex': True,
            'font.family': 'serif'
    }

    matplotlib.rcParams.update(params)


latexify()

df_ipopt = pd.read_pickle('feasible_ipopt_stats_crane_08-09-2022.pkl')
df_ipopt_50FE = pd.read_pickle('feasible_ipopt_stats_crane_FE50_09-09-2022.pkl')
df_ipopt_70FE = pd.read_pickle('feasible_ipopt_stats_crane_FE70_09-09-2022.pkl')
df_ipopt_100FE = pd.read_pickle('feasible_ipopt_stats_crane_FE100_09-09-2022.pkl')

df_fpsqp = pd.read_pickle('fslp_stats_crane_08-09-2022.pkl')
df_fpsqp_50FE = pd.read_pickle('fslp_stats_crane_FE50_09-09-2022.pkl')
df_fpsqp_70FE = pd.read_pickle('fslp_stats_crane_FE70_09-09-2022.pkl')
df_fpsqp_100FE = pd.read_pickle('fslp_stats_crane_FE100_09-09-2022.pkl')

df_fslp_andersonM1 = pd.read_pickle('fslp_stats_crane_wANDERSONm1_08-09-2022.pkl')

# df_fpsqp = pd.read_pickle('feasible_fp_stats_crane27-02-2022.pkl')
# pd.set_option("max_rows", None)
indexNames = df_fpsqp[df_fpsqp['success'] == False ].index
df_fpsqp.drop(indexNames, inplace=True)
df_ipopt.drop(indexNames, inplace=True)

data1 = pd.concat([df_ipopt['t_wall_total'], df_fpsqp['t_wall'], df_fpsqp['t_wall_zero_slacks']], axis=1).assign(FiniteElements=20)
data1.rename(columns={"t_wall_total": "Ipopt", "t_wall": "FSLP - optimal", "t_wall_zero_slacks": "FSLP - feasible"})
data2 = pd.concat([df_ipopt_50FE['t_wall_total'], df_fpsqp_50FE['t_wall'], df_fpsqp_50FE['t_wall_zero_slacks']], axis=1).assign(FiniteElements=50)
data2.rename(columns={"t_wall_total": "Ipopt", "t_wall": "FSLP - optimal", "t_wall_zero_slacks": "FSLP - feasible"})
data3 = pd.concat([df_ipopt_70FE['t_wall_total'], df_fpsqp_70FE['t_wall'], df_fpsqp_70FE['t_wall_zero_slacks']], axis=1).assign(FiniteElements=70)
data3.rename(columns={"t_wall_total": "Ipopt", "t_wall": "FSLP - optimal", "t_wall_zero_slacks": "FSLP - feasible"})
data4 = pd.concat([df_ipopt_100FE['t_wall_total'], df_fpsqp_100FE['t_wall'], df_fpsqp_100FE['t_wall_zero_slacks']], axis=1).assign(FiniteElements=100)
data4.rename(columns={"t_wall_total": "Ipopt", "t_wall": "FSLP - optimal", "t_wall_zero_slacks": "FSLP - feasible"})

cdf = pd.concat([data1, data2, data3, data4])
mdf = pd.melt(cdf, id_vars=['FiniteElements'], var_name=['Letter'])
print(mdf.head())

plt.figure(figsize=(5,5))
ax = sns.boxplot(x="FiniteElements", y="value", hue="Letter", data=mdf)
ax.margins(x=2)
ax.set_xlabel("number of internal discretization steps")
ax.set_ylabel("wall time [s]")
handles, _ = ax.get_legend_handles_labels()
ax.legend(handles, ['Ipopt', 'FSLP - optimal', 'FSLP - feasible'])
# ax._legend.remove()
plt.grid(alpha=0.5)
# plt.legend()
# plt.legend(labels=['Ipopt', 'FSLP - optimal', 'FSLP - feasible'], loc='upper left')    
plt.tight_layout()
# plt.savefig(f"boxplot_times.pdf", dpi=300, bbox_inches='tight', pad_inches=0.01)
# plt.show()

# %% Plot boxplot in different subplots next to each other
fig, axes = plt.subplots(1, 4, figsize=(15, 5), sharey=True)
fig.suptitle('Number of Internal Discretization Steps')

data1 = pd.concat([df_ipopt['t_wall_total'], df_fpsqp['t_wall'], df_fpsqp['t_wall_zero_slacks']], axis=1)
data1.columns = ["Ipopt", "FSLP - optimal", "FSLP - feasible"]
# data1.boxplot(column=["Ipopt", "FSLP - optimal", "FSLP - feasible"])
# print(data1.head())
# print(data1)
sns.boxplot(ax=axes[0], x="variable", y="value", data=pd.melt(data1))
axes[0].set_title('20 steps')
axes[0].set_ylabel('wall time [s]')
axes[0].set_xlabel(' ')
axes[0].grid(alpha=0.5)
# plt.ylabel("wall time [s]")
# plt.ylabel("wall time [s]")

data2 = pd.concat([df_ipopt_50FE['t_wall_total'], df_fpsqp_50FE['t_wall'], df_fpsqp_50FE['t_wall_zero_slacks']], axis=1)
data2.columns = ["Ipopt", "FSLP - optimal", "FSLP - feasible"]
sns.boxplot(ax=axes[1], x="variable", y="value", data=pd.melt(data2))
axes[1].set_ylabel(' ')
axes[1].set_xlabel(' ')
axes[1].set_title('50 steps')
axes[1].grid(alpha=0.5)

data3 = pd.concat([df_ipopt_70FE['t_wall_total'], df_fpsqp_70FE['t_wall'], df_fpsqp_70FE['t_wall_zero_slacks']], axis=1)
data3.columns = ["Ipopt", "FSLP - optimal", "FSLP - feasible"]
sns.boxplot(ax=axes[2], x="variable", y="value", data=pd.melt(data3))
axes[2].set_title('70 steps')
axes[2].set_ylabel(' ')
axes[2].set_xlabel(' ')
axes[2].grid(alpha=0.5)

data4 = pd.concat([df_ipopt_100FE['t_wall_total'], df_fpsqp_100FE['t_wall'], df_fpsqp_100FE['t_wall_zero_slacks']], axis=1)
data4.columns = ["Ipopt", "FSLP - optimal", "FSLP - feasible"]
sns.boxplot(ax=axes[3], x="variable", y="value", data=pd.melt(data4))
axes[3].set_title('100 steps')
axes[3].set_ylabel(' ')
axes[3].set_xlabel(' ')
axes[3].grid(alpha=0.5)
plt.tight_layout()
plt.savefig(f"flandersmake_boxplot_times.pdf", dpi=300, bbox_inches='tight', pad_inches=0.01)
# data2.boxplot(column=["Ipopt", "FSLP - optimal", "FSLP - feasible"])
# plt.ylim((-0.75, 26))
# ax = sns.boxplot(x="variable", y="value", data=pd.melt(data2))
# ax.set_xlabel(" ")
# # ax.set_ylabel("wall time [s]")
# ax.set_ylim(-0.75,26)
# plt.grid(alpha=0.5)
# plt.tight_layout()
plt.show()
