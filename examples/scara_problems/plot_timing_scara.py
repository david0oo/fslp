import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from datetime import datetime
import seaborn as sns

def latexify():
    params = {'backend': 'ps',
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

# df_ipopt = pd.read_pickle('feasible_ipopt_stats_crane_MA5725-03-2022.pkl')
# df_ipopt_50FE = pd.read_pickle('feasible_ipopt_stats_crane_FE50_MA5729-03-2022.pkl')
# df_ipopt_70FE = pd.read_pickle('feasible_ipopt_stats_crane_FE70_MA5729-03-2022.pkl')
df_ipopt = pd.read_pickle('feasible_ipopt_stats_crane_FE100_MA5720-06-2022.pkl')

# df_fpsqp = pd.read_pickle('feasible_slp_stats_crane_wCPLEX25-03-2022.pkl')
# df_fpsqp_50FE = pd.read_pickle('feasible_slp_stats_crane_wCPLEX_FE5029-03-2022.pkl')
# df_fpsqp_70FE = pd.read_pickle('feasible_slp_stats_crane_wCPLEX_FE7029-03-2022.pkl')
df_fpsqp = pd.read_pickle('feasible_slp_stats_scara_wCPLEX_whole_dataset20-06-2022.pkl')

# df_fpsqp = pd.read_pickle('feasible_fp_stats_crane27-02-2022.pkl')
pd.set_option("max_rows", None)
indexNames = df_ipopt[df_ipopt['success'] == False ].index
df_fpsqp.drop(indexNames, inplace=True)
df_ipopt.drop(indexNames, inplace=True)
indexNames = df_ipopt[df_ipopt['t_wall_total'] > 500 ].index
df_fpsqp.drop(indexNames, inplace=True)
df_ipopt.drop(indexNames, inplace=True)

data1 = pd.concat([df_ipopt['t_wall_total'], df_fpsqp['t_wall'], df_fpsqp['t_wall_zero_slacks']], axis=1).assign(FiniteElements=20)
data1.rename(columns={"t_wall_total": "Ipopt", "t_wall": "FSLP - optimal", "t_wall_zero_slacks": "FSLP - feasible"})
# data1 = pd.concat([df_ipopt['t_wall_total'], df_fpsqp['t_wall']], axis=1).assign(FiniteElements=20)
# data1.rename(columns={"t_wall_total": "Ipopt", "t_wall": "FSLP - optimal"})

# data2 = pd.concat([df_ipopt_50FE['t_wall_total'], df_fpsqp_50FE['t_wall'], df_fpsqp_50FE['t_wall_zero_slacks']], axis=1).assign(FiniteElements=50)
# data2.rename(columns={"t_wall_total": "Ipopt", "t_wall": "FSLP - optimal", "t_wall_zero_slacks": "FSLP - feasible"})
# data3 = pd.concat([df_ipopt_70FE['t_wall_total'], df_fpsqp_70FE['t_wall'], df_fpsqp_70FE['t_wall_zero_slacks']], axis=1).assign(FiniteElements=70)
# data3.rename(columns={"t_wall_total": "Ipopt", "t_wall": "FSLP - optimal", "t_wall_zero_slacks": "FSLP - feasible"})
# data4 = pd.concat([df_ipopt_100FE['t_wall_total'], df_fpsqp_100FE['t_wall'], df_fpsqp_100FE['t_wall_zero_slacks']], axis=1).assign(FiniteElements=100)
# data4.rename(columns={"t_wall_total": "Ipopt", "t_wall": "FSLP - optimal", "t_wall_zero_slacks": "FSLP - feasible"})

cdf = data1
# cdf = pd.concat([data1, data2, data3, data4])
mdf = pd.melt(cdf, id_vars=['FiniteElements'], var_name=['Letter'])
print(mdf.head())

plt.figure(figsize=(5,7))
ax = sns.boxplot(x="FiniteElements", y="value", hue="Letter", data=mdf)
ax.set_xlabel("number of internal discretization steps")
ax.set_ylabel("wall time [s]")
handles, _ = ax.get_legend_handles_labels()
ax.legend(handles, ['Ipopt', 'FSLP - optimal', 'FSLP - feasible'])
# ax.legend(handles, ['Ipopt', 'FSLP'])
# ax._legend.remove()
# plt.grid(alpha=0.5)
# plt.legend()
# plt.legend(labels=['Ipopt', 'FSLP - optimal', 'FSLP - feasible'], loc='upper left')    
plt.tight_layout()
plt.savefig(f"scara_boxplot_times.pdf", dpi=300, bbox_inches='tight', pad_inches=0.01)
plt.show()
