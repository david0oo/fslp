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


plt.figure(figsize=(5,7.5))
# Plot the times for different initializations
# plt.subplot(3, 1, 1)
# plt.subplot(211)
n_probs = len(df_fpsqp)

iter_ipopt = df_fpsqp['t_wall'].value_counts().rename_axis('iterations').to_frame('counts')
cum_iter_ipopt = iter_ipopt.sort_values(by='iterations').cumsum()/n_probs
plt.step(np.array(cum_iter_ipopt.index), cum_iter_ipopt['counts'], label='FSLP - optimal', linestyle='solid')

iter_ipopt = df_ipopt['t_wall_total'].value_counts().rename_axis('iterations').to_frame('counts')
cum_iter_ipopt = iter_ipopt.sort_values(by='iterations').cumsum()/n_probs
plt.step(np.array(cum_iter_ipopt.index), cum_iter_ipopt['counts'], label='Ipopt', linestyle='dotted')

iter_ipopt = df_fpsqp['t_wall_zero_slacks'].value_counts().rename_axis('iterations').to_frame('counts')
cum_iter_ipopt = iter_ipopt.sort_values(by='iterations').cumsum()/n_probs
plt.step(np.array(cum_iter_ipopt.index), cum_iter_ipopt['counts'], label='FSLP - feasible', linestyle='dashed')

# iter_ipopt = df_ipopt_100FE['t_wall_total'].value_counts().rename_axis('iterations').to_frame('counts')
# cum_iter_ipopt = iter_ipopt.sort_values(by='iterations').cumsum()/n_probs
# plt.step(np.array(cum_iter_ipopt.index), cum_iter_ipopt['counts'], label='100 FE', linestyle='dashdot')

plt.grid(alpha=0.5)
plt.legend(loc='lower right')
plt.xlabel('wall time [s]')
plt.ylabel('ratio of solved problems')
# plt.tight_layout()
# plt.xlim((0,30))

# plt.subplot(212)
# plt.subplot(3, 1, 2)
# # Plot the times for more finite elements

# n_probs = len(df_fpsqp)

# iter_ipopt = df_fpsqp_50FE['t_wall'].value_counts().rename_axis('iterations').to_frame('counts')
# cum_iter_ipopt = iter_ipopt.sort_values(by='iterations').cumsum()/n_probs
# plt.step(cum_iter_ipopt['counts'], np.array(cum_iter_ipopt.index), label='FSLP - optimal', linestyle='solid')

# iter_ipopt = df_ipopt_50FE['t_wall_total'].value_counts().rename_axis('iterations').to_frame('counts')
# cum_iter_ipopt = iter_ipopt.sort_values(by='iterations').cumsum()/n_probs
# plt.step(cum_iter_ipopt['counts'], np.array(cum_iter_ipopt.index), label='Ipopt', linestyle='dotted')

# iter_ipopt = df_fpsqp_50FE['t_wall_zero_slacks'].value_counts().rename_axis('iterations').to_frame('counts')
# cum_iter_ipopt = iter_ipopt.sort_values(by='iterations').cumsum()/n_probs
# plt.step(cum_iter_ipopt['counts'], np.array(cum_iter_ipopt.index), label='FSLP - feasible', linestyle='dashed')

# # iter_ipopt = df_fpsqp_100FE['t_wall'].value_counts().rename_axis('iterations').to_frame('counts')
# # cum_iter_ipopt = iter_ipopt.sort_values(by='iterations').cumsum()/n_probs
# # plt.step(np.array(cum_iter_ipopt.index), cum_iter_ipopt['counts'], label='100 FE', linestyle='dashdot')
# plt.grid(alpha=0.5)
# plt.legend(loc='lower right')
# # plt.xlabel('wall time [s] - more finite elements')
# plt.ylabel('ratio of solved problems')
# # plt.xlim((0,30))

# plt.subplot(3, 1, 3)
# # plt.subplot(213)

# iter_ipopt = df_fpsqp_70FE['t_wall'].value_counts().rename_axis('iterations').to_frame('counts')
# cum_iter_ipopt = iter_ipopt.sort_values(by='iterations').cumsum()/n_probs
# plt.step(np.array(cum_iter_ipopt.index), cum_iter_ipopt['counts'], label='FSLP - optimal', linestyle='solid')

# iter_ipopt = df_ipopt_70FE['t_wall_total'].value_counts().rename_axis('iterations').to_frame('counts')
# cum_iter_ipopt = iter_ipopt.sort_values(by='iterations').cumsum()/n_probs
# plt.step(np.array(cum_iter_ipopt.index), cum_iter_ipopt['counts'], label='Ipopt', linestyle='dotted')

# iter_ipopt = df_fpsqp_70FE['t_wall_zero_slacks'].value_counts().rename_axis('iterations').to_frame('counts')
# cum_iter_ipopt = iter_ipopt.sort_values(by='iterations').cumsum()/n_probs
# plt.step(np.array(cum_iter_ipopt.index), cum_iter_ipopt['counts'], label='FSLP - feasible', linestyle='dashed')
# plt.grid(alpha=0.5)
# plt.legend(loc='lower right')
# plt.xlabel('wall time [s]')
# plt.ylabel('ratio of solved problems')
# plt.xlim((0,30))


plt.tight_layout()
plt.savefig(f"performance_probs_time.pdf", dpi=300, bbox_inches='tight', pad_inches=0.01)
plt.show()