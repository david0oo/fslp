import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from datetime import datetime
import seaborn as sns

def latexify():
    params = {'backend': 'ps',
            #'text.latex.preamble': r"\usepackage{amsmath}",
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

# df_ipopt = pd.read_pickle('feasible_ipopt_stats_crane_T4_MA5726-03-2022.pkl')
df_ipopt = pd.read_pickle('feasible_ipopt_stats_scara_MA5729-09-2022.pkl')
# df_fpsqp = pd.read_pickle('feasible_slp_stats_scara_wIPOPT14-03-2022.pkl')
# df_fpsqp = pd.read_pickle('feasible_slp_stats_scara_wIPOPT20-03-2022.pkl')
# df_fpsqp_2 = pd.read_pickle('feasible_slp_stats_scara_wCPLEX_T426-03-2022.pkl')
df_fpsqp = pd.read_pickle('feasible_slp_stats_scara_wCPLEX_feas_iter29-09-2022.pkl')
df_fpsqp_anderson_m1 = pd.read_pickle('feasible_slp_stats_scara_wCPLEX_Anderson_m129-09-2022.pkl')
df_fpsqp_anderson_m5 = pd.read_pickle('feasible_slp_stats_scara_wCPLEX_Anderson_m529-09-2022.pkl')
# df_fpsqp = pd.read_pickle('feasible_slp_stats_crane_wCcd JUPLEX_10x10_10e-522-03-2022.pkl')

# df_fpsqp = pd.read_pickle('feasible_fp_stats_crane27-02-2022.pkl')
# pd.set_option("max_rows", None)
# indexNames = df_fpsqp[df_fpsqp['success'] == False ].index
# df_fpsqp.drop(indexNames, inplace=True)
# df_ipopt.drop(indexNames, inplace=True)


# plt.figure(figsize=(5,3))
# n_probs = len(df_fpsqp)
# iter_fpsqp = df_fpsqp['t_wall'].value_counts().rename_axis('iterations').to_frame('counts')
# cum_iter_fpsqp = iter_fpsqp.sort_values(by='iterations').cumsum()/n_probs
# plt.step(np.array(cum_iter_fpsqp.index), cum_iter_fpsqp['counts'], label='FSLP - optimal solution', linestyle='solid')
# iter_fpsqp = df_fpsqp['t_wall_zero_slacks'].value_counts().rename_axis('iterations').to_frame('counts')
# cum_iter_fpsqp = iter_fpsqp.sort_values(by='iterations').cumsum()/n_probs
# plt.step(np.array(cum_iter_fpsqp.index), cum_iter_fpsqp['counts'], label='FSLP - feasible solution', linestyle='dotted')
# iter_fpsqp = df_fpsqp_2['t_wall'].value_counts().rename_axis('iterations').to_frame('counts')
# cum_iter_fpsqp = iter_fpsqp.sort_values(by='iterations').cumsum()/n_probs
# plt.step(np.array(cum_iter_fpsqp.index), cum_iter_fpsqp['counts'], label='FSLP T4 - optimal solution', linestyle='dashed')
# iter_fpsqp = df_fpsqp_2['t_wall_zero_slacks'].value_counts().rename_axis('iterations').to_frame('counts')
# cum_iter_fpsqp = iter_fpsqp.sort_values(by='iterations').cumsum()/n_probs
# plt.step(np.array(cum_iter_fpsqp.index), cum_iter_fpsqp['counts'], label='FSLP T4 - feasible solution', linestyle='dashdot')
# plt.grid(alpha=0.5)
# plt.legend(loc='lower right')
# plt.xlabel('wall time [s]')
# plt.ylabel('ratio of solved problems')
# plt.tight_layout()
# # plt.savefig(f"performance_times_plot.pdf", dpi=300, bbox_inches='tight', pad_inches=0.01)
# plt.show()


# plt.figure(figsize=(5,2.5))
# n_probs = len(df_fpsqp)
# iter_fpsqp = df_fpsqp['t_wall'].value_counts().rename_axis('iterations').to_frame('counts')
# cum_iter_fpsqp = iter_fpsqp.sort_values(by='iterations').cumsum()/n_probs
# plt.step(np.array(cum_iter_fpsqp.index), cum_iter_fpsqp['counts'], label='FSLP - optimal solution', linestyle='solid')
# iter_ipopt = df_ipopt['t_wall_total'].value_counts().rename_axis('iterations').to_frame('counts')
# cum_iter_ipopt = iter_ipopt.sort_values(by='iterations').cumsum()/n_probs
# plt.step(np.array(cum_iter_ipopt.index), cum_iter_ipopt['counts'], label='IPOPT - optimal solution', linestyle='dotted')
# iter_fpsqp = df_fpsqp['t_wall_zero_slacks'].value_counts().rename_axis('iterations').to_frame('counts')
# cum_iter_fpsqp = iter_fpsqp.sort_values(by='iterations').cumsum()/n_probs
# plt.step(np.array(cum_iter_fpsqp.index), cum_iter_fpsqp['counts'], label='FSLP - feasible solution', linestyle='dashed')
# plt.grid(alpha=0.5)
# plt.legend(loc='lower right')
# plt.xlabel('wall time [s]')
# plt.ylabel('ratio of solved problems')
# plt.tight_layout()
# # plt.savefig(f"performance_times_plot.pdf", dpi=300, bbox_inches='tight', pad_inches=0.01)
# plt.show()


# relative cumulated results
# matplotlib.rcParams['axes.linewidth'] = 1
plt.figure(figsize=(5,5))
plt.subplot(211)
n_probs = len(df_fpsqp)
iter_fpsqp = df_fpsqp['iters'].value_counts().rename_axis('iterations').to_frame('counts')
cum_iter_fpsqp = iter_fpsqp.sort_values(by='iterations').cumsum()/n_probs
plt.step(np.array(cum_iter_fpsqp.index), cum_iter_fpsqp['counts'], label='FSLP - optimal solution', linestyle='solid')

# ax = cum_iter_fpsqp.plot(drawstyle="steps", figsize=(5,2.8), y='counts', kind = 'line', xlabel='Outer iteration number', ylabel='ratio of solved problems', label='FSLP - optimal solution', grid=True, linestyle='solid')
# ax.set_xlim(0,140)
# iter_ipopt = df_ipopt['iters'].value_counts().rename_axis('iterations').to_frame('counts')
# cum_iter_ipopt = iter_ipopt.sort_values(by='iterations').cumsum()/n_probs
# plt.step(np.array(cum_iter_ipopt.index), cum_iter_ipopt['counts'], label='IPOPT - optimal solution', linestyle='dotted')
# cum_iter_ipopt.plot(drawstyle="steps", ax=ax, y='counts', kind = 'line', label='IPOPT - optimal solution', xlabel='Outer iteration number',grid=True, linestyle='dotted')

iter_fpsqp_feas = df_fpsqp['slack_zero_iter'].value_counts().rename_axis('iterations').to_frame('counts')
cum_iter_fpsqp_feas = iter_fpsqp_feas.sort_values(by='iterations').cumsum()/n_probs
plt.step(np.array(cum_iter_fpsqp_feas.index), cum_iter_fpsqp_feas['counts'], label='FSLP - feasible solution', linestyle='dashed')
# cum_iter_fpsqp_feas.plot(drawstyle="steps", ax=ax, y='counts', kind = 'line', label='FSLP - feasible solution', xlabel='Outer iteration number',grid=True, linestyle='dashed')

iter_fpsqp_andersonM1 = df_fpsqp_anderson_m1['iters'].value_counts().rename_axis('iterations').to_frame('counts')
cum_iter_fpsqp = iter_fpsqp_andersonM1.sort_values(by='iterations').cumsum()/n_probs
plt.step(np.array(cum_iter_fpsqp.index), cum_iter_fpsqp['counts'], label='FSLP Am1 - optimal solution', linestyle='solid')

iter_fpsqp_andersonM1_feas = df_fpsqp_anderson_m1['slack_zero_iter'].value_counts().rename_axis('iterations').to_frame('counts')
cum_iter_fpsqp_feas = iter_fpsqp_andersonM1_feas.sort_values(by='iterations').cumsum()/n_probs
plt.step(np.array(cum_iter_fpsqp_feas.index), cum_iter_fpsqp_feas['counts'], label='FSLP Am1 - feasible solution', linestyle='dashed')

iter_fpsqp_andersonM5 = df_fpsqp_anderson_m5['iters'].value_counts().rename_axis('iterations').to_frame('counts')
cum_iter_fpsqp = iter_fpsqp_andersonM5.sort_values(by='iterations').cumsum()/n_probs
plt.step(np.array(cum_iter_fpsqp.index), cum_iter_fpsqp['counts'], label='FSLP Am5 - optimal solution', linestyle='solid')

iter_fpsqp_andersonM5_feas = df_fpsqp_anderson_m5['slack_zero_iter'].value_counts().rename_axis('iterations').to_frame('counts')
cum_iter_fpsqp_feas = iter_fpsqp_andersonM5_feas.sort_values(by='iterations').cumsum()/n_probs
plt.step(np.array(cum_iter_fpsqp_feas.index), cum_iter_fpsqp_feas['counts'], label='FSLP Am5 - feasible solution', linestyle='dashed')

# Calculate IPOPT feasibility
lol_obj = []
lol_inf = []
for i in range(df_ipopt['obj'].shape[0]):
    lol_obj.append(np.min(np.where(np.array(df_ipopt['obj'][i]) < 100)))
for i in range(df_ipopt['inf_pr'].shape[0]):
    lol = np.where(np.array(df_ipopt['inf_pr'][i]) < 1e-7)[0]
    if not lol.any():
        print('here')
        lol_inf.append(len(df_ipopt['inf_pr'][i]))
        continue
    lol_tmp = lol
    for j in range(lol.shape[0]):
        if lol_tmp.shape[0]==1 or np.all(lol_tmp[1:] - lol_tmp[:-1] == 1):
            lol_inf.append(np.min(lol_tmp))
            break
        else:
            lol_tmp = lol_tmp[1:]
zero_slacks_ipopt = np.maximum(lol_obj, lol_inf)
df_ipopt['zero_slacks'] = zero_slacks_ipopt

iter_ipopt_feas = df_ipopt['zero_slacks'].value_counts().rename_axis('iterations').to_frame('counts')
cum_iter_ipopt_feas = iter_ipopt_feas.sort_values(by='iterations').cumsum()/n_probs
# cum_iter_ipopt_feas.plot(drawstyle="steps", ax=ax, y='counts', kind = 'line', label='IPOPT - feasible solution', xlabel='outer iteration number',grid=True, linestyle='dashdot')
# plt.step(np.array(cum_iter_ipopt_feas.index), cum_iter_ipopt_feas['counts'], label='IPOPT - feasible solution', linestyle='dashdot')
plt.grid(alpha=0.5)
plt.legend(loc='lower right')
plt.xlabel('outer iteration number')
plt.ylabel('ratio of solved problems')

# plt.grid(alpha=0.5)
# plt.xlim((0,140))
# plt.step(np.array(cum_iter_fpsqp.index), cum_iter_fpsqp['counts'], label='FSLP - optimal solution', linestyle='solid')

plt.subplot(212)
n_probs = len(df_fpsqp)
iter_fpsqp = df_fpsqp['n_eval_g'].value_counts().rename_axis('iterations').to_frame('counts')
cum_iter_fpsqp = iter_fpsqp.sort_values(by='iterations').cumsum()/n_probs
# ax = cum_iter_fpsqp.plot(drawstyle="steps", figsize=(5,3), y='counts', kind = 'line', xlabel='Number of constraint evaluations', ylabel='ratio of solved problems', label='FSLP', grid=True, linestyle='solid')
plt.step(np.array(cum_iter_fpsqp.index), cum_iter_fpsqp['counts'], label='FSLP - optimal solution', linestyle='solid')
plt.xlim(0,500)

iter_ipopt = df_ipopt['n_eval_g'].value_counts().rename_axis('iterations').to_frame('counts')
cum_iter_ipopt = iter_ipopt.sort_values(by='iterations').cumsum()/n_probs
# cum_iter_ipopt.plot(drawstyle="steps", ax=ax, y='counts', kind = 'line', label='IPOPT', xlabel='Number of constraint evaluations',grid=True, linestyle='dotted')
plt.step(np.array(cum_iter_ipopt.index), cum_iter_ipopt['counts'], label='IPOPT - optimal solution', linestyle='dotted')

iter_fpsqp = df_fpsqp['slack_zero_n_eval_g'].value_counts().rename_axis('iterations').to_frame('counts')
cum_iter_fpsqp = iter_fpsqp.sort_values(by='iterations').cumsum()/n_probs
# cum_iter_ipopt.plot(drawstyle="steps", ax=ax, y='counts', kind = 'line', label='IPOPT', xlabel='Number of constraint evaluations',grid=True, linestyle='dotted')
plt.step(np.array(cum_iter_fpsqp.index), cum_iter_fpsqp['counts'], label='FSLP - feasible solution', linestyle='dashed')

iter_fpsqp_andersonM1 = df_fpsqp_anderson_m1['n_eval_g'].value_counts().rename_axis('iterations').to_frame('counts')
cum_iter_fpsqp = iter_fpsqp_andersonM1.sort_values(by='iterations').cumsum()/n_probs
# ax = cum_iter_fpsqp.plot(drawstyle="steps", figsize=(5,3), y='counts', kind = 'line', xlabel='Number of constraint evaluations', ylabel='ratio of solved problems', label='FSLP', grid=True, linestyle='solid')
plt.step(np.array(cum_iter_fpsqp.index), cum_iter_fpsqp['counts'], label='FSLP Am1 - optimal solution', linestyle='solid')

iter_fpsqp_andersonM1 = df_fpsqp_anderson_m1['slack_zero_n_eval_g'].value_counts().rename_axis('iterations').to_frame('counts')
cum_iter_fpsqp = iter_fpsqp_andersonM1.sort_values(by='iterations').cumsum()/n_probs
# cum_iter_ipopt.plot(drawstyle="steps", ax=ax, y='counts', kind = 'line', label='IPOPT', xlabel='Number of constraint evaluations',grid=True, linestyle='dotted')
plt.step(np.array(cum_iter_fpsqp.index), cum_iter_fpsqp['counts'], label='FSLP Am1 - feasible solution', linestyle='dashed')

iter_fpsqp_andersonM5 = df_fpsqp_anderson_m5['n_eval_g'].value_counts().rename_axis('iterations').to_frame('counts')
cum_iter_fpsqp = iter_fpsqp_andersonM5.sort_values(by='iterations').cumsum()/n_probs
# ax = cum_iter_fpsqp.plot(drawstyle="steps", figsize=(5,3), y='counts', kind = 'line', xlabel='Number of constraint evaluations', ylabel='ratio of solved problems', label='FSLP', grid=True, linestyle='solid')
plt.step(np.array(cum_iter_fpsqp.index), cum_iter_fpsqp['counts'], label='FSLP Am5 - optimal solution', linestyle='solid')

iter_fpsqp_andersonM5 = df_fpsqp_anderson_m5['slack_zero_n_eval_g'].value_counts().rename_axis('iterations').to_frame('counts')
cum_iter_fpsqp = iter_fpsqp_andersonM5.sort_values(by='iterations').cumsum()/n_probs
# cum_iter_ipopt.plot(drawstyle="steps", ax=ax, y='counts', kind = 'line', label='IPOPT', xlabel='Number of constraint evaluations',grid=True, linestyle='dotted')
plt.step(np.array(cum_iter_fpsqp.index), cum_iter_fpsqp['counts'], label='FSLP Am5 - feasible solution', linestyle='dashed')

plt.legend(loc='lower right')
plt.xlabel('number of constraint evaluations')
plt.ylabel('ratio of all occurences')



plt.grid(alpha=0.5)
plt.tight_layout()
# plt.savefig(f"performance_probs_cons_plot.pdf", dpi=300, bbox_inches='tight', pad_inches=0.01)
plt.show()
