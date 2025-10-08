

import os, re
import numpy as np
import hcp_utils as hcp
from statsmodels.stats.multitest import multipletests
from itertools import permutations, combinations
from matplotlib import pyplot as plt
import cmasher as cm
import pandas as pd
import seaborn as sns
from glob import glob
from PIL import Image

from brainannlib.stats_and_metrics import ttest_batch_test, fisher_r_to_z as rz, fisher_z_to_r as zr,  get_sign
from brainannlib.visualization import plot_flatmap, plot_surface, shifted_cmap, fig2pil, cortex, add_rois


root_data_dir = os.environ["ALGONAUTS_ROOT_DIR"]
acc_dir= f"{root_data_dir}/ann_brain_data/outputs"
mri_dir = f"{root_data_dir}/ann_brain_data/mri";

res_img_dir = "/scratch-scc/users/robert.scholz2/emmy_code/results/img"
res_dir = "/scratch-scc/users/robert.scholz2/emmy_code/results/"

cmaps_dir= "/mnt/vast-standard/home/robert.scholz2/u14262/micromamba/envs/algoenv/share/pycortex/colormaps"
cortex.options.config.set("webgl", "colormaps", cmaps_dir)


sign_fn = res_dir+f"/sign_best_models_all_subjs_figures_ev10thTR_5000perms_p0.05.npy"
sign_dict = np.load(sign_fn, allow_pickle=1).item()
print(sign_dict.keys(),sign_dict['whisper-small.4L18T29S'].keys())
real_corr, p, p_fdr = sign_dict['whisper-small.4L18T29S'][1]
print(real_corr.shape, p.shape)

sign_fn = res_dir+f"/sign_best_model_pairs_all_subjs_figures_ev10thTR_5000perms_p0.05.npy"
paired_sign_dict = np.load(sign_fn, allow_pickle=1).item()
print(paired_sign_dict.keys())


###############################################################
# Figure 1 - Overview Point/Stripplot


scores_df_all["model_display_name"] = [".".join(s.split(".")[:-1]) for s in scores_df_all["model"]]

selected_data = scores_df_all[
        (scores_df_all['scored_data'] == "figures") & 
        ~(scores_df_all["model"].isin(excluded_models))
]

import seaborn as sns
colors = sns.color_palette("tab10")[:3]
palette = dict(visual=colors[0], audio=colors[2], language=colors[1])

plt.figure(figsize=(5,3))
sns.stripplot(data=selected_data[selected_data["pretrained"]], x="model_display_name", y="score", hue="modality",
        dodge=False, jitter=True, marker="o", alpha=1, legend=False, palette=palette);

sns.stripplot(data=selected_data[~selected_data["pretrained"]], x="model_display_name", y="score", hue="modality",
        dodge=False, jitter=True, marker="s", alpha=.3, legend=False, palette=palette);
plt.setp(plt.gca().get_xticklabels(), rotation=45, ha='right');
for y in plt.gca().get_yticks(): plt.gca().axhline(y, color='#DDD', linestyle='--', linewidth=0.5, zorder=0)
for x in [0, 6, 10]:  # example x positions
    plt.gca().axvspan(x - 0.5, x + 0.5, color='orange', alpha=0.066)
plt.gcf().set_dpi(300)




###############################################################
# Figure 1 - Correlation plot

def show_heatplot(data, xlabels=None, ylabels=None, cmap="coolwarm", title=None,figsize=(25, 10), colorbar=True, ax=None, vmin=None, vmax=None, aspect="auto"):
    #v=np.absolute(data).max()
    if ax is None: ax=plt.figure(figsize=figsize).gca()
    im=ax.matshow(data, vmin=vmin, vmax=vmax, cmap=cmap, aspect=aspect); 
    if colorbar: plt.colorbar(im, ax=ax);
    if not(ylabels is None): ax.set_yticks(np.arange(len(ylabels))); ax.set_yticklabels(ylabels)
    if not(xlabels is None): ax.set_xticks(np.arange(len(xlabels))); ax.set_xticklabels(xlabels)
    for (i, j), val in np.ndenumerate(data):
        ax.text(j, i, f'{round(val,2)}', ha='center', va='center', color='black');
    if not(title is None): ax.set_title(title);
    return ax;

preferred_sym_cmap= cm.prinsenvlag_r


selection = scores_df_all[
        (scores_df_all['scored_data'] == "figures") & 
        ~(scores_df_all["model"].isin(excluded_models))
]

selection= selection[scores_df_all.pretrained].sort_values(by=["modality"], ascending=True)

model_keys=selection["model"].unique()
model_disp_names = [model_display_name_map[k] for k in model_keys]

s2 = movie_set_names.index("figures")
explained_maps = np.array([scores_mvsets[k][:, s2].mean(0) for k in model_keys])

labels = [k[:4] for k in model_keys]
fig, axs = plt.subplots(1,1,figsize=(12,5.3), sharey=True);
plt.suptitle(f"Correlation of score maps averaged across 3 subjects");

corrs=np.corrcoef(explained_maps[:,:59412])
labels = [k[:4]+"." for k in model_keys]
show_heatplot(corrs, labels, model_disp_names, ax=axs, colorbar=True, cmap=preferred_sym_cmap, vmin=-1, vmax=1, aspect='auto');
plt.gcf().set_dpi(300)