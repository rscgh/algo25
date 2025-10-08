

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