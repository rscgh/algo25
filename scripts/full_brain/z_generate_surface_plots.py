
import numpy as np
import pickle as pk
import pandas as pd
from matplotlib import pyplot as plt

from brainspace.utils.parcellation  import map_to_labels, reduce_by_labels
import hcp_utils as hcp
from surfplot import Plot
from neuromaps.datasets import fetch_fslr
import numpy as np

from brainspace.datasets  import load_parcellation
from matplotlib.colors import ListedColormap
from brainannlib.visualization import shifted_cmap
from brainannlib.stats_and_metrics import fisher_r_to_z as rz, fisher_z_to_r as zr,  get_sign


root = r"C:\tmp\OwnCloud\docs\neuroconnlab\\ann_model\git\\"


# Loading the overall results data from the Ridge/OLS models
data_fn = r"C:\research\neuroconnlab\ann_model\git\results\overall_results.v3.npy"
payload = np.load(data_fn, allow_pickle=1).item()
print(payload.keys())

scores_mvsets= payload["scores_mvsets"]
movie_set_names= payload["movie_set_names"]
scores_table_savg = payload["scores_table_savg"]
baselines_dict_bw=payload["baselines_dict_bw"]
best_models = ["Llama-3.1-8B.4L3T1000W", "whisper-small.4L18T29S", "slow_r50.EMBD"]
best_model_scores_table=scores_table_savg[scores_table_savg["model"].isin(best_models)]

# load the spin-permutation results
sign_fn = r"C:\research\neuroconnlab\ann_model\git\results\sign_best_models_all_subjs_figures_ev10thTR_5000perms_p0.05.npy"
sign_dict = np.load(sign_fn, allow_pickle=1).item()
#print(sign_dict.keys(),sign_dict['whisper-small.4L18T29S'].keys())
#real_corr, p, p_fdr = sign_dict['whisper-small.4L18T29S'][1]
#print(real_corr.shape, p.shape)

# load the paired spin-permutation results (i.e. representations from pre-trained vs randomly initilized models)
sign_fn = r"C:\research\neuroconnlab\ann_model\git\results\sign_best_model_pairs_all_subjs_figures_ev10thTR_5000perms_p0.05.npy"
paired_sign_dict = np.load(sign_fn, allow_pickle=1).item()
print(paired_sign_dict.keys())

###############################################################
# Prepare plotting on the surfaces

surfaces = fetch_fslr()
lh, rh = surfaces['inflated']
#lh, rh = surfaces['inflated']

import io, PIL
def fig2pil(fig, pad_inches=0.02):
  buf = io.BytesIO()
  fig.savefig(buf, bbox_inches="tight", pad_inches=pad_inches);
  buf.seek(0)
  return PIL.Image.open(buf)

def show_pdata(data, ld = None, is_32k=False, cmap="coolwarm", mw_val=0, title=None, layout="grid", zoom=1.45,pltkwargs={}, **kwargs):
  out = data if ld is None else map_to_labels(data, ld, mask=ld!=mw_val, fill=0)

  if out.shape[0] in [29696]:
    lcd = hcp.left_cortex_data(out)
    plkwargs = dict(size=(800, 300))
    plkwargs.update(pltkwargs)
    p = Plot(surf_lh=lh, **plkwargs) 
    pkwargs = dict(cmap=cmap, cbar=True);
    pkwargs.update(kwargs)
    p.add_layer({'left': lcd},  **pkwargs)

  if out.shape[0] in [91282, 59412]:
    lcd = hcp.left_cortex_data(out)
    rcd = hcp.right_cortex_data(out)
    plkwargs = dict(size=(1400, 300), zoom=zoom, layout=layout)
    plkwargs.update(pltkwargs)
    p = Plot(surf_lh=lh, surf_rh=rh, **plkwargs) 
    pkwargs = dict(cmap=cmap, cbar=True);
    pkwargs.update(kwargs)
    p.add_layer({'left': lcd, 'right': rcd},  **pkwargs)

  fig = p.build()
  if not(title is None): fig.axes[0].set_title(title)
  return fig

###############################################################
# Prepare thresholding by significance across subjects

from statsmodels.stats.multitest import multipletests
from scipy.stats import combine_pvalues

def get_sign(model_id, sign_dict, sign_threshold=0.001):
    if not(model_id in sign_dict.keys()): 
        print("No significance info for", model_id)
        return None, None, slice(0,0)
    p_uncorr = np.array([v[-2] for k,v in sign_dict[model_id].items()]) # accumulate all subjects
    p_uncorr = np.apply_along_axis(lambda p: combine_pvalues(p, method='stouffer')[1], 0, p_uncorr)
    _, p_corr, _, _ = multipletests(p_uncorr, alpha=sign_threshold, method='fdr_bh')
    
    mask = p_corr>sign_threshold
    return p_uncorr, p_corr, mask

###############################################################
# Figure 1 - Overall results

mv_idx = movie_set_names.index("figures")
sign_threshold=.001

##-------------------------------------------------------------
# Figure 1A - Average scores per model

excluded_models=["slow_r50.LASTPOOL", "slow_r50.EmbdPCAonAll", "llama-3-8b-bnb-4bit-contpretr-lora-friends-0.1.4L7T1000W",
                "Llama-3.1-8B.4L7T1000W", "Llama-3.1-8B.4L10T1000W", "Qwen2.5-7B.4L7T1000W+untr", "Llama-3.1-8B.4L7T1000W+untr"]

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



##-------------------------------------------------------------
# Figure 1 C-E: Brain surfaces (absolute scores)

cmap_inferno = plt.get_cmap("inferno").copy()
cmap_inferno.set_under((0.8,0.8,0.8, 1)) # make everything below zero gray

for i, (_,row) in enumerate(best_model_scores_table.iterrows()):
    scores = scores_mvsets[row["model"]][:,mv_idx,:].copy()
    ref_data = scores.mean(0)
    fig=show_pdata(ref_data, cmap=cmap_inferno, ld = None, cbar = False, color_range=(0, 0.85), zoom=1.45, title=row["model"]);

baseline_figures = np.array(baselines_dict_bw["figures"]) # [3, n_vertices] 

for i, (_,row) in enumerate(best_model_scores_table.iterrows()):
    scores = scores_mvsets[row["model"]][:,mv_idx,:].copy()
    scores = scores.mean(0)[:59412]

    # mask out non-significant vertices
    _, p_corr, mask = get_sign(row["model"], sign_dict, sign_threshold=sign_threshold)
    scores[mask] = -0.001
    print(row["model"], scores.max().round(3))

    fig=show_pdata(scores, cmap=cmap_inferno, ld = None, color_range=(0.001,0.769),\
                   cbar = True,zoom=1.45, layout="row", title=row["model"]);


##-------------------------------------------------------------
# Figure 1 F-H: Brain surfaces (baseline normalized scores)

import cmasher as cm
cmap_ccnorm = cm.chroma#plt.get_cmap("cividis").copy()
colors = cmap_ccnorm(np.linspace(0, 0.85, cmap_ccnorm.N))
gv=0.75
colors[0] = (gv,gv,gv, 1)#*100+ list(colors)  # RGBA for grey
cmap_ccnorm = ListedColormap(colors)
cmap_ccnorm.set_bad((gv,gv,gv, 1))
cmap_ccnorm.set_bad((1,1,1, 1))

baseline_figures = np.array(baselines_dict_bw["figures"]) # [3, n_vertices] 

for i, (_,row) in enumerate(best_model_scores_table.iterrows()):
    scores = scores_mvsets[row["model"]][:,mv_idx,:].copy()
    norm_scores = (scores/ np.sqrt(baseline_figures)).copy()

    ref_data = scores.mean(0)
    mean_norm_scores = norm_scores.mean(0)[:59412]

    _, p_corr, mask = get_sign(row["model"], sign_dict, sign_threshold=sign_threshold)
    mean_norm_scores[mask] = -0.001
    print(row["model"], np.nanmax(mean_norm_scores).round(3), np.nanpercentile(mean_norm_scores, 99))
    
    fig=show_pdata(mean_norm_scores, cmap=cmap_ccnorm, ld = None, cbar = True, color_range=(0, .85), zoom=1.45, title=row["model"]);
    fig.set_dpi(250); plt.show()


###############################################################
# Figure 3 - Trained vs untrained

trained_models = ["Llama-3.1-8B.4L3T1000W", "whisper-small.4L18T29S", "slow_r50.EMBD"]

s2 = movie_set_names.index("figures")

for trained_mod in trained_models:
    untrained_mod= trained_mod+"+untr"
    
    scores1 = scores_mvsets[trained_mod].mean(0)
    scores2 = scores_mvsets[untrained_mod].mean(0)
    z_difference = (rz(scores1)-rz(scores2))[s2].copy()[:59412]
    
    cmap1, vmin, vmax = shifted_cmap(cm.wildfire, data=z_difference, greymidpoint=False)
    print(trained_mod,"vs untrained", vmin.round(2), vmax.round(2));
    
    fig=show_pdata(z_difference, cmap=cmap1, ld = None, cbar = True, color_range=(vmin, vmax), zoom=1.45, title=trained_mod);
    plt.show()

    name = trained_mod+"-"+untrained_mod
    _, p_corr, mask = get_sign(name,paired_sign_dict, sign_threshold=sign_threshold)
    if p_corr is None: continue;
    z_difference[p_corr>0.01] = 0.001
    
    cmap1, vmin, vmax = shifted_cmap(cm.wildfire, data=z_difference, greymidpoint=True, grey=.75)
    fig=show_pdata(z_difference, cmap=cmap1, ld = None, cbar = True, color_range=(vmin, vmax), zoom=1.45, title=trained_mod);
    plt.show()
    

###############################################################
# Figure 4 - (Full) Stacked model

var_deriv_fn = r"C:\research\neuroconnlab\ann_model\git\results\/varpart_all_stacked_on_s06_allsubjs_figures_ev10thTR.v2.R2.optim_and_score_on_figures.deriv.npy"
p = np.load(var_deriv_fn, allow_pickle=1).item()
print(p.keys())

full= p["full"]                                 # full stacked predictions R2
rfi= p["rfi"]                                   # room for improvement
percentage_unimodal=p["percentage_unimodal"]    # percentage uniquely unimodal


vmin = np.percentile(full.mean(0),1)
vmax = np.percentile(full.mean(0),99)
title="Full stacked model predictions"
fig=show_pdata(full.mean(0), cmap="inferno", cbar = True, color_range=(vmin, vmax), title=title);


###############################################################
# Figure 5 - Room for improvement


#define colormaps for plotting R2
import cmasher as cm
from matplotlib.colors import LinearSegmentedColormap

colors = cm.chroma(np.linspace(0, 0.8, 256))
cmap_percR2 = LinearSegmentedColormap.from_list("truncated", colors)

vmin = np.percentile(rfi.mean(0),1)
vmax = np.percentile(rfi.mean(0),99)
title="Room for improvement"
fig=show_pdata(rfi.mean(0), cmap=cmap_percR2, cbar = True, color_range=(vmin, vmax), title=title);