import os, re
import numpy as np
import hcp_utils as hcp
from statsmodels.stats.multitest import multipletests
from itertools import permutations, combinations
import pandas as pd
from glob import glob

from brainannlib.stats_and_metrics import fisher_r_to_z as rz, fisher_z_to_r as zr,  get_sign

root_data_dir = os.environ["ALGONAUTS_ROOT_DIR"]
acc_dir= f"{root_data_dir}/ann_brain_data/outputs"
res_dir = "/scratch-scc/users/robert.scholz2/emmy_code/results/"

movie_set_names = [f"friends-s0{i+1}" for i in range(6)] + [ "bourne", "wolf", "figures", "life"]

# Loading of the baselines?

def load_scores_v2(model_id, v=False, replace_nan=0):
    fn=f"{root_data_dir}/ann_brain_data/scores/full_brain_models.{model_id}.s1to5.*.scores.avgmodel.npy"
    files = sorted(glob(fn));
    cumm_epi, cumm_mv, subjs = [], [],[]
    for fn in files:
        subj = re.findall(r's1to5\.(.*?)\.scores\.avgmodel', fn)[0]
        data = np.load(fn, allow_pickle=1).item()
        sc1=np.array([v for k,v in data["scores_bw_epi"].items()])
        sc2=np.array([v for k,v in data["scores_bw_mvset"].items()])
        if not(replace_nan ==False):
            nan_mask1, nan_mask2 = np.isnan(sc1), np.isnan(sc2)
            sc1[nan_mask1], sc2[nan_mask2] = replace_nan, replace_nan;
            nan_locations1 = np.argwhere(nan_mask1.sum(1)>=1)
            nan_locations2 = np.argwhere(nan_mask2.sum(1)>=1)
            if len(nan_locations1+nan_locations2)>0:
                print(f"Replacing NaNs by {replace_nan} at:", subj, nan_locations1, "and", nan_locations2)

        subjs.append(subj); cumm_epi.append(sc1); cumm_mv.append(sc2);

    cumm_epi=np.array(cumm_epi);
    cumm_mv=np.array(cumm_mv);
    if v: print(subjs, cumm_epi.shape, cumm_mv.shape)
    return cumm_epi, cumm_mv, subjs



score_files = glob(f"{root_data_dir}/ann_brain_data/scores/full_brain*sub-01*.scores.avgmodel.npy")
scores_mvsets={}
scores_epi={}
subjs = None

for file in score_files:
    model_id = re.findall(r'full_brain_models\.(.*?)\.s1to5', file)[0]
    if model_id in ["vivit-fmri-cl.EMBD","SmolLM2-ext"]: continue;

    cumm_epi, cumm_mv, isubjs = load_scores_v2(model_id)
    if model_id == "Llama_3.1_8B.4L3T1000W":
        model_id = "Llama-3.1-8B.4L3T1000W"

    #if len(isubjs)<5: print("skip:",model_id); continue;
    if len(isubjs)<5: print("skip:",model_id); continue;
    scores_mvsets[model_id] = cumm_mv # (3, 10, 91282)
    scores_epi[model_id] = cumm_epi # e.g.  (3, 94, 91282)
    print(model_id, " "*(30-len(model_id)), cumm_mv.shape, cumm_epi.shape, isubjs)
    if subjs is None: subjs = isubjs;
    if not (isubjs==subjs[:len(isubjs)]): print("Subjects order disagrees")


# collect the scores for each model in a single dataframe
scores_df_all=[]
for model, score in scores_mvsets.items():
    df = pd.DataFrame(score.mean(-1), columns=movie_set_names)
    df['subject'] = subjs[:len(score)]
    df = df.melt(id_vars='subject', var_name='scored_data', value_name='score')
    df["model"] = model
    scores_df_all.append(df)

scores_df_all = pd.concat(scores_df_all, ignore_index=True)

# add information about the model modality
modality_prefixes = {
    "audio": ["whisper", "AST", "BEATs"],
    "visual": ["dino", "slow_r50", "vivit", "resnet50", "CLIPViT", "vjep"],
    "language": ["Llama", "SmolLM", "Qwen"], 
}

def get_modality(model_id):
    for mod, prefixes in modality_prefixes.items():
        if any(model_id.lower().startswith(p.lower()) for p in prefixes):
            return mod
    return None

scores_df_all["modality"] = scores_df_all["model"].apply(get_modality)
scores_df_all['modality'] = pd.Categorical(scores_df_all['modality'], categories=["language", "audio", "visual"], ordered=True)

test_data = ['friends-s06', 'figures',  'wolf', 'bourne', 'life']
scores_df_all["data_type"] = np.where(scores_df_all["scored_data"].isin(test_data), "test", "train")
scores_df_all["pretrained"] = np.where(scores_df_all["model"].str.endswith("untr"), False, True)
scores_df_all = scores_df_all.sort_values(['modality','model', 'subject'])

# average across subjects
def highlight_max(s):
    if s.name in ["model", "modality"]: return ['' for v in s]
    return ['font-weight: bold' if v == s.max() else '' for v in s]

scores_table_savg = scores_df_all.pivot_table(index='model', columns='scored_data', values='score', aggfunc='mean')
count = scores_df_all.pivot_table(index='model', columns='scored_data', values='score', aggfunc='count')
scores_table_savg['count'] = count.sum(axis=1) / len(movie_set_names) # Sum of counts across all tests for each model
scores_table_savg = scores_table_savg.reset_index().round(3)
scores_table_savg["modality"] = scores_table_savg["model"].apply(get_modality)

scores_table_savg = scores_table_savg.sort_values(by="figures", ascending=False)
column_order = ['model', 'modality', 'count'] + movie_set_names
scores_table_savg = scores_table_savg[column_order]
scores_table_savg.style.apply(highlight_max, axis=0).format(precision=4) 


# saving scores_table_savg



# loading of the baselines
fn=f"{acc_dir}/retest_bw+pca_baseline.npy"
baselines_dict_bw = np.load(fn, allow_pickle=1).item()["scores_bw"]
print("baselines_bw:", baselines_dict_bw.keys(), np.array(baselines_dict_bw["figures"]).shape)

# saving all results in a file for quicker loading
payload=dict(scores_df_all=scores_df_all, scores_mvsets=scores_mvsets, movie_set_names=movie_set_names, scores_table_savg=scores_table_savg, baselines_dict_bw=baselines_dict_bw)
data_fn=res_dir+"/overall_results.v3.npy"
np.save(data_fn, payload)


####################################################################################
# Significance values

#import os
#from statsmodels.stats.multitest import multipletests
from brainannlib.stats_and_metrics import fisher_r_to_z
from matplotlib import pyplot as plt

#root_data_dir = os.environ["ALGONAUTS_ROOT_DIR"]
featred_dir = f"{root_data_dir}/ann_brain_data/featred"

#best_fsets = ["Llama-3.1-8B.4L3T1000W", "whisper-small.4L18T29S", "slow_r50.LASTPOOL"]
best_fsets = ["Llama-3.1-8B.4L3T1000W", "whisper-small.4L18T29S", "slow_r50.EMBD"]
all_fsets = best_fsets+["Llama-3.1-8B.4L3T1000W+untr", "whisper-small.4L18T29S+untr", "slow_r50.EMBD+untr", "whisper-large-v3.4L18T29S"]

subjs=["sub-01", "sub-03", "sub-05"]
sign_threshold=0.05


def load_parcel_scores(feature_set, sub):
    #score_fn = f"{root_data_dir}/ann_brain_data/scores/full_brain_models.{feature_set}.s1to5.{sub}.scores.1000_parcel_validation.npy"
    sign_fn = f"{root_data_dir}/ann_brain_data/scores/full_brain_models.{feature_set}.s1to5.{sub}.significance.avgmodel.npy"
    data = np.load(sign_fn, allow_pickle=1).item()
    # dict(permuted_scores=permuted_scores, pvals_corrected=pvals_corrected, pvals=pvals)
    return data

perm_scores = {}; real_score={}
for model_id in all_fsets:
    d=np.array([load_parcel_scores(model_id, sub)["permuted_scores"] for sub in subjs])
    print(d.shape)
    perm_scores[model_id]=d;
    real_score[model_id]=np.array([load_parcel_scores(model_id, sub)["bwscore"] for sub in subjs])
    print(real_score[model_id].shape)



imgs=[]; titles=[]

pvalsd = {}

for model_id in all_fsets:
    pvalsd[model_id]={}
    for s, sub in enumerate(subjs):
        perms= perm_scores[model_id][s]
        real = real_score[model_id][s]

        pvals = (np.sum(np.abs(perms) >= np.abs(real), axis=0) + 1) / (perms.shape[0] + 1)
        _, pvals_corrected, _, _ = multipletests(pvals, alpha=sign_threshold, method='fdr_bh')
        pvalsd[model_id][s] = (real, pvals, pvals_corrected)
        real = real.copy()
        real[pvals_corrected>=sign_threshold] = -0.01
        #cmap1, vmin, vmax = shifted_cmap(cm.wildfire, data=real, greymidpoint=False)
        print(model_id, sub, np.nanmin(real).round(2), np.nanmax(real).round(2))

        if not(model_id in best_fsets): continue;
        cmap = plt.get_cmap("plasma").copy()
        cmap.set_under((0.8,0.8,0.8, 1)) # make everything below zero gray

        fig = plot_flatmap(real, mtype="HCP_S1200", data_style="fullHCP", cmap=cmap, vmin=0, vmax=0.75);
        imgs.append(fig2pil(fig)); plt.close(fig);
        titles.append(f"{model_id} for {sub}")

axs = np.array(plt.subplots(3,3, figsize=(16,10))[1]).flatten()
for ax in axs: ax.axis("off");
for i, im in enumerate(imgs): 
    axs[i].imshow(im);  axs[i].set_title(titles[i]);
plt.tight_layout();

res_img_dir = "/scratch-scc/users/robert.scholz2/emmy_code/results/img"
img_fn= res_img_dir + f'/sign_corr_best_models_all_subjs_figures_{sign_threshold}.png'
plt.gcf().savefig(img_fn, format='png', dpi=300)
plt.close();

sign_fn = res_dir+f"/sign_best_models_all_subjs_figures_ev10thTR_5000perms_p{sign_threshold}.npy"
np.save(sign_fn, pvalsd)


modality_pairs =[
    ("Llama-3.1-8B.4L3T1000W", "Llama-3.1-8B.4L3T1000W+untr"),
    ("whisper-small.4L18T29S", "whisper-small.4L18T29S+untr"),
    ("slow_r50.EMBD", "slow_r50.EMBD+untr"),
]

pvalsd = {}
read_diff_d = {}

imgs=[]; titles=[]
for mod1, mod2 in tqdm(modality_pairs):
    name=mod1+"-"+mod2
    pvalsd[name]={}; read_diff_d[name]={}
    for s, sub in enumerate(subjs):

        pscore1=perm_scores[mod1][s] 
        pscore2=perm_scores[mod2][s]

        real_diff_z = fisher_r_to_z(real_score[mod1][s])-fisher_r_to_z(real_score[mod2][s])
        read_diff_d[name][s]=real_diff_z

        perm_diff = fisher_r_to_z(pscore1)-fisher_r_to_z(pscore2)
        pvals = (np.sum(np.abs(perm_diff) >= np.abs(real_diff_z), axis=0) + 1) / (perm_diff.shape[0] + 1)
        _, pvals_corrected, _, _ = multipletests(pvals, alpha=sign_threshold, method='fdr_bh')
        pvalsd[name][s] = (real_diff_z, pvals, pvals_corrected)

sign_fn = res_dir+f"/sign_best_model_pairs_all_subjs_figures_ev10thTR_5000perms_p{sign_threshold}.npy"
np.save(sign_fn, pvalsd)