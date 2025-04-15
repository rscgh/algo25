
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV  #Ridge, LinearRegression
from sklearn.pipeline import make_pipeline

from brainannlib.stats_and_metrics import make_kfolds

from brainannlib.stats_and_metrics import corr_score

def delay_by_hrf(X, delays=None):
    if (delays is None) or (len(delays)==0):
        return X;
    n_samples, n_features = X.shape
    X_delayed = np.zeros((n_samples, n_features * len(delays)), dtype=X.dtype)
    for idx, delay in enumerate(delays):
        beg, end = idx * n_features, (idx + 1) * n_features
        if delay == 0:
            X_delayed[:, beg:end] = X
        elif delay > 0:
            X_delayed[delay:, beg:end] = X[:-delay]
        elif delay < 0:
            X_delayed[:-abs(delay), beg:end] = X[abs(delay):]

    return X_delayed

def get_regression_pipeline(mode="RidgeCV", delays=None, alpha_values="default"):
    
    scaler = StandardScaler(with_mean=True, with_std=False)
    
    regression = None;
    if mode == "RidgeCV":
      if alpha_values == "default": 
        alpha_values = np.logspace(-1,5,7).tolist()
      #regression = RidgeCV(alphas=alpha_values, store_cv_values = True,scoring = 'explained_variance')
      regression = RidgeCV(alphas=alpha_values, store_cv_results = True, scoring = 'explained_variance')

    if delays is None:
        return make_pipeline(scaler, regression,)
    
    from voxelwise_tutorials.delayer import Delayer
    delayer = Delayer(delays=delays)
    return make_pipeline(scaler, delayer, regression,)



def run_cv_predictions_v2(ann_data, brain_data, k=5, perm_type="blocks", pipeline = None, hrf_delays = None, v=False, score_fn=None):
    
    if pipeline is None: 
        # alphas = np.concatenate((np.linspace(50000, 120000, 30).round(), np.logspace(-1,6,8)))
        pipeline = get_regression_pipeline("RidgeCV") #, alphas = alphas
    if score_fn is None:
        score_fn = corr_score
    
    pred_brain_data = np.zeros_like(brain_data)
    if v: 
        print("Regression input data:", ann_data.shape, brain_data.shape)
        print("Regression pipeline:", get_regression_pipeline)
        print("Other:", dict(k=k, perm_type=perm_type, delays=hrf_delays, score_fn=score_fn))
    
    
    delayed_ann_data = delay_by_hrf(ann_data, delays=hrf_delays)
    
    n_samples = ann_data.shape[0]
    folds = make_kfolds(n_samples, k=k, perm_type=perm_type)
    
    n_targets = brain_data.shape[1]
    scores = np.zeros((len(folds), n_targets));
    alphas = []
    
    for f, (train, test) in enumerate(folds):

        x_train, x_test = delayed_ann_data[train], delayed_ann_data[test]   
        y_train, y_test = brain_data[train], brain_data[test]

        _ = pipeline.fit(x_train, y_train)
        y_test_pred = pipeline.predict(x_test)
        cs = score_fn(y_test, y_test_pred)
        pred_brain_data[test,: ] = y_test_pred
        scores[f,:] = cs
        if "alpha_" in dir(pipeline[-1]):
            alphas.append(pipeline[-1].alpha_)
        else:
            attrs= [x for x in dir(pipeline[-1]) if "alpha" in x]
            a = getattr(pipe[-1], attrs[0]) if len(attrs)>1 else 0;
            alphas.append(a)
        
        if v: 
            print(f"Scores for fold {f}:", cs.mean().round(3), cs[:10].round(2))
    
    mean_scores_across_folds = scores.mean(0);
    posthoc_scores = score_fn(brain_data, pred_brain_data) 
    
    if v: 
        print("-----")
        print("Resulting Scores of shape:", scores.shape)
        print("-----")
        print("Mean scores across folds:")
        print(mean_scores_across_folds.mean().round(3), mean_scores_across_folds[:10].round(2))
        print("-----")
        print("Best alphas:")
        print(np.array([np.median(a) for a in alphas]))
        print("-----")
        print("Scores from after stitching together test-set predictions:")
        print(posthoc_scores.mean().round(3), posthoc_scores[:10].round(2))
    
    return pred_brain_data, scores, mean_scores_across_folds, posthoc_scores, alphas;

