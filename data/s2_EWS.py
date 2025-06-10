import numpy as np
import pandas as pd

from scipy.stats import mannwhitneyu
from tqdm import tqdm
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from joblib import Parallel, delayed


import s0_subroutines as s0

#---------------------------------------------------#
#------------------ FEATURE EXTRACTION -------------#
#---------------------------------------------------#

def featurize(df, feat_settings):
    """
    Extracts a subset of features from the input DataFrame using tsfresh.

    - Uses `extract_features` with the provided settings to compute a wide variety
      of time‐series features for each 'id' group, sorted by 'time'.
    - Removes any initial non‐signal columns (sometimes tsfresh emits class‐label
      columns prefixed 'ts') by selecting only the real features.

    Parameters:
    -----------
    df : pd.DataFrame
        Must contain columns ['id','time','ts','label'] (TSFRESH input format).
    feat_settings : dict or ComprehensiveFCParameters
        Configuration specifying which features to compute.

    Returns:
    --------
    pd.DataFrame
        The extracted feature matrix, one row per 'id'.
    """
    # Run tsfresh feature extraction, filling missing values by imputation
    raw_feats = extract_features(
        df,
        column_id='id',
        column_sort='time',
        default_fc_parameters=feat_settings,
        impute_function=impute
    )

    # Determine which columns are real features vs. ts-prefixed label columns
    first_prefix = raw_feats.columns[0].split('_')[0]
    if first_prefix == 'ts':
        # Drop the initial 'ts...' columns (they’re not real features)
        subset_idx = np.arange(830, raw_feats.shape[1])
    else:
        # Otherwise assume the first 830 are real and drop beyond that
        subset_idx = np.arange(830)

    return raw_feats.iloc[:, subset_idx].copy()


#---------------------------------------------------#
#------------------ EWS ANALYSIS -------------------#
#---------------------------------------------------#


def EWS_analysis(d_preDF, Nsurr, feat_settings):
    """
    Perform Early Warning Signals (EWS) analysis by:
      1. Generating Nsurr surrogate time‐series (random shuffle of entire ts).
      2. Extracting features in parallel for real and surrogate datasets.

    Parameters:
    -----------
    d_preDF : pd.DataFrame
        Pre‐DF TSFRESH‐formatted table with ['id','time','ts','label'].
    Nsurr : int
        Number of surrogate datasets to generate.
    feat_settings : dict or ComprehensiveFCParameters
        Configuration for tsfresh feature extraction.

    Returns:
    --------
    d_features : pd.DataFrame
        Concatenated feature set for each real window id.
    l_features_surr : list of pd.DataFrame
        List of feature‐DataFrames, one per surrogate run.
    """
    # Unique window IDs in the pre‐DF table
    a_ids = np.unique(d_preDF.id)
    T     = d_preDF.shape[0]

    # Build surrogate columns ts1, ts2, ..., tsNsurr by shuffling the entire series
    d_surr = d_preDF.copy()
    for ns in range(Nsurr):
        perm = np.random.permutation(T)
        d_surr[f'ts{ns+1}'] = d_surr.ts.values[perm]

    # Helper to featurize a single real window
    def _featurize_real(ID):
        df_win = d_preDF[d_preDF.id == ID]
        return featurize(df_win, feat_settings)

    # Extract features for the real data in parallel
    d_features = pd.concat(
        Parallel(n_jobs=-1)(
            delayed(_featurize_real)(ID) for ID in a_ids
        ),
        axis=0
    )

    # Helper to featurize one surrogate dataset
    def _featurize_surr(ns):
        df_s = d_surr.copy()
        # Overwrite ts column with the shuffled one
        df_s.ts = df_s[f'ts{ns+1}']
        # Only keep the minimal required columns for featurization
        df_s = df_s[['id','time','ts','label']]

        # Extract for each window ID
        return pd.concat([
            featurize(df_s[df_s.id == ID], feat_settings)
            for ID in a_ids
        ], axis=0)

    # Extract features for each surrogate in parallel
    l_features_surr = Parallel(n_jobs=-1)(
        delayed(_featurize_surr)(ns) for ns in range(Nsurr)
    )

    return d_features, l_features_surr


#---------------------------------------------------#
#------------------ MW ANALYSIS --------------------#
#---------------------------------------------------#

def MW_analysis(real_features, surr_features, Wmw, Nrndm):
    """
    Compute Mann–Whitney U–based similarity between each feature's sliding-window
    distribution and randomly sampled windows, for both real and surrogate data.

    Returns:
    --------
    a_tright : np.ndarray
        Array of right‐hand window start indices.
    a_mw : np.ndarray
        Shape (Nfeat, Nshift) of MW‐U statistics for real features.
    a_mw_rndm : np.ndarray
        Shape (Nsurr, Nfeat, Nshift) of MW‐U stats for surrogate features.
    """
    Nfeat = real_features.shape[1]
    Tmax  = real_features.shape[0]

    # Define left/right windows around the midpoint
    a_tleft  = np.arange(Tmax//2, Tmax - Wmw)
    a_tright = np.arange(Tmax//2 + Wmw, Tmax)
    Nshift   = a_tleft.size
    Nsurr    = len(surr_features)

    # Compute MW-U for one real feature index k
    def _mw_real(k):
        series = real_features.iloc[:, k].values
        out    = np.zeros(Nshift)
        for idx, t0 in enumerate(a_tleft):
            ref = series[t0:t0+Wmw]
            # Sample Nrndm windows from the first half
            rnd_idx = np.random.choice(np.arange(Tmax//2 - Wmw), Nrndm, replace=False)
            samp1   = np.array([series[r:r+Wmw] for r in rnd_idx])
            # Compute U-statistics and normalize
            uvals   = np.array([mannwhitneyu(samp1[j], ref)[0] for j in range(Nrndm)]) / (Wmw**2)
            out[idx] = np.mean(2*np.abs(uvals - 0.5))
        return out

    # Parallel real-data MW
    a_mw = np.array(Parallel(n_jobs=-1)(
        delayed(_mw_real)(k) for k in tqdm(range(Nfeat))
    ))

    # Compute MW for each surrogate ns, feature k
    def _mw_surr(ns, k):
        series = surr_features[ns].iloc[:, k].values
        out    = np.zeros(Nshift)
        for idx, t0 in enumerate(a_tleft):
            ref = series[t0:t0+Wmw]
            rnd_idx = np.random.choice(np.arange(Tmax//2 - Wmw), Nrndm, replace=False)
            samp1   = np.array([series[r:r+Wmw] for r in rnd_idx])
            uvals   = np.array([mannwhitneyu(samp1[j], ref)[0] for j in range(Nrndm)]) / (Wmw**2)
            out[idx] = np.mean(2*np.abs(uvals - 0.5))
        return out

    # Flatten surrogate loops (nsurr × feat) then reshape
    flat = Parallel(n_jobs=-1)(
        delayed(_mw_surr)(ns, k)
        for ns in tqdm(range(Nsurr), desc="Surr loops")
        for k in tqdm(range(Nfeat), desc="Feat loops", leave=False)
    )
    a_mw_rndm = np.array(flat).reshape(Nsurr, Nfeat, Nshift)

    return a_tright, a_mw, a_mw_rndm


#---------------------------------------------------#
#------------------ EWS SPLITTING ------------------#
#---------------------------------------------------#

def EWS_split(mw, mw_surr, time, alpha, athresh):
    """
    Compare real vs. surrogate MW-U curves to detect the earliest regime shift
    (EWS) where a feature’s U-statistic significantly exceeds surrogate-based
    confidence.

    Returns:
    --------
    np.ndarray of shift times (subset of `time`) where EWS detected.
    """
    Nsurr, Nfeat, Nshift = mw_surr.shape

    # Global surrogate‐based confidence envelope at level alpha
    a_conf = np.quantile(mw_surr.reshape(Nsurr*Nfeat, Nshift), alpha, axis=0)

    shifts = []
    for k in tqdm(range(Nfeat), desc="Feature EWS"):
        feat_curve = mw[k, :]
        above      = np.where(feat_curve - a_conf > 0)[0]
        if above.size > 1:
            # run‐length encode gaps in the significant‐index array
            rle = s0.rle(np.diff(above))
            # require a final run of at least athresh consecutive points
            if rle[-1,0] == 1 and rle[-1,1] > athresh:
                # pick the earliest index of that run
                idx = above[-rle[-1,1]] - 1
                shifts.append(time[idx])

    return np.array(shifts)
