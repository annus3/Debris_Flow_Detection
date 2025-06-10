import numpy as np
import pandas as pd

from tqdm import tqdm
from statsmodels.stats.multitest import multipletests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

import s0_subroutines as s0

def getfam(feature_name):
    """
    Extract the 'family' portion of a tsfresh feature name.

    tsfresh names features as '<family>__<function>__<parameters>'.
    This returns the family substring immediately following the leading underscore.

    Parameters:
    -----------
    feature_name : str
        Full tsfresh feature identifier.

    Returns:
    --------
    str
        The feature family (first level grouping).
    """
    # Split on '__' and return the first token (strip leading underscore if present)
    return np.char.split(feature_name, sep='__').ravel()[0][1]


def getfeat(feature_name):
    """
    Extract the 'feature' (function) portion of a tsfresh feature name.

    The last token after splitting on '__' corresponds to the extraction function.

    Parameters:
    -----------
    feature_name : str
        Full tsfresh feature identifier.

    Returns:
    --------
    str
        The specific feature function name.
    """
    return np.char.split(feature_name, sep='__').ravel()[0][-1]


#---------------------------------------------------#
#-------------- P-VALUE CALCULATION ----------------#
#---------------------------------------------------#

def get_pvals(mw, mw_surr, MT_METHOD, alpha):
    """
    Compute empirical p-values for Mann-Whitney U tests by comparing real vs. surrogate statistics,
    then apply multiple-testing correction along each time shift.

    Parameters:
    -----------
    mw        : np.ndarray, shape (Nfeat, Nshift)
                Mann-Whitney statistics for real features.
    mw_surr   : np.ndarray, shape (Nsurr, Nfeat, Nshift)
                Mann-Whitney statistics for surrogate features.
    MT_METHOD : str
                Method for multiple-testing correction (e.g., 'fdr_bh').
    alpha     : float
                Significance level for correction.

    Returns:
    --------
    a_pval_corr : np.ndarray, shape (Nfeat, Nshift)
                   Corrected p-values for each feature and shift.
    """
    Nsurr, Nfeat, Nshift = mw_surr.shape

    # Flatten surrogates to shape (Nsurr*Nfeat, Nshift) for comparison
    a_surr = mw_surr.reshape(Nsurr * Nfeat, Nshift)

    # Initialize raw p-value container
    a_pvals = np.zeros((Nfeat, Nshift))

    # For each time shift, count surrogates exceeding the real statistic
    for t in tqdm(range(Nshift), desc="Raw p-value calc"):
        # a_pvals[nf, t] = fraction of (surrogate > real) over all (Nsurr×Nfeat)
        a_pvals[:, t] = np.array([
            np.sum(a_surr[:, t] > mw[nf, t]) / (Nsurr * Nfeat)
            for nf in range(Nfeat)
        ])

    # Copy for corrected values
    a_pval_corr = a_pvals.copy()

    # Apply multiple-testing correction per column (time shift)
    for t in tqdm(range(Nshift), desc="MT correction"):
        # multipletests returns (reject, pvals_corrected, _, _)
        _, corrected, _, _ = multipletests(a_pvals[:, t], alpha=alpha, method=MT_METHOD)
        a_pval_corr[:, t] = corrected

    return a_pval_corr


#---------------------------------------------------#
#----------- SIGNIFICANT-FAMILY COUNTS -------------#
#---------------------------------------------------#

def count_signif_families(feature_alerts, featnames, famlist, mw_surr):
    """
    Build a coincidence matrix of feature pairs and count the (possibly fractional)
    number of significant feature families at each time shift.

    Each family’s contribution is weighted by the fraction of its features that fired.

    Parameters:
    -----------
    feature_alerts : np.ndarray, shape (Nfeat, Nshift)
                     Boolean mask of significance per feature/shfit.
    featnames      : np.ndarray of str, length Nfeat
                     Original feature name strings.
    famlist        : np.ndarray of str, length Nfeat
                     Family labels corresponding to each feature index.
    mw_surr        : np.ndarray, shape (Nsurr, Nfeat, Nshift)
                     Surrogate Mann-Whitney stats.

    Returns:
    --------
    probmat_feat : np.ndarray, shape (Nfeat, Nfeat)
                   Counts of co-occurrence for each feature pair.
    Nff          : np.ndarray, shape (Nshift,)
                   Fractional count of significant families per shift.
    """
    Nsurr, Nfeat, Nshift = mw_surr.shape

    # Unique families and their total feature counts
    all_fams, all_fam_sizes = np.unique(famlist, return_counts=True)
    Nfams = all_fams.size

    # Initialize co-occurrence and family-count arrays
    probmat_feat = np.zeros((Nfeat, Nfeat), dtype=int)
    Nff = np.zeros(Nshift)

    # Loop through time shifts
    for t in range(Nshift):
        # Indices of features flagged as significant at shift t
        sig_idx = np.where(feature_alerts[:, t])[0]
        Lc = sig_idx.size

        # Update co-occurrence matrix: increment for every feature pair
        if Lc > 0:
            mat_i, mat_j = np.meshgrid(sig_idx, sig_idx)
            probmat_feat[mat_i, mat_j] += 1

            # Count fractional family significance
            fams, counts = np.unique(famlist[sig_idx], return_counts=True)
            fam_fractions = np.zeros(Nfams)

            for fid, fam in enumerate(all_fams):
                # fraction = (#significant in this family) / (total in family)
                idx = np.where(fams == fam)[0]
                if idx.size:
                    fam_fractions[fid] = counts[idx[0]] / all_fam_sizes[fid]

            # Sum fractional contributions
            Nff[t] = fam_fractions.sum()

    return probmat_feat, Nff


#---------------------------------------------------#
#-------------- BINOMIAL SIMULATIONS ---------------#
#---------------------------------------------------#

def generate_sequences(N, T, p):
    """
    Create N binary sequences of length T with probability p of 1’s.
    Used to simulate random-significance patterns.

    Returns:
    --------
    np.ndarray of shape (N, T)
    """
    return np.random.binomial(1, p, size=(N, T))


def simulate_coincidence(sequences, coincidence_matrix):
    """
    Impose pairwise coincidences on binary sequences according to
    a pre-computed coincidence probability matrix.

    Parameters:
    -----------
    sequences          : np.ndarray(N, T)
                         Original binary indicator matrix.
    coincidence_matrix : np.ndarray(N, N)
                         Pairwise coincidence probabilities.

    Returns:
    --------
    adjusted_sequences : np.ndarray(N, T)
    """
    adjusted = sequences.copy()
    N, T = sequences.shape

    # For each feature pair, swap bits with probability given
    for i in range(N):
        for j in range(i + 1, N):
            prob = coincidence_matrix[i, j]
            if prob > 0:
                mask = np.random.rand(T) < prob
                adjusted[i, mask] = sequences[j, mask]
                adjusted[j, mask] = sequences[i, mask]

    return adjusted


#---------------------------------------------------#
#------------- CONSECUTIVE WARNINGS ---------------#
#---------------------------------------------------#

def consec_warnings(binary_seq):
    """
    Compute, for each index in a binary warning array, the length
    of the current run of 1’s up to that point.

    Returns:
    --------
    np.ndarray of same shape as input.
    """
    # Pad with zeros at both ends to detect run boundaries
    ext = np.concatenate(([0], binary_seq, [0]))
    # Find change points
    idx = np.flatnonzero(ext[1:] != ext[:-1])
    # Mark run lengths at the end of each run
    ext[1:][idx[1::2]] = idx[1::2] - idx[::2]
    # Cumulative sum yields run-length at each position
    return ext.cumsum()[1:-1]


#---------------------------------------------------#
#------------- FEATURE IMPORTANCE & EVALUATION ------#
#---------------------------------------------------#

def feature_importance(mw, shift_idx, featidx_warn, featnames, permut_import=False):
    """
    Train a RandomForest classifier to distinguish pre- vs. post-shift windows using
    features that flagged warnings. Return either standard or permutation importances.

    Parameters:
    -----------
    mw             : np.ndarray, shape (Nfeat, Nshift)
                     Mann-Whitney stats per feature/shift.
    shift_idx      : int
                     Index of the detected shift in time.
    featidx_warn   : list or array
                     Indices of features that exceeded warning threshold.
    featnames      : list of str
                     Names corresponding to those indices.
    permut_import  : bool
                     If True, compute permutation importance after fitting.

    Returns:
    --------
    importances : np.ndarray, length = len(featidx_warn)
    """
    Nshift = mw.shape[1]

    # Binary labels: 0 for pre-shift, 1 for post-shift
    labels = np.hstack([np.zeros(Nshift - shift_idx),
                        np.ones(shift_idx)])

    # Build feature matrix for warning features: shape (Nshift, Nwarn)
    X = pd.DataFrame(
        mw[featidx_warn, :].T,
        columns=[featnames[i] for i in featidx_warn]
    )

    # Split train/test stratified on labels
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, stratify=labels, random_state=42, n_jobs =-1
    )

    # Fit RandomForest
    clf = RandomForestClassifier(random_state=0)
    clf.fit(X_train, y_train)
    print(f"RF accuracy: {clf.score(X_test, y_test):.3f}")

    # Standard feature importances
    importances = clf.feature_importances_

    # Optional permutation importances
    if permut_import:
        result = permutation_importance(
            clf, X_test, y_test,
            n_repeats=50, random_state=42, n_jobs=-1
        )
        importances = result.importances_mean

    return importances


def feature_warnings(feature_alerts, Nfeat, alert_thresh, wtime):
    """
    For each feature, determine:
      - Earliest time it sustained alert_thresh consecutive warnings.
      - Number of earlier 'false-positive' runs.

    Parameters:
    -----------
    feature_alerts : np.ndarray, shape (Nfeat, Nshift)
                     Binary flag per feature/shift.
    Nfeat          : int
                     Total feature count.
    alert_thresh   : int
                     Consecutive-warning threshold to count as valid alert.
    wtime          : np.ndarray, shape (Nshift,)
                     Time axis for shifts.

    Returns:
    --------
    featwarns : np.ndarray, length Nfeat
                Warning time per feature (NaN if none).
    fps       : np.ndarray, length Nfeat
                Count of false-positive runs before the valid alert.
    """
    featwarns = np.full(Nfeat, np.nan)
    fps       = np.full(Nfeat, np.nan)

    for k in range(Nfeat):
        # RLE on warning flags for this feature
        runs = s0.rle(feature_alerts[k, :].astype(int))
        # Extract lengths of runs where value==1
        ones_runs = runs[runs[:, 0] == 1, 1]

        if ones_runs.size:
            # Valid alert if any run exceeds threshold
            long_runs = np.where(ones_runs > alert_thresh)[0]
            if long_runs.size:
                # Index of the final long run -> earliest alert time
                idx = long_runs[-1]
                featwarns[k] = wtime[-ones_runs[idx]]
                # False positives = all earlier long runs
                fps[k] = max(0, long_runs.size - 1)

    return featwarns, fps


def best_of_category(feature_table, metric):
    """
    From a DataFrame of features with 'family', 'name', and a performance metric,
    select the best (max or min) feature per family.

    Parameters:
    -----------
    feature_table : pd.DataFrame
                    Must include columns ['family', 'name', metric].
    metric        : str
                    Column name to optimize ('warning_time', 'importance', etc.).
                    Maximize for 'importance', minimize for 'warning_time'.

    Returns:
    --------
    pd.DataFrame
        Rows of the best feature per family.
    """
    # Determine grouping behavior
    if metric in ['importance']:
        best = feature_table.groupby('family')[metric].nlargest(1)
    else:
        best = feature_table.groupby('family')[metric].nsmallest(1)

    # Extract final rows
    rows = best.index.get_level_values(1)
    out = feature_table.loc[rows, ['family', 'name', metric]].reset_index(drop=True)
    return out
