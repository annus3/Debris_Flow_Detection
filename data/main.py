import os
import numpy as np
import pandas as pd


from obspy import read, UTCDateTime
import scipy
from scipy.stats import mannwhitneyu, kstest
from tqdm import tqdm
from tsfresh.feature_extraction import ComprehensiveFCParameters, EfficientFCParameters
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import gridspec
from cmcrameri import cm


import s0_subroutines as sub
import s1_PRE_DF as s1
import s2_EWS as s2
import s3_alert as s3

# Define base directories for data inputs and outputs.
# Using os.path.join ensures compatibility across operating systems.
BASE_DATA_DIR    = os.path.join(os.path.expanduser('~'), 'Desktop', 'GFZ', 'DF', 'df_data')
FEATURE_BASE_DIR = os.path.join(BASE_DATA_DIR, 'debris_flow_feature_vectors')
MW_BASE_DIR      = os.path.join(BASE_DATA_DIR, 'mann_whitney_testing')
SEGMENT_BASE_DIR = os.path.join(BASE_DATA_DIR, 'debris_flow_segments')
RESULTS_SEG_DIR  = os.path.join(BASE_DATA_DIR, 'results', 'segmentation')

#---------------------------------------------------#
#------------------ GLOBALS ------------------------#
#---------------------------------------------------#
# List of event dates and their corresponding start (T0) and end (T1) times.
l_dates = ['2013-07-22', '2014-05-07', '2014-06-19', '2014-06-23',
           '2014-07-08', '2014-07-12', '2014-07-20', '2014-07-23',
           '2014-07-28', '2014-08-02', '2014-08-08', '2014-09-08']
l_T0    = ['14:45:00', '13:50:00', '12:10:00', '16:00:00',
           '08:00:00', '12:55:00', '22:00:00', '20:55:00',
           '14:00:00', '17:10:00', '17:10:00', '17:55:00']
l_T1    = ['19:45:00', '18:50:00', '17:10:00', '21:00:00',
           '13:00:00', '17:55:00', '01:00:00', '02:55:00',
           '21:00:00', '22:10:00', '22:10:00', '22:55:00']

# Select which event to process (index into the above lists)
M = 1

#---------------------------------------------------#
#------------------ INPUT SETUP --------------------#
#---------------------------------------------------#
# Retrieve the event-specific date and times based on the selection index M.
DATE = l_dates[M]
T0   = l_T0[M]
T1   = l_T1[M]

#---------------------------------------------------#
#-------------- STEP 0: PREPROCESS -----------------#
#---------------------------------------------------#
# Read seismic signal, build the time axis, handle midnight cases,
# and apply detrending and bandpass filtering.
SAC_DIR       = os.path.join(BASE_DATA_DIR, f"{DATE}-DF")
sac_file_path = os.path.join(SAC_DIR, 'IGB02', f"{DATE}-BHZ")
o_stz         = read(sac_file_path)
print(f"Reading data from: {sac_file_path}")
print(o_stz)

# Extract full time series array
a_fullts = np.array(o_stz[0].data)

# Handle events that cross midnight
if M == 6:
    DATE1, DATE2 = '2014-07-20', '2014-07-21'
elif M == 7:
    DATE1, DATE2 = '2014-07-23', '2014-07-24'
else:
    DATE1, DATE2 = DATE, DATE

# Build time axis at 5 ms intervals
a_tax = np.arange(
    np.datetime64(f"{DATE1} {T0}"),
    np.datetime64(f"{DATE1} {T1}"),
    np.timedelta64(5, 'ms')
)
t0   = UTCDateTime(f"{DATE1} {T0}")
tfin = UTCDateTime(f"{DATE2} {T1}")

# Detrend and bandpass filter (1–45 Hz)
o_stz.detrend('linear')
o_stz.detrend('demean')
o_stz.filter('bandpass', freqmin=1, freqmax=45)
a_fullts = np.array(o_stz)[0, :-1]

#---------------------------------------------------#
#--------------- STEP 1: SEGMENTATION --------------#
#---------------------------------------------------#
# Use the PRE_DF module to split waveform into pre-DF segments
# and save the resulting catalogue.
RUN, SAVE = True, True
signal     = o_stz
w, step    = 10, 10    # window size and step in seconds
min_size   = 28        # minimum segment size
buffer     = 30        # buffer seconds
noise_win  = 3600      # noise window size in seconds

if RUN:
    DF_time1, DF_time2, d_preDF = s1.PRE_DF_split(
        signal, w, step, t0, tfin, min_size, buffer, noise_win
    )
    print(d_preDF)
    if SAVE:
        output_path = os.path.join(BASE_DATA_DIR, f"catalogue_{t0.date}.csv")
        d_preDF.to_csv(output_path, index=False)

#---------------------------------------------------#
#--------------- STEP 2.1: FEATURIZATION ------------#
#---------------------------------------------------#
# Extract features from pre-DF segments and generate surrogates.
RUN, SAVE = True, True
Wmw    = 20   # MW window count
Nrndm  = 10   # random noise windows
feat_settings = EfficientFCParameters()
Nsurr  = 5    # surrogate sets

if RUN:
    d_features, l_features_surr = s2.EWS_analysis(d_preDF, Nsurr, feat_settings)
    feature_cols = d_features.columns
    # Determine feature-family counts
    a_feature_families, a_ffamily_sizes = np.unique(
        np.hstack([s3.getfam(feature_cols[n]) for n in range(feature_cols.size)]),
        return_counts=True
    )

if SAVE:
    output_dir = os.path.join(FEATURE_BASE_DIR, DATE)
    os.makedirs(output_dir, exist_ok=True)
    d_features.to_csv(os.path.join(output_dir, f"featvec_{DATE}.csv"), index=False)

    # Save surrogates in parallel
    from joblib import Parallel, delayed
    def save_surrogate(idx, features_surr, date, out_dir):
        features_surr[idx].to_csv(
            os.path.join(out_dir, f"featvec_surr_{idx+1}_{date}.csv"),
            index=False
        )
    Parallel(n_jobs=-1)(
        delayed(save_surrogate)(ns, l_features_surr, DATE, output_dir)
        for ns in range(Nsurr)
    )

#---------------------------------------------------#
#--------------- STEP 2.2: MW TESTING --------------#
#---------------------------------------------------#
# Run Mann–Whitney U tests and save the result matrices.
RUN, LOAD, SAVE = True, True, True

if LOAD:
    feat_dir = os.path.join(FEATURE_BASE_DIR, DATE)
    d_features = pd.read_csv(os.path.join(feat_dir, f"featvec_{DATE}.csv"))
    l_features_surr = []
    for i in range(Nsurr):
        l_features_surr.append(
            pd.read_csv(os.path.join(feat_dir, f"featvec_surr_{i+1}_{DATE}.csv"))
        )

if RUN:
    a_time, a_mw, a_mw_rndm = s2.MW_analysis(d_features, l_features_surr, Wmw, Nrndm)

if SAVE:
    np.save(os.path.join(MW_BASE_DIR, f"mwmat_{DATE}_time.npy"), a_time)
    np.save(os.path.join(MW_BASE_DIR, f"mwmat_{DATE}.npy"), a_mw)
    np.save(os.path.join(MW_BASE_DIR, f"mwmat_surr_{DATE}.npy"), a_mw_rndm)

#---------------------------------------------------#
#--------------- STEP 3.1: ALERTS & STATS -----------#
#---------------------------------------------------#
# Compute p-values, early-warning alerts, and feature stats.
RUN, LOAD = True, True

if LOAD:
    a_time      = np.load(os.path.join(MW_BASE_DIR, f"mwmat_{DATE}_time.npy"))
    a_mw        = np.load(os.path.join(MW_BASE_DIR, f"mwmat_{DATE}.npy"))
    a_mw_rndm   = np.load(os.path.join(MW_BASE_DIR, f"mwmat_surr_{DATE}.npy"))

#— Analysis parameters —#
alpha        = 0.1
MT_METHOD    = 'fdr_by'
Nbinom       = 50
alpha_binom  = 0.1
ALERT_THRESH = 30

#— Time axis in minutes before DF —#
a_twarn = (a_time[-1] - a_time) * 10 / 60

#— Exclude constant features —#
mask          = np.any(np.diff(a_mw, axis=1) != 0, axis=1)
a_mw          = a_mw[mask]
a_mw_rndm     = a_mw_rndm[:, mask]
l_featnames   = d_features.columns[mask]
Nfeat, Nshift = a_mw.shape
a_famlist     = np.hstack([s3.getfam(name) for name in l_featnames])

#— Corrected p-values —#
a_pval_corr = s3.get_pvals(a_mw, a_mw_rndm, MT_METHOD, alpha)

#— Which features “fired” at each shift —#
a_feat_alerts = (a_pval_corr < alpha)

#— Build coincidence matrix and fractional family counts —#
probmat_feat, a_Nff = s3.count_signif_families(
    a_feat_alerts,
    l_featnames,
    a_famlist,
    a_mw_rndm
)

#— Normalize to get Nfeat×Nfeat probability matrix —#
a_coincmat = probmat_feat.astype(float) / Nshift

#— Binomial surrogate testing —#
a_Nff_binom = np.zeros((Nbinom, Nshift))
p = 0.2
for n in tqdm(range(Nbinom), desc="Binomial sims"):
    seq     = s3.generate_sequences(Nfeat, Nshift, p)
    adj_seq = s3.simulate_coincidence(seq, a_coincmat)
    _, a_Nff_binom[n] = s3.count_signif_families(
        adj_seq,
        l_featnames,
        a_famlist,
        a_mw_rndm
    )

#— Early‐warning threshold & index —#
warnings_conf   = np.quantile(
    a_Nff_binom / np.unique(a_famlist).size,
    1 - alpha_binom
)
a_warnings      = a_Nff / np.unique(a_famlist).size
a_cont_warnings = s3.consec_warnings((a_warnings > warnings_conf).astype(int))

try:
    raw_idx = np.where(a_cont_warnings > ALERT_THRESH)[0][0] - ALERT_THRESH
    IDX     = max(0, raw_idx)
except IndexError:
    IDX = a_twarn.size - 2

EWS_minutes = a_twarn[IDX]
EWS_time    = DF_time1 - EWS_minutes * 60

#— Feature‐level warnings and stats —#
a_featwarns, a_fps = s3.feature_warnings(
    a_feat_alerts,
    Nfeat,
    ALERT_THRESH,
    a_twarn
)

#— Indices of features that actually warned —#
featidx_warn      = np.where(~np.isnan(a_featwarns))[0]
l_nonanfeats      = l_featnames[featidx_warn]
a_featwarns_nonan = a_featwarns[featidx_warn]
a_fps_nonan       = np.where(
    np.isnan(a_fps[featidx_warn]),
    0,
    a_fps[featidx_warn]
)
Lnonan = len(featidx_warn)

#— Redundancy and importance —#
a_redmat      = np.corrcoef(a_mw)
a_featredund0 = np.hstack([
    a_redmat[nf, np.where(a_famlist[nf] != a_famlist)[0]].mean()
    for nf in range(Nfeat)
])
a_featredund = a_featredund0[featidx_warn]

a_import = s3.feature_importance(
    a_mw,
    IDX,
    featidx_warn,
    l_featnames
)

#— Compile significant‐feature DataFrame —#
base_time = UTCDateTime(f"{DATE1} {T0}")
d_sigfeats = pd.DataFrame({
    'name':         l_nonanfeats,
    'family':       a_famlist[featidx_warn],
    # add each warning offset (in minutes) to base_time via pure-Python float
    'shift_time':   [base_time + float(dt) for dt in a_featwarns_nonan],
    'warning_time': a_featwarns_nonan,
    '#FPs':         a_fps_nonan,
    'redundancy':   a_featredund,
    'importance':   a_import
})

#— Best‐of‐feature tables —#
d_bestof_warning = s3.best_of_category(d_sigfeats, 'warning_time')
d_bestof_fps     = s3.best_of_category(d_sigfeats, '#FPs')
d_bestof_import  = s3.best_of_category(d_sigfeats, 'importance')

#---------------------------------------------------#
#--------------- STEP 3.2: SEGMENTATION ------------#
#---------------------------------------------------#
# Segment the full time series into noise, precursor, and DF windows
noise_idx = np.where(a_tax < EWS_time)[0]
ews_idx   = np.where((a_tax >= EWS_time) & (a_tax < DF_time1))[0]
df_idx    = np.where((a_tax >= DF_time1) & (a_tax < DF_time2))[0]

noise_time, noise_ts = a_tax[noise_idx], a_fullts[noise_idx]
ews_time,   ews_ts   = a_tax[ews_idx],   a_fullts[ews_idx]
df_time,    df_ts    = a_tax[df_idx],    a_fullts[df_idx]

seg_dir = os.path.join(SEGMENT_BASE_DIR, DATE)
os.makedirs(seg_dir, exist_ok=True)

# Save each segment’s time and data arrays
np.save(os.path.join(seg_dir, f"{DATE}_noise_time.npy"), noise_time)
np.save(os.path.join(seg_dir, f"{DATE}_noise_ts.npy"),   noise_ts)
np.save(os.path.join(seg_dir, f"{DATE}_ews_time.npy"),   ews_time)
np.save(os.path.join(seg_dir, f"{DATE}_ews_ts.npy"),     ews_ts)
np.save(os.path.join(seg_dir, f"{DATE}_df_time.npy"),    df_time)
np.save(os.path.join(seg_dir, f"{DATE}_df_ts.npy"),      df_ts)

#---------------------------------------------------#
#--------------- STEP 3.3: PLOTTING ---------------#
#---------------------------------------------------#
# Generate and save segmentation and EWS plots.

res_dir = os.path.join(RESULTS_SEG_DIR, DATE)
os.makedirs(res_dir, exist_ok=True)

PLOT = True
if PLOT:
    # Plot styling
    plt.style.use('dark_background')
    plt.rcParams.update({
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'axes.linewidth': 1.5, 'font.size': 12,
        'axes.xmargin': 0.01, 'axes.ymargin': 0.1
    })

    
    def rle(seq):
        from itertools import groupby
        return np.array([(k, len(list(g))) for k, g in groupby(seq)])

    fam_labels = np.hstack([s3.getfam(name) for name in l_featnames])
    reps       = rle(fam_labels)[:, 1].astype(int)
    edges      = np.cumsum(reps)
    locs       = edges - np.diff(np.hstack([0, edges]) / 2)
    cmap       = cm.managua(np.linspace(0, 1, edges.size))
    bar_colors = np.repeat(cmap, reps, axis=0)

    xformatter = mdates.DateFormatter('%H:%M')
    fig        = plt.figure()
    gs         = gridspec.GridSpec(14, 20, wspace=1, hspace=2)

    # Full time series plot
    ax_full = plt.subplot(gs[0:3, :])
    ax_full.plot(noise_time, noise_ts, color='gray')
    ax_full.plot(ews_time,   ews_ts,   color='goldenrod')
    ax_full.plot(df_time,    df_ts,    color='lightcoral')
    ax_full.set_ylabel('amplitude')
    ax_full.xaxis.set_major_formatter(xformatter)
    ax_full.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    # Precursor-only plot
    ax_prec = plt.subplot(gs[4:7, :])
    ax_prec.plot(noise_time, noise_ts, color='gray')
    ax_prec.plot(ews_time,   ews_ts,   color='goldenrod')
    ax_prec.set_ylabel('amplitude')
    ax_prec.xaxis.set_major_formatter(xformatter)
    ax_prec.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    # EWS probability plot
    ax_prob = plt.subplot(gs[8:14, :])
    ax_prob.scatter(a_twarn, np.where(a_warnings <= warnings_conf, a_warnings, np.nan), alpha=0.5)
    ax_prob.scatter(a_twarn, np.where(a_warnings >  warnings_conf, a_warnings, np.nan))
    ax_prob.hlines(warnings_conf, a_twarn[0], a_twarn[-1], linestyle='--', linewidth=2)
    ax_prob.set_xlabel('minutes before DF')
    ax_prob.set_ylabel('signif. feat. families')
    ax_prob.axvline(a_twarn[0], color='white')

    plt.savefig(os.path.join(res_dir, f"{DATE}_EWS.png"), dpi=300, bbox_inches='tight')

    # Bar chart of warning times
    fig = plt.figure(figsize=(5, 50))
    plt.barh(np.arange(Lnonan), a_featwarns_nonan, color=bar_colors)
    plt.yticks(locs, fam_labels[edges - 1], rotation=45)
    plt.tight_layout()
    plt.hlines(edges - 0.5, 0, np.nanmax(a_featwarns_nonan))
    plt.vlines(a_twarn[IDX], 0, Lnonan, linestyle='--')
    plt.xlabel('warning time [minutes]')
    plt.savefig(os.path.join(res_dir, f"{DATE}_warningtimes.png"), dpi=300, bbox_inches='tight')
