import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from tqdm import tqdm
import ruptures as rpt


def RMS(ts):
    """
    Calculate the Root Mean Square (RMS) of a time series.

    Parameters:
    ts (array-like): Input time series

    Returns:
    float: RMS value of the input time series
    """
    return np.sqrt(np.mean(np.array(ts)**2))


def PRE_DF_split(signal, w, step, t0, tfin,
                 min_size=28, buffer=60, noise_win=3600,
                 verbose=False):
    """
    Split the input signal into pre-debris flow (pre-DF) and debris flow (DF) segments,
    using simple change point detection on the RMS of the signal.

    Parameters:
    signal      (obspy.Stream)    : Input seismic signal
    w           (int)             : Window size for RMS calculation (sec)
    step        (int)             : Step size for moving window (sec)
    t0          (UTCDateTime)     : Start time of the signal
    tfin        (UTCDateTime)     : End time of the signal
    min_size    (int, optional)   : Min segment size for breakpoint detection (default: 28)
    buffer      (int, optional)   : Buffer time (sec) before DF onset (default: 60)
    noise_win   (int, optional)   : Noise window length (sec) for pre-DF (default: 3600)
    verbose     (bool, optional)  : If True, show RMS‐breakpoint diagnostic plot

    Returns:
    --------
    shift_time  (UTCDateTime)     : Estimated start time of DF (minus buffer)
    end_time    (UTCDateTime)     : Estimated end time of DF
    d_train     (pd.DataFrame)     : TS-Fresh formatted DataFrame for pre-DF
    """

    # Total number of samples in the first trace
    T = np.array(signal)[0, :].size

    # Build sliding windows, compute RMS per window
    i = 1
    t1, t2 = t0 + i * step, t0 + i * step + w
    l_ts, l_t2, l_rms = [], [], []

    while t2 < tfin:
        o_trimz = signal.copy()
        try:
            # Trim to [t1, t2] and extract data array
            o_trimz.trim(starttime=t1, endtime=t2)
            a_z = np.array(o_trimz)[0, :-1]
            l_ts.append(a_z)
            l_t2.append(t2)
            l_rms.append(RMS(a_z))
        except Exception as e:
            print(f"Warning: could not compute RMS for window ending {t2}: {e}")

        i += 1
        t1, t2 = t0 + i * step, t0 + i * step + w

    # Convert list of RMS values to numpy array (was np.hstack, which fails on scalars)
    a_rms = np.array(l_rms)

    # Change-point detection via binary segmentation on the RMS series
    algo = rpt.Binseg(model="l2", min_size=min_size)
    algo.fit(a_rms)
    result = algo.predict(n_bkps=2)

    # Map breakpoints back to times (subtract buffer in windows)
    IDX1 = result[0] - int(buffer / w)
    shift_time = l_t2[IDX1]
    IDX2 = result[1]
    end_time = l_t2[IDX2]

    # Stack all windowed time series into 2D array
    a_segts = np.vstack(l_ts)
    Tseg = a_segts.shape[1]

    # Compute how many windows count as noise / pre-DF
    noise_windows = int(noise_win / w)
    Nnoise = IDX1 - noise_windows
    Npre   = noise_windows

    # Assemble training TS-Fresh DataFrame: one long table with id/time/ts/label
    a_traints   = a_segts[:IDX1, :]
    a_ids       = np.repeat(np.arange(1, IDX1 + 1), Tseg)
    a_segtime   = np.tile(np.arange(Tseg), IDX1)
    a_labels    = np.hstack([
                     np.zeros(Nnoise * Tseg),
                     np.ones(Npre * Tseg)
                   ])

    d_train = pd.DataFrame({
        "id":    a_ids,
        "time":  a_segtime,
        "ts":    a_traints.ravel(),
        "label": a_labels.astype(int)
    })

    # Optional diagnostic plot
    if verbose:
        fig, ax = plt.subplots()
        ax.plot(l_t2, a_rms, label='RMS')
        ax.vlines([shift_time, end_time],
                  ymin=0, ymax=np.max(a_rms),
                  color='darkorange', linestyle='--',
                  label='Detected breakpoints')
        ax.legend()
        ax.set_title("RMS with Detected Debris-Flow Onset/End")
        plt.show()

    return shift_time, end_time, d_train
