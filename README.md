# Debris‐Flow Early‐Warning Analysis

This subfolder contains a custom implementation of the debris‐flow EWS pipeline.  
_In the same repository you will also find:_
- **`alt_implementation/`** — a parallel version of these scripts, organized differently (see `alt_implementation/README.md`).  
- **Official TSFRESH modules**:  
  - `TSFRESH_feature_calculators.py`  
  - `TSFRESH_settings.py`  

---

## ⚙️ Installation

```bash
cd path/to/this/folder
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
````

---

## 📖 Usage

Ensure your SAC data lives under:

```
data/df_data/<DATE>-DF/IGB02/<DATE>-BHZ
```

Run the pipeline for event index `M`:

```bash
python main.py
```

Edit `main.py`:

```python
M = 1  # select event by index
```

---

## 🗂️ Script Folder: File Descriptions

### 1. `main.py`

**Driver script** orchestrating the entire workflow.
**Steps:**

1. **Preprocessing (STEP 0)**

   * Reads SAC waveform → detrend & filter → builds time axis
   * **Input:** `data/df_data/<DATE>-DF/IGB02/<DATE>-BHZ`
2. **Segmentation (STEP 1)**

   * `s1_PRE_DF.PRE_DF_split()` → saves catalogue CSV
   * **Output:** `assets/catalogue_<DATE>.csv`
3. **Featurization (STEP 2.1)**

   * `s2_EWS.EWS_analysis()` → saves feature vectors & surrogates
   * **Output:** `assets/debris_flow_feature_vectors/<DATE>/featvec_*.csv`
4. **MW Testing (STEP 2.2)**

   * `s2_EWS.MW_analysis()` → saves `.npy` matrices
   * **Output:** `assets/mann_whitney_testing/mwmat_<DATE>*.npy`
5. **Alerts & Stats (STEP 3.1)**

   * `s3_alerts.get_pvals()`, `count_signif_families()`, etc. → compute EWS time
6. **Segmentation Save (STEP 3.2)**

   * Partition time‐series into noise/precursor/DF segments → saves `.npy`
   * **Output:** `assets/debris_flow_segments/<DATE>/*.npy`
7. **Plotting (STEP 3.3)**

   * Generates EWS & warning‐times figures
   * **Output:** `assets/segmentation/<DATE>/*.png`

---

### 2. `s0_subroutines.py`

Utility functions used across modules.

* `rle(sequence, series=None)`:
  Run‐length encoding on a sequence. Returns either all `(value, length)` pairs or, if `series` specified, only lengths for runs equal to that value.

---

### 3. `s1_PRE_DF.py`

Pre‐DF segmentation via RMS & change‐point detection:

* `PRE_DF_split(signal, w, step, t0, tfin, min_size, buffer, noise_win, verbose=False)`

  * Slides window of width `w` every `step` seconds
  * Computes RMS → binary segmentation (`ruptures.Binseg`) → detects start/end of DF
  * Returns `(shift_time, end_time, d_train)`
  * `d_train`: TSFRESH‐formatted DataFrame for all pre‐DF windows

---

### 4. `s2_EWS.py`

Feature extraction & Mann‐Whitney analysis:

* `featurize(df, feat_settings)`: extract tsfresh features for one window
* `EWS_analysis(d_preDF, Nsurr, feat_settings)`:

  * Generate `Nsurr` surrogates by shuffling
  * Parallel featurization for real & surrogate windows
* `MW_analysis(real_feats, surr_feats, Wmw, Nrndm)`:

  * Compute Mann–Whitney U over sliding-window vs. random noise segments
  * Returns `(a_tright, a_mw, a_mw_rndm)`

---

### 5. `s3_alerts.py`

Statistical testing & alert generation:

* **Extraction helpers**:

  * `getfam(name)`, `getfeat(name)`: parse tsfresh feature names
* **P-value routines**:

  * `get_pvals(mw, mw_surr, MT_METHOD, alpha)`: empirical + corrected p-values
  * `count_signif_families(feature_alerts, featnames, famlist, mw_surr)`: co-occurrence matrix + fractional family counts
* **Simulation**:

  * `generate_sequences(N, T, p)`, `simulate_coincidence(sequences, coincidence_matrix)`
* **Warning detection**:

  * `consec_warnings(binary_seq)`: run‐length encoding of alert flags
  * `feature_warnings(feature_alerts, Nfeat, alert_thresh, wtime)`: earliest warn time & false-positive count per feature
* **Evaluation**:

  * `feature_importance(...)`: RandomForest (or permutation) importances
  * `best_of_category(feature_table, metric)`: select top feature per family by metric
  
6. **Official TSFRESH modules**

   * **`TSFRESH_feature_calculators.py`**

     > From tsfresh (MIT license). Defines all low-level time‐series feature calculators (simple & combiner).
   * **`TSFRESH_settings.py`**

     > From tsfresh (MIT license). Controls which features are enabled during extraction.

---

## 🔗 Data & Output Paths

* **Catalogue**:
  `not_included`

* **Features**:
  `assets/debris_flow_feature_vectors/<DATE>/`

* **MW Matrices**:
  `assets/mann_whitney_testing/mwmat_<DATE>*.npy`

* **Segments**:
  `assets/debris_flow_segments/<DATE>/*.npy`

* **Plots**:
  `assets/segmentation/<DATE>/*.png`

---

## 📦 Dependencies

```bash
pip install -r requirements.txt
```

Key packages:

* `numpy`, `pandas`, `obspy`
* `scipy`, `tqdm`, `ruptures`
* `tsfresh`, `statsmodels`, `scikit-learn`
* `matplotlib`, `cmcrameri`, `joblib`

---

## 📜 License

All custom code: **MIT License**.
TSFRESH modules: **MIT License** (see their header comments).

```
```
