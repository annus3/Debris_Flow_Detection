# Debris‐Flow Early‐Warning Analysis (DF-EWS)

A Python framework for detecting **debris-flow early warning signals** using seismic data, TSFRESH-based time-series features, and Mann–Whitney statistical analysis.

This project enables scientific exploration and practical detection of early warning signals preceding debris-flow events by applying a robust signal processing and machine learning pipeline.

---

## 🔍 Project Description

**Debris‐Flow Early‐Warning Analysis (DF-EWS)** is a modular framework to process seismic data for debris-flow events and detect early warning signals (EWS) based on statistical and machine learning techniques. The pipeline supports:

- SAC waveform preprocessing and time alignment
- Precursor segmentation using RMS-based windowing
- Feature extraction via TSFRESH (with surrogate null modeling)
- Mann–Whitney U statistical testing
- Alert signal detection and correction
- Visualization of warning timelines and time-series segments

Developed in collaboration with geophysical monitoring researchers, this pipeline is designed for reproducibility, parallel processing, and clear visualization of early warnings.

---

## 📦 Installation

```bash
cd path/to/this/folder
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 🚀 Usage

Ensure your seismic data is placed under:

```
data/df_data/<DATE>-DF/IGB02/<DATE>-BHZ
```

To run the pipeline for a given event:

1. Edit the selected index in `main.py`:
   ```python
   M = 1  # select event index (0-based)
   ```

2. Run the main script:
   ```bash
   python main.py
   ```

---

## 📁 Scripts: File Descriptions

### `main.py` – Driver Pipeline

Runs the full debris-flow EWS pipeline in seven steps:

1. **Preprocessing**  
   - Reads SAC file → detrends, filters (1–45 Hz)  
   - Builds uniform time axis  
   - Adjusts for midnight-crossing events  

2. **Segmentation**  
   - Uses `PRE_DF_split()` to extract pre-event segments  
   - Saves catalogue CSV with window metadata  

3. **Feature Extraction**  
   - Calls `EWS_analysis()` with TSFRESH EfficientFCParameters  
   - Generates `Nsurr` surrogate time series by shuffling  

4. **Mann–Whitney Testing**  
   - `MW_analysis()` computes MW-U statistics  
   - Saves matrices for real and surrogate features  

5. **Alerts & Statistics**  
   - Computes p-values with multiple-testing correction  
   - Builds family-wise alert counts and co-occurrence matrix  
   - Determines early warning timestamp and top features  

6. **Final Segmentation**  
   - Splits time series into noise / precursor / DF  
   - Saves each as `.npy` arrays  

7. **Plotting**  
   - Creates EWS probability plot and per-feature warning timeline  

---

### `s0_subroutines.py` – Utilities

- `rle(sequence, series=None)`  
  Run-Length Encoding of 1D sequences (used to detect sustained alert regions)

---

### `s1_PRE_DF.py` – Precursor Segmentation

- `PRE_DF_split()`  
  - Computes RMS over sliding window  
  - Applies binary segmentation to detect DF onset  
  - Outputs a TSFRESH-formatted DataFrame

- Helper functions:  
  - `getfam()`, `getfeat()` for parsing TSFRESH feature names  
  - `get_pvals()` to apply empirical and corrected p-value tests  
  - `generate_sequences()`, `simulate_coincidence()` for null hypothesis modeling  
  - `consec_warnings()`, `feature_importance()`, `feature_warnings()`, `best_of_category()`  

---

### `s2_EWS.py` – Feature Extraction and EWS Analysis

- `featurize()`  
  Extracts TSFRESH features per window using a given feature set configuration.

- `EWS_analysis()`  
  - Computes features for real and surrogate datasets  
  - Runs in parallel for performance  

- `MW_analysis()`  
  - Applies Mann–Whitney U testing across time shifts  
  - Outputs: time axis, MW scores for real and surrogate features  

- `EWS_split()`  
  - Detects earliest point where real features diverge from surrogate distribution  

---

### `s3_alerts.py` – Alerts, Warnings & Evaluation

- `get_pvals()`  
  Applies empirical p-value computation and multiple-testing correction (e.g. FDR)

- `count_signif_families()`  
  Computes family-level alert frequency and co-occurrence

- `consec_warnings()`  
  Encodes sequences of binary alert signals

- `feature_warnings()`  
  - Detects when features trigger sustained alerts  
  - Counts false positives before real alerts

- `feature_importance()`  
  Uses Random Forest (or permutation importance) to assess signal discriminability

- `best_of_category()`  
  Selects best-performing feature per family (e.g. by importance or early warning time)

---

### `TSFRESH_feature_calculators.py` & `TSFRESH_settings.py`

- Directly imported from the [TSFRESH](https://github.com/blue-yonder/tsfresh) repository (MIT license)
- Used to customize and control which time-series features to compute

---

## 📂 Data and Output Structure

```bash
data/df_data/<DATE>-DF/IGB02/<DATE>-BHZ    # Input seismic SAC waveform
assets/
├── catalogue_<DATE>.csv (not included)    # Segmentation catalogue
├── debris_flow_feature_vectors/<DATE>/    # TSFRESH real and surrogate features
├── mann_whitney_testing/                  # MW-U matrices (real and surrogates)
├── debris_flow_segments/<DATE>/           # .npy files for noise/precursor/DF
├── segmentation/<DATE>/                   # PNG figures for EWS and alerts
```

---

## 🧪 Dependencies

Install all required packages via:

```bash
pip install -r requirements.txt
```

### Key Libraries:

- `numpy`, `pandas`, `scipy`
- `obspy`, `ruptures`, `tsfresh`
- `scikit-learn`, `statsmodels`, `joblib`
- `tqdm`, `matplotlib`, `cmcrameri`

---

## 📄 License

- All custom code: **MIT License**
- TSFRESH modules used: **MIT License** (see original project)

---

## 📫 Citation / Contact

If you use this pipeline in your research, please cite the repository or contact the author.

> Developed as part of a seismic early warning study for debris-flow events.
