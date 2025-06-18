# Anomaly Detection with Simulated Pretraining and Autoencoders

This repository contains the codebase for our paper on anomaly detection using pretraining on simulated data followed by fine-tuning on real data. We evaluate both vanilla and variational autoencoders (VAE) across various data splits and report their performance in identifying anomalous signals.

---

## 📁 Project Structure

```
root/
├── config.py                        # Global hyperparameters and path settings
├── data/
│   ├── data.py                     # Functions to load real/sim/anomaly datasets
│   └── datasets.py                # Custom PyTorch datasets for windowed training
├── models.py                       # LinearAE and LinearVAE definitions
├── evaluate.py                     # Metrics and loss computation for evaluation
├── loss.py                         # VAE loss function
├── training/
│   ├── pre_training.py
│   ├── pretraining_VAE.py
│   ├── pretraining_fixed_size.py
│   └── pretraining_fixed_size_VAE.py
├── optimization/
│   ├── optimize_pretraining.py
│   ├── optimize_pretraining_VAE.py
│   ├── optimize_Vanilla.py
│   └── optimizeVAE.py
├── results/                        # CSVs, plots, and logs
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### ✅ 1. Clone the repository

```bash
git clone https://github.com/yourusername/anomaly-autoencoder-paper.git
cd anomaly-autoencoder-paper
```

### ✅ 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate
```

### ✅ 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📂 Data

The required data is packaged in a zip archive:

```
Data_preprocessed.zip
```

After unzipping it, you'll need to **modify the paths in `config.py`** to point to your local data directories.

### Example:
```python
# config.py
real_data_folder = "/path/to/Data_preprocessed/Real/"
sim_data_folder = "/path/to/Data_preprocessed/Simulated/"
anom_folder_A1 = "/path/to/Data_preprocessed/Anomalies_A1/"
anom_folder_A2 = "/path/to/Data_preprocessed/Anomalies_A2/"
```


---

## 📊 How to Run Experiments

### 🔁 Optimize a Vanilla Autoencoder:
```bash
python optimization/optimize_Vanilla.py
```

### 🔁 Optimize a Variational Autoencoder (VAE):
```bash
python optimization/optimizeVAE.py
```

### 🔁 Run pretraining grid search (VAE):
```bash
python training/pretraining_VAE.py
```

> All results are saved in the `results/` folder as `.csv` and `.png`.

---

## 📈 Results

The scripts output:
- Sequence-level reconstruction losses
- Anomaly classification thresholds
- Precision, recall, F1, and accuracy
- Plot exports (e.g., `final_f1_VAE.png`)

---

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
tbd
```


