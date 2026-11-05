# Titanic survival: PyTorch classifier + Streamlit app

Predicts Titanic passenger survival from Kaggle's `train.csv`: EDA in a notebook, preprocessing based on it, a PyTorch neural network trained by `train.py`, and a Streamlit app (`ds_app.py`) that loads the saved model, scores a CSV and shows metrics and plots.

Only `train.csv` is used, split once into 712 training and 179 validation passengers (stratified, seed 42).

| | Accuracy | ROC AUC |
|---|---|---|
| 5-fold cross-validation (training split) | 83.4% ± 2.5% | 0.876 |
| **Validation split** | **83.2%** | **0.870** |
| Baseline: "women survive" | 77.7% | – |

## Setup instructions

- Python 3.10+ (CPU is enough).
- A Kaggle API token, to download the data:
  1. [kaggle.com/settings](https://www.kaggle.com/settings) → API → **Create New Token**.
  2. Accept the competition rules at [kaggle.com/competitions/titanic/rules](https://www.kaggle.com/competitions/titanic/rules).
  3. Copy `.env.example` to `.env` and paste the token after `KAGGLE_API_TOKEN=`. `.env` is git-ignored. (On Colab, use a secret named `KAGGLE_API_TOKEN` instead.)

A trained model is already in `artifacts/`, so the app works without a token.

## Installation commands

```bash
git clone https://github.com/hillysegal1/assignment.git
cd assignment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # Windows: copy .env.example .env, then add your token
```

## Run instructions

**Train the model**

```bash
python train.py                                # downloads train.csv from Kaggle
python train.py --data-path path/to/train.csv  # or use a local copy (no token needed)
```

It writes `artifacts/model.pt` (weights, network shape and fitted preprocessing), `artifacts/validation.csv` (the held-out passengers) and `artifacts/training_report.json`. `python train.py --help` lists further options.

**Run the app**

```bash
streamlit run ds_app.py
```

It opens at `http://localhost:8501`.

**EDA notebook**

```bash
jupyter notebook notebooks/eda.ipynb
```

## Example usage

1. Start the app. The sidebar is pre-filled with `artifacts/model.pt` and `artifacts/validation.csv`; enter any other model or CSV path, or upload a CSV.
2. Click **Run inference**.
3. Explore the tabs:
   - **Metrics:** accuracy, precision, recall, F1, ROC AUC, log loss, and a comparison with simple baselines.
   - **Plots:** confusion matrix, probability distribution, ROC and precision–recall curves, the threshold trade-off, calibration.
   - **By group:** results per sex and class.
   - **Features:** permutation importance, showing which columns the model relies on.
   - **Predictions:** a table per passenger, with a download button.
   - **Training:** the cross-validation learning curves.
4. Move the **decision threshold** slider to see the precision/recall trade-off.

Try `data/sample_unlabeled.csv` to see predictions without metrics. Wrong paths or missing columns show a clear message.

## Architecture and design choices

```
train.py            training script: data split, training loop, cross-validation, final fit, saving
ds_app.py           Streamlit app
src/titanic/        code shared by train.py and the app: data.py, preprocessing.py, model.py, checkpoint.py, evaluate.py
notebooks/eda.ipynb exploratory data analysis
artifacts/          trained model and validation split
data/               small sample CSVs (full data: Kaggle)
tests/              pytest checks
```

- **No leakage.** The EDA's survival analysis and all preprocessing statistics use the training split only. The validation split is scored once, at the end.
- **Preprocessing from the EDA:**
  - Title extracted from Name; rare titles mapped to Mr or Mrs by sex.
  - Missing ages filled with the median per title, with Miss split by whether the passenger travels with parents or children.
  - Zero fares treated as missing, then log-transformed.
  - Cabin reduced to a "recorded" flag.
  - One-hot class, port and title; standardised numbers.
  - The fitted values are saved inside the checkpoint, so the app transforms data exactly as training did.
- **Model:** an MLP, 17 → 32 → 16 → 1, with ReLU, dropout 0.2, AdamW and `BCEWithLogitsLoss`. Network sizes from 8 to 128 units, and a logistic regression, all scored within noise of each other in cross-validation; the data, not the architecture, limits accuracy.
- **Training:** 5-fold cross-validation picks the number of epochs (42) by held-out log loss, then one model is trained on all training rows. Runs are fully reproducible.
- **Safe checkpoint:** it loads with `torch.load(weights_only=True)`.
- **Evaluation:** metrics beyond accuracy, baselines, calibration and permutation importance. The model is weakest on 3rd-class women and 1st-class men, whose survival was close to a coin toss.
