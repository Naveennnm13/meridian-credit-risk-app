# Meridian Credit Intelligence

An explainable credit-risk decisioning tool built on the UCI Statlog German Credit
dataset. Every decision comes with a plain-language reason, a visual breakdown of what
drove it, and documentation of the fairness audit that shaped which features the model
is allowed to use.

**Live demo:** https://meridian-credit-risk-app.onrender.com/

> This is a portfolio / learning project, not a production underwriting system. The
> dataset is small, dated, and denominated in a currency that no longer exists (see
> [Known limitations](#known-limitations)).

## How it works

1. **Form submission** — `templates/index.html` renders an 18-field applicant form from
   `label_maps.pkl` (the human-readable labels shown in each dropdown). On submit,
   `static/script.js` POSTs the raw field values as JSON to `/predict`.
2. **Encoding** — `app.py` encodes each categorical field using the exact
   `LabelEncoder` fit during training (`label_encoders.pkl`), then assembles a single-row
   DataFrame in the model's expected feature order.
3. **Prediction** — the row is passed to a pre-trained `XGBClassifier`
   (`credit_model.pkl`) to get a Good/Bad Credit decision and an approval probability.
4. **Explainability (SHAP)** — a `shap.TreeExplainer` computes the per-feature
   contribution (SHAP value) of every input for *this specific prediction*, not just
   global feature importance. The top contributions are returned to the frontend and
   rendered as a horizontal bar chart (Chart.js) showing which factors pushed the
   decision toward approval or denial, and by how much.
5. **Plain-language explanation (LLM layer)** — the strongest SHAP factors (in each
   direction) are passed to an LLM (OpenAI `gpt-4o-mini`), which is prompted to produce a
   fixed-format explanation: primary reason, supporting factors, one offsetting factor,
   and a one-sentence summary — written the way an adverse-action notice would need to
   read, with no model internals or jargon exposed to the end user.
6. **Confidence calibration** — predictions within `CONFIDENCE_THRESHOLD = 0.08` of the
   0.5 decision boundary are labeled "Borderline" rather than presented with false
   certainty. The threshold was chosen by inspecting the actual distribution of predicted
   probabilities on the test set, not picked arbitrarily.

The whole response (decision, probability, confidence, explanation, SHAP factors) is
returned as one JSON payload and rendered in place in the results panel.

## Dataset

[UCI Statlog (German Credit Data)](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data) —
1,000 historical loan applications from a German bank, each labeled Good or Bad credit
risk, with attributes such as checking/savings account status, credit history, loan
purpose, duration, and employment. Donated to the UCI Machine Learning Repository by
Hans Hofmann in 1994.

The dataset's original 20 attributes were reduced to the **18 features** this model
actually uses — see [Fairness audit](#fairness-audit-and-bias-metric) below for why two
were dropped.

## Model

- **Algorithm:** `XGBClassifier` (gradient-boosted trees), 200 estimators, max depth 4,
  learning rate 0.05, `objective=binary:logistic`.
- **Class weighting:** `scale_pos_weight ≈ 0.43` (3:7), deliberately down-weighting the
  "Good Credit" class so the model errs toward flagging risk rather than missing it.
- **Deliberate precision/recall tradeoff:** the model prioritizes **recall on Bad
  Credit** over raw accuracy — **67% recall, 53% precision** on the Bad Credit class —
  because a missed default is costlier to a lender than a false rejection. A grid search
  that pushed recall higher was tested and rejected because it rejected too many
  creditworthy applicants.
- **Explainability:** `shap.TreeExplainer`, giving exact per-decision feature
  attributions for tree ensembles (not an approximation).

Model training, validation, and the fairness audit below were done separately (Colab
notebook, not included in this repo); this repo only serves the trained artifacts.

## Fairness audit and bias metric

**Metric used: disparate approval rate (demographic/statistical parity) across
subgroups.** Before finalizing the feature set, the model's approval rate was compared
across subgroups defined by two sensitive attributes present in the original dataset:

- `personal_status` — conflates **gender and marital status** in the source data (e.g.
  "male single" vs. "female divorced/separated/married"), a known issue with this dataset.
- `foreign_worker` — immigration/residency status.

The audit found the model produced **unequal approval rates across these groups with no
corresponding accuracy benefit** — i.e., the features were adding disparate impact
without improving predictive performance. Both features were removed entirely and the
model retrained on the remaining 18 features, which eliminated the disparity at no cost
to accuracy, precision, or recall.

This is a blunt, pre-training fix (exclude the sensitive attribute) rather than a
post-hoc fairness constraint (e.g. equalized odds, calibration parity) applied at
prediction time. It's documented here, including the tradeoff it represents, rather than
hidden — same as the dataset's own data-collection artifacts below.

## Known limitations

- **Counterintuitive credit history coding.** The dataset codes `"no credit history /
  all paid duly"` as a *negative* signal relative to `"critical account, other credits
  existing"` — the reverse of what most people would expect. This is a known artifact of
  how the original data was collected, not a bug, and the app's Methodology section
  surfaces it rather than masking it.
- **Currency and dataset age.** Loan/credit amounts are in **Deutsche Marks (DM)** — the
  dataset predates the Euro, reflecting German lending conditions from the early 1990s.
  Amounts are not converted or rescaled to modern currency.
- **Category coverage gap.** `label_maps.pkl` (used to render dropdown options) was
  originally generated from the full official attribute codebook, but the `purpose`
  category `"Vacation"` never appears in the actual 1,000-row training data, so its
  `LabelEncoder` never learned it. It has been removed from the dropdown so the UI only
  ever offers values the model can actually handle; `/predict` also fails cleanly with a
  descriptive error on any future unseen-category mismatch instead of crashing.

## Project structure

```
app.py                  Flask backend — encoding, prediction, SHAP, LLM explanation
templates/index.html    Applicant form + results panel (built from label_maps.pkl)
static/style.css        Styling
static/script.js        Form submission, results rendering, SHAP chart (Chart.js)
credit_model.pkl        Trained XGBClassifier (18 features)
label_encoders.pkl      One LabelEncoder per categorical column, fit during training
label_maps.pkl          Maps raw codes (e.g. "A11") to human-readable form labels
requirements.txt        Python dependencies
Procfile                Render/gunicorn start command
```

## Running locally

```powershell
python -m venv venv
./venv/Scripts/Activate.ps1
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your-key-here
```

Then run:

```powershell
python app.py
```

The app will be available at `http://127.0.0.1:5000`.

## Deployment

Deployed on [Render](https://render.com) (free tier):

- **Build command:** `pip install -r requirements.txt`
- **Start command:** `gunicorn app:app` (see `Procfile`)
- **Environment variable:** `OPENAI_API_KEY` set in the Render dashboard, not committed
  to the repo
