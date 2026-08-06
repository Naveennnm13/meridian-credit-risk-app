from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
import shap
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# ---- Load artifacts (same files you built in Colab) ----
model = joblib.load("credit_model.pkl")
encoders = joblib.load("label_encoders.pkl")
label_maps = joblib.load("label_maps.pkl")
explainer = shap.TreeExplainer(model)

FEATURE_ORDER = [
    "checking_account", "duration", "credit_history", "purpose", "credit_amount",
    "savings_account", "employment", "installment_rate", "other_debtors",
    "residence_since", "property", "age", "other_installments", "housing",
    "existing_credits", "job", "dependents", "telephone"
]

# Fields that are free-form numbers, not dropdowns
NUMERIC_FIELDS = {
    "duration": {"label": "Loan duration (months)", "min": 1, "max": 72, "default": 12},
    "credit_amount": {"label": "Credit amount requested (DM)", "min": 0, "max": 20000, "default": 2000},
    "installment_rate": {"label": "Installment rate (% of income)", "min": 1, "max": 4, "default": 2},
    "age": {"label": "Age", "min": 18, "max": 100, "default": 30},
    "residence_since": {"label": "Years at current residence", "min": 1, "max": 4, "default": 2},
    "existing_credits": {"label": "Number of existing credits", "min": 1, "max": 4, "default": 1},
    "dependents": {"label": "Number of dependents", "min": 1, "max": 2, "default": 1},
}

FEATURE_DISPLAY_NAMES = {
    "checking_account": "Checking Account", "duration": "Loan Duration",
    "credit_history": "Credit History", "purpose": "Loan Purpose",
    "credit_amount": "Credit Amount", "savings_account": "Savings Account",
    "employment": "Employment Length", "installment_rate": "Installment Rate",
    "other_debtors": "Other Debtors", "residence_since": "Residence Duration",
    "property": "Property", "age": "Age", "other_installments": "Other Installments",
    "housing": "Housing", "existing_credits": "Existing Credits", "job": "Job Type",
    "dependents": "Dependents", "telephone": "Telephone"
}

CONFIDENCE_THRESHOLD = 0.08

# Default selection for each dropdown, chosen to represent a realistic/typical
# applicant rather than whichever category happens to be listed first.
CATEGORICAL_DEFAULTS = {
    "checking_account": "A12",     # 0 to 200 DM
    "savings_account": "A61",      # Less than 100 DM
    "other_debtors": "A101",       # None
    "other_installments": "A143",  # None
    "purpose": "A43",              # Radio/television
    "employment": "A73",           # 1 to 4 years
    "job": "A173",                 # Skilled employee/official
    "housing": "A152",             # Owned
    "property": "A121",            # Real estate
    "telephone": "A192",           # Registered under customer's name
}


@app.route("/")
def index():
    return render_template(
        "index.html",
        label_maps=label_maps,
        numeric_fields=NUMERIC_FIELDS,
        feature_order=FEATURE_ORDER,
        categorical_defaults=CATEGORICAL_DEFAULTS
    )


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    try:
        raw_input = {}
        for col in FEATURE_ORDER:
            if col in label_maps:
                # categorical: frontend sends the raw code (e.g. "A11") directly
                raw_input[col] = data[col]
            else:
                raw_input[col] = float(data[col])

        encoded_row = {}
        for col in FEATURE_ORDER:
            if col in encoders:
                try:
                    encoded_row[col] = encoders[col].transform([raw_input[col]])[0]
                except ValueError:
                    label = FEATURE_DISPLAY_NAMES.get(col, col)
                    return jsonify({"error": f"'{raw_input[col]}' isn't a supported value for {label}."}), 400
            else:
                encoded_row[col] = raw_input[col]

        input_df = pd.DataFrame([encoded_row])[FEATURE_ORDER]

        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]
        decision = "Good Credit" if pred == 1 else "Bad Credit"

        confidence_gap = abs(proba[1] - 0.5)
        is_borderline = bool(confidence_gap < CONFIDENCE_THRESHOLD)
        confidence_label = "Borderline" if is_borderline else "High Confidence"

        shap_vals = explainer.shap_values(input_df)[0]
        contributions = list(zip(FEATURE_ORDER, shap_vals, input_df.iloc[0]))
        contributions.sort(key=lambda x: abs(x[1]), reverse=True)

        positive_factors = [c for c in contributions if c[1] > 0][:3]
        negative_factors = [c for c in contributions if c[1] < 0][:3]

        def factor_list_str(factors):
            return ", ".join([f[0].replace("_", " ") for f in factors]) if factors else "none identified"

        prompt = f"""A credit model classified this applicant as: {decision}
Confidence: {confidence_label}

Factors that hurt approval, strongest first: {factor_list_str(negative_factors)}
Factors that helped approval, strongest first: {factor_list_str(positive_factors)}

Write a response in EXACTLY this format, nothing more:

Primary reason: [the single strongest factor working against the decision, or if decision is Good Credit, the strongest factor supporting it]
Supporting factors: [1-2 more factors in the SAME direction as the primary reason, or "none"]
Offsetting factor: [one factor working in the OPPOSITE direction from the primary reason, or "none"]
Summary: [one plain sentence tying it together]

Do not mention SHAP, model internals, or technical terms."""

        try:
            client = OpenAI()
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )
            explanation = response.choices[0].message.content
        except Exception as e:
            explanation = f"Primary reason: unavailable\nSummary: Explanation service error: {e}"

        chart_data = [
            {
                "feature": FEATURE_DISPLAY_NAMES.get(name, name),
                "impact": round(float(val), 4)
            }
            for name, val, _ in contributions[:10]
        ]

        return jsonify({
            "decision": decision,
            "confidence": confidence_label,
            "is_borderline": is_borderline,
            "probability": round(float(proba[1]) * 100, 1),
            "explanation": explanation,
            "factors": chart_data
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5000)
