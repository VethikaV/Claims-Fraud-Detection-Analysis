from flask import Flask, render_template, request, redirect, url_for, session, flash
import pandas as pd
import joblib
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = "supersecretkey"  # Needed for session handling

# Load model
model = joblib.load("./model/final_model.pkl")

FEATURE_COLUMNS = [
    'DiagnosisDiversity_perProvider',
    'ProcedureDiversity_perProvider',
    'NumBeneficiaries_perProvider',
    'AvgClaimAmount_perProvider',
    'State_freq',
    'County_freq',
    'LengthOfStay',
    'ClmDiagnosisCode_10_freq',
    'ClmDiagnosisCode_8_freq',
    'TotalClaimAmount_perBene',
    'Race_freq',
    'ClmDiagnosisCode_7_freq',
    'AvgChronicCond_perBene',
    'ClmProcedureCode_3_freq',
    'ClmProcedureCode_1_freq',
    'ClmDiagnosisCode_9_freq',
    'ClmDiagnosisCode_2_freq',
    'TotalChronicCond_perBene',
    'AvgClaimAmount_perBene',
    'ClmDiagnosisCode_6_freq'
]

# ---------------- ROUTES ----------------
@app.route("/", methods=["GET", "POST"])
def login():
    """Login page"""
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        # ✅ Accept ANY credentials (not restricted to USER_CREDENTIALS)
        if username and password:
            session["user"] = username
            return redirect(url_for("home"))
        else:
            flash("Please enter both username and password", "danger")

    return render_template("login.html")



@app.route("/home")
def home():
    """Home page after login"""
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("home.html", username=session.get("user"))


@app.route("/form")
def form():
    """Form page"""
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("form_with_gauge.html", feature_columns=FEATURE_COLUMNS)


@app.route("/predict", methods=["POST"])
def predict():
    """Prediction route for manual entry"""
    if "user" not in session:
        return redirect(url_for("login"))

    # --- Safety: check if file was also uploaded ---
    if "file" in request.files and request.files["file"].filename != "":
        flash("Please use either Manual Entry OR Excel Upload, not both.", "warning")
        return redirect(url_for("form"))

    try:
        # Build input data in correct column order
        input_data = [float(request.form.get(col, 0)) for col in FEATURE_COLUMNS]

        claim_id = request.form.get("ClaimID", "Manual Entry")
        amount = request.form.get("ClaimAmount_sum", 0)

        # Predict
        prediction = model.predict([input_data])[0]
        probability = model.predict_proba([input_data])[0][1] * 100

        result_text = "Fraudulent" if prediction == 1 else "Not Fraudulent"

        return render_template(
            "result.html",
            prediction=result_text,
            probability=round(probability, 2),
            claim_id=claim_id,
            amount=amount,
            username=session.get("user", "Guest")
        )
    except Exception as e:
        flash(f"Error in manual input: {str(e)}", "danger")
        return redirect(url_for("form"))


# ---------------- Excel Upload Route ----------------
UPLOAD_FOLDER = "./uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/predict_excel", methods=["POST"])
def predict_excel():
    """Handle Excel upload and prediction"""
    if "user" not in session:
        return redirect(url_for("login"))

    # --- Safety: prevent using both manual form + Excel ---
    if any(request.form.get(col) for col in FEATURE_COLUMNS):
        flash("Please use either Manual Entry OR Excel Upload, not both.", "warning")
        return redirect(url_for("form"))

    if "file" not in request.files:
        flash("No file uploaded", "danger")
        return redirect(url_for("form"))

    file = request.files["file"]
    if file.filename == "":
        flash("No file selected", "danger")
        return redirect(url_for("form"))

    try:
        # Save file locally
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(file.filename))
        file.save(filepath)

        # Read Excel
        df = pd.read_excel(filepath, engine="openpyxl")

        # ✅ Clean column names
        df.columns = df.columns.str.strip()
        df.columns = df.columns.str.replace(r"\s+", "_", regex=True)

        # ✅ Validate (ignore order, just check required cols exist)
        missing_cols = set(FEATURE_COLUMNS) - set(df.columns)
        if missing_cols:
            flash(f"Excel file is missing required columns: {', '.join(missing_cols)}", "danger")
            return redirect(url_for("form"))

        # ✅ Ensure correct order & fill missing with 0
        df = df.reindex(columns=FEATURE_COLUMNS, fill_value=0)

        # ✅ Ensure only 1 row
        if len(df) != 1:
            flash("Excel must contain exactly one row of input", "danger")
            return redirect(url_for("form"))

        # Prepare input for model
        input_data = df.values.tolist()[0]

        # Predict
        prediction = model.predict([input_data])[0]
        probability = model.predict_proba([input_data])[0][1] * 100
        result_text = "Fraudulent" if prediction == 1 else "Not Fraudulent"

        return render_template(
            "result.html",
            prediction=result_text,
            probability=round(probability, 2),
            claim_id="From Excel",
            amount=df.get("ClaimAmount_sum", [0])[0] if "ClaimAmount_sum" in df else 0,
            username=session.get("user", "Guest")
        )

    except Exception as e:
        flash(f"Error processing Excel: {str(e)}", "danger")
        return redirect(url_for("form"))



# ---------------- MAIN ----------------
if __name__ == "__main__":
    app.run(debug=True)
