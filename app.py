from flask import Flask, render_template, request, send_file, jsonify
import pandas as pd
import numpy as np
import os
import json
from io import BytesIO
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load semantic search model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Globals to store state
df_cleaned = None
embeddings = None
search_column = None

# ---------- Cleaning Function ----------
def clean_dataset(df):
    report = {}

    # Initial stats
    report["initial_shape"] = f"{df.shape[0]} x {df.shape[1]}"
    report["missing_values_before"] = df.isnull().sum().to_dict()

    # Remove duplicates
    df = df.drop_duplicates()

    # Handle missing values
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.number):
            df[col] = df[col].fillna(df[col].mean())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

    # Try to standardize dates
    for col in df.columns:
        try:
            df[col] = pd.to_datetime(df[col])
        except:
            pass

    # Final stats
    report["final_shape"] = f"{df.shape[0]} x {df.shape[1]}"
    report["missing_values_after"] = df.isnull().sum().to_dict()

    return df, report

# ---------- Routes ----------
@app.route("/", methods=["GET", "POST"])
def index():
    global df_cleaned, embeddings, search_column

    if request.method == "POST":
        file = request.files.get("file")
        search_column = request.form.get("search_column", "").strip()

        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            # Load dataset
            if filepath.endswith(".csv"):
                df = pd.read_csv(filepath)
            else:
                df = pd.read_excel(filepath)

            # Clean dataset
            df_cleaned, report = clean_dataset(df)

            # Create embeddings for search
            embeddings = None
            if search_column in df_cleaned.columns:
                texts = df_cleaned[search_column].astype(str).tolist()
                embeddings = model.encode(texts)

            return render_template(
                "index.html",
                 columns=df_cleaned.columns,
                 report=report,
                 report_before=json.dumps(report["missing_values_before"]),
                 report_after=json.dumps(report["missing_values_after"]),
                 uploaded=True
            )
     


    # Default GET request
    return render_template("index.html", uploaded=False)

@app.route("/search", methods=["POST"])
def search():
    global embeddings, df_cleaned, search_column
    query = request.form.get("query", "").strip()
    if embeddings is None or search_column not in df_cleaned.columns:
        return jsonify({"results": []})

    query_emb = model.encode([query])
    scores = cosine_similarity(query_emb, embeddings)[0]
    top_idx = np.argsort(scores)[::-1][:5]
    results = df_cleaned.iloc[top_idx][search_column].astype(str).tolist()
    return jsonify({"results": results})

@app.route("/download")
def download():
    global df_cleaned
    output = BytesIO()
    df_cleaned.to_csv(output, index=False)
    output.seek(0)
    return send_file(output, as_attachment=True, download_name="cleaned_dataset.csv")

if __name__ == "__main__":
    app.run(debug=True)
