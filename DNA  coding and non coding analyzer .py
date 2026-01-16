# ============================
# STREAMLIT APP: DNA CODING vs NON-CODING (UPGRADED WITH METRICS)
# ============================
import streamlit as st
import pandas as pd
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from io import StringIO
import plotly.express as px

# ============================
# FILE PATHS
# ============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, "rf_model.pkl")
VECTORIZER_FILE = os.path.join(BASE_DIR, "vectorizer.pkl")
METRICS_FILE = os.path.join(BASE_DIR, "model_metrics.pkl")  # Save metrics

# ============================
# K-MER FUNCTION
# ============================
def get_kmers(sequence, k=4):
    sequence = str(sequence).upper()
    return " ".join([sequence[i:i+k] for i in range(len(sequence)-k+1)])

# ============================
# TRAIN MODEL FUNCTION
# ============================
def train_model(df, seq_col, label_col, k=4):
    # Binary label conversion
    classes = df[label_col].unique()
    if len(classes) != 2:
        st.error("❌ Dataset must have exactly 2 classes for coding/non-coding!")
        return None, None

    positive_class = classes[0]
    df['binary_class'] = df[label_col].apply(lambda x: 1 if x == positive_class else 0)

    # k-mer feature extraction
    df['kmer_seq'] = df[seq_col].apply(lambda x: get_kmers(x, k=k))
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['kmer_seq'])
    y = df['binary_class']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    # Evaluation
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    st.success(f"✅ Model trained! Test Accuracy: {acc*100:.2f}%")
    st.text(classification_report(y_test, y_pred))

    # Save model & vectorizer & metrics
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(rf, f)
    with open(VECTORIZER_FILE, "wb") as f:
        pickle.dump(vectorizer, f)
    with open(METRICS_FILE, "wb") as f:
        pickle.dump({"accuracy": acc, "classification_report": report}, f)

    st.info("💾 Model, vectorizer, and metrics saved for future use!")

    return rf, vectorizer, acc, report

# ============================
# LOAD MODEL
# ============================
def load_model():
    if os.path.exists(MODEL_FILE) and os.path.exists(VECTORIZER_FILE):
        with open(MODEL_FILE, "rb") as f:
            rf_model = pickle.load(f)
        with open(VECTORIZER_FILE, "rb") as f:
            vectorizer = pickle.load(f)
        st.success("✅ Model and vectorizer loaded successfully!")
        # Load metrics if exists
        if os.path.exists(METRICS_FILE):
            with open(METRICS_FILE, "rb") as f:
                metrics = pickle.load(f)
            st.info(f"Model Accuracy: {metrics['accuracy']*100:.2f}%")
            st.text(classification_report(metrics['classification_report']))
        return rf_model, vectorizer
    else:
        return None, None

# ============================
# PREDICTION FUNCTION
# ============================
def predict_dna(rf_model, vectorizer, dna_seq):
    kmer_seq = get_kmers(dna_seq)
    X_input = vectorizer.transform([kmer_seq])
    probs = rf_model.predict_proba(X_input)[0]  # [Non-coding, Coding]
    prediction = "Coding" if probs[1] > probs[0] else "Non-Coding"
    return prediction, probs

# ============================
# GENOME FRACTION
# ============================
def genome_fraction(rf_model, vectorizer, seq, window=50):
    seq = seq.upper()
    coding_count = 0
    noncoding_count = 0

    if len(seq) < window:
        return 0,0

    for i in range(0, len(seq)-window+1, window):
        fragment = seq[i:i+window]
        prediction, probs = predict_dna(rf_model, vectorizer, fragment)
        if prediction == "Coding":
            coding_count +=1
        else:
            noncoding_count +=1

    total = coding_count + noncoding_count
    coding_perc = (coding_count/total)*100 if total>0 else 0
    noncoding_perc = (noncoding_count/total)*100 if total>0 else 0
    return coding_perc, noncoding_perc

# ============================
# FASTA PARSER
# ============================
def parse_fasta(file_content):
    sequences = {}
    current_seq_name = ""
    for line in file_content.splitlines():
        line = line.strip()
        if line.startswith(">"):
            current_seq_name = line[1:].strip()
            sequences[current_seq_name] = ""
        else:
            sequences[current_seq_name] += line
    return sequences

# ============================
# STREAMLIT UI
# ============================
st.set_page_config(page_title="DNA Coding vs Non-Coding", layout="wide")
st.title("🧬 DNA Coding vs Non-Coding Classifier (With Metrics)")

st.markdown("""
Upload your **dataset** for training, or **FASTA/TXT** for prediction.  
You can also paste sequences manually. Model accuracy and classification report are displayed.
""")

# ----------------------------
# Dataset upload for training
# ----------------------------
dataset_file = st.file_uploader("Upload dataset (.csv or .txt) for training", type=["csv","txt"])
rf_model, vectorizer = load_model()

if dataset_file is not None:
    df = pd.read_csv(dataset_file, sep=None, engine='python')
    st.success(f"✅ Dataset loaded: {df.shape[0]} sequences, {df.shape[1]} columns")
    st.dataframe(df.head())

    seq_col = st.selectbox("Select sequence column:", df.columns)
    label_col = st.selectbox("Select label column (Coding/Non-Coding):", df.columns)
    kmer_size = st.slider("Select k-mer size:", min_value=3, max_value=10, value=4)

    if st.button("Train Model on Uploaded Dataset"):
        rf_model, vectorizer, acc, report = train_model(df, seq_col, label_col, k=kmer_size)
        st.info(f"Model Accuracy: {acc*100:.2f}%")
        st.text(classification_report(report))

# ----------------------------
# File upload for prediction
# ----------------------------
st.subheader("Predict Coding vs Non-Coding")
uploaded_file = st.file_uploader("Upload FASTA/TXT sequences for prediction", type=["fasta","fa","txt"])

sequences = {}
if uploaded_file is not None:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    content = stringio.read()
    if uploaded_file.name.endswith(("fasta","fa")):
        sequences = parse_fasta(content)
    else:
        lines = content.strip().split("\n")
        sequences = {f"Seq_{i+1}": line.strip() for i, line in enumerate(lines)}
    st.success(f"✅ {len(sequences)} sequences loaded from file.")

manual_input = st.text_area("Or paste sequence manually:", height=150)
if manual_input.strip() != "":
    sequences["Manual_Sequence"] = manual_input.strip()

window_size = st.number_input("Window size for genome fraction analysis:", min_value=10, max_value=1000, value=50)
confidence_threshold = st.slider("Confidence threshold for Coding classification (%)", min_value=50, max_value=100, value=50)

# ----------------------------
# Predict & Analyze
# ----------------------------
if st.button("Predict & Analyze Sequences"):
    if len(sequences) == 0:
        st.warning("⚠️ No sequences provided.")
    elif rf_model is None or vectorizer is None:
        st.error("❌ Model not trained or loaded! Please train or load a model first.")
    else:
        results_list = []
        for name, seq in sequences.items():
            prediction, probs = predict_dna(rf_model, vectorizer, seq)
            if probs[1]*100 < confidence_threshold:
                prediction = "Non-Coding"
            coding_perc, noncoding_perc = genome_fraction(rf_model, vectorizer, seq, window=window_size)
            results_list.append({
                "Sequence": name,
                "Prediction": prediction,
                "Probability Coding (%)": round(probs[1]*100,2),
                "Probability Non-Coding (%)": round(probs[0]*100,2),
                "Coding Fraction (%)": round(coding_perc,2),
                "Non-Coding Fraction (%)": round(noncoding_perc,2)
            })

        results_df = pd.DataFrame(results_list)
        st.subheader("Prediction Results Table")
        st.dataframe(results_df)

        # Download button
        csv = results_df.to_csv(index=False).encode()
        st.download_button("Download Results as CSV", data=csv, file_name="dna_prediction_results.csv", mime="text/csv")

        # Bar chart
        st.subheader("Coding vs Non-Coding Fraction per Sequence")
        fig = px.bar(results_df, x="Sequence", y=["Coding Fraction (%)","Non-Coding Fraction (%)"],
                     barmode="group", title="Coding vs Non-Coding Fraction")
        st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# Dataset Class Distribution
# ----------------------------
if dataset_file is not None:
    st.subheader("Dataset Class Distribution")
    if label_col in df.columns:
        counts = df[label_col].value_counts().reset_index()
        counts.columns = ["Class","Count"]
        fig2 = px.pie(counts, names="Class", values="Count", title="Coding vs Non-Coding Class Distribution")
        st.plotly_chart(fig2, use_container_width=True)
