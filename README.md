

# 🧬 DNA Coding vs Non-Coding Classifier 

A **Streamlit web application** for predicting whether DNA sequences are **coding or non-coding**.
The app supports **manual sequence input**, **FASTA/TXT uploads**, and **training on custom datasets**.
It provides **real-time predictions, interactive genome fraction analysis, and dynamic visualizations**.

---

## ⚡ Features (Interactive)

* **Upload your dataset** for model training in real-time
* **Manual DNA sequence input** with instant predictions
* **FASTA/TXT file upload** for batch analysis
* **Dynamic k-mer size selection** for feature extraction
* **Genome fraction analysis** with adjustable window size
* **Confidence threshold slider** for coding classification
* **Download predictions as CSV**
* **Interactive bar charts** for coding vs non-coding fractions
* **Interactive pie chart** for dataset class distribution
* **Display model accuracy and classification report**
* **Multi-sequence visualization side-by-side**

---

## 🛠 Installation

```bash
git clone https://github.com/yourusername/dna-coding-noncoding-classifier.git
cd dna-coding-noncoding-classifier
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
pip install -r requirements.txt
streamlit run app.py
```

**Example `requirements.txt`:**

```
streamlit==1.25.0
pandas==2.1.1
scikit-learn==1.3.2
plotly==6.1.0
```

---

## 🧩 How to Use (Interactive Workflow)

1. **Train Model (Optional)**

   * Upload dataset (`.csv` or `.txt`)
   * Select **sequence column** and **label column**
   * Adjust **k-mer size slider**
   * Click **Train Model** → Accuracy and classification report appear instantly

2. **Predict Sequences**

   * Upload **FASTA/TXT** file or paste sequence manually
   * Adjust **window size** for genome fraction analysis
   * Use **confidence threshold slider**
   * Click **Predict & Analyze** → Interactive table + bar chart

3. **Visualize Results**

   * Interactive bar chart for **Coding vs Non-Coding fraction per sequence**
   * Pie chart for **dataset class distribution**
   * Download CSV of predictions

---

## 📂 Dataset Format

| Sequence      | Label      |
| ------------- | ---------- |
| ATGCGTACGATCG | Coding     |
| CGTACGATCGTAC | Non-Coding |
| ...           | ...        |

* Exactly **2 classes** required for coding/non-coding
* Any **tabular dataset (.csv or .txt)** can be used

---

## 🔗 Repository Structure

```
dna-coding-noncoding-classifier/
│
├── app.py                  # Streamlit interactive application
├── requirements.txt        # Dependencies
├── datasets/               # Example datasets
│   └── example_dataset.csv
├── rf_model.pkl            # Saved Random Forest model (after training)
├── vectorizer.pkl          # Saved CountVectorizer (after training)

```

---



