# Duplicate Question Pair Detection (NLP → Transformers)

## 📌 Overview

This project focuses on **Duplicate Question Pair Detection**, a core NLP problem where the goal is to determine whether two questions are semantically equivalent. Such systems are widely used in platforms like **Quora, Stack Overflow, customer support forums, and search engines** to reduce redundancy and improve information retrieval.

What makes this project strong is its **comparative and progressive approach**:

* Starting with **basic NLP features** (baseline)
* Improving performance using **advanced feature engineering**
* Finally leveraging **Transformer / LLM-based semantic models**

This progression clearly demonstrates the evolution from **lexical similarity** to **deep semantic understanding**.

---

## 🧠 Problem Statement

Given a pair of questions `(q1, q2)`, predict whether they are **duplicates** (i.e., convey the same meaning).

* **Input:** Two natural language questions
* **Output:** Binary label

  * `1` → Duplicate
  * `0` → Not Duplicate

---

## 📂 Project Structure

```
Duplicate_Question_Pair/
│
├── without_advanced_features.ipynb   # Baseline NLP features
├── With_Advanced_Features.ipynb      # Advanced feature engineering
├── withllm.py                        # Transformer / LLM-based model
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

The project is based on the **Quora Question Pairs** dataset, which contains labeled pairs of questions indicating whether they are semantically duplicate.

* `question1`, `question2`: Text inputs
* `is_duplicate`: Target label

The dataset is preprocessed and split into training and evaluation sets with appropriate handling of class imbalance.

---

## 🔍 Methodology (Comparative Approach)

### 1️⃣ Baseline Model — Basic NLP Features

**File:** `without_advanced_features.ipynb`

This notebook establishes a **simple and interpretable baseline** using classical NLP techniques.

**Key Steps:**

* Text preprocessing: lowercasing, tokenization, stopword removal
* Feature extraction:

  * Question length and length difference
  * Common word count and ratios
  * Token overlap metrics
* Traditional ML models trained on handcrafted features

**Insight:**
While simple and interpretable, these features struggle to capture paraphrases and semantic similarity.

---

### 2️⃣ Enhanced Model — Advanced NLP Feature Engineering

**File:** `With_Advanced_Features.ipynb`

This stage improves upon the baseline by enriching the feature space with more expressive similarity signals.

**Advanced Features Include:**

* TF-IDF vector representations
* Fuzzy string matching scores
* Character-level similarity metrics
* Semantic overlap and ratio-based features

**Outcome:**

* Improved performance over the baseline model
* Better handling of lexical variations
* Clear demonstration of the impact of feature engineering

**Limitation:**
Despite improvements, these methods still rely heavily on surface-level text similarity.

---

### 3️⃣ Advanced Model — Transformer / LLM-Based Semantic Similarity

**File:** `withllm.py`

This module implements a **deep learning–based solution** using pre-trained Transformer models to capture contextual meaning.

**Key Highlights:**

* Uses Transformer-based sentence embeddings (e.g., Sentence-BERT style models)
* Captures deep semantic relationships beyond exact word overlap
* Computes similarity scores and applies threshold-based classification
* Structured as a clean, inference-ready Python pipeline

**Advantages:**

* Strong semantic understanding
* Robust to paraphrasing and rewording
* Suitable for real-world deployment scenarios

---

## 🔄 Comparative Summary

| Approach           | Strengths                   | Limitations           |
| ------------------ | --------------------------- | --------------------- |
| Basic NLP Features | Simple, fast, interpretable | Fails on paraphrases  |
| Advanced Features  | Better lexical coverage     | Manual feature effort |
| Transformer Models | Deep semantic understanding | Higher compute cost   |

This comparison justifies the transition from classical NLP methods to modern transformer-based approaches for production-grade systems.

---

## 🛠️ Tech Stack

* **Programming Language:** Python
* **Libraries & Frameworks:**

  * Scikit-learn
  * Hugging Face Transformers / Sentence-Transformers
  * PyTorch
  * NLTK / spaCy (for preprocessing)
* **Modeling:** Classical ML + Transformer-based embeddings

---

## 🚀 How to Run

1. Clone the repository

   ```bash
   git clone https://github.com/student-ChestaVashishtha/Duplicate_Question_Pair.git
   cd Duplicate_Question_Pair
   ```

2. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```

3. Run transformer-based model

   ```bash
   python withllm.py
   ```

4. Open notebooks for feature-based experiments

   ```bash
   jupyter notebook
   ```

---

## 📈 Key Learnings

* Classical NLP methods provide interpretability but limited semantic power
* Feature engineering can significantly improve traditional models
* Transformer-based models outperform feature-based approaches for semantic tasks
* Model selection should balance performance, interpretability, and computational cost

---

## 📌 Applications

* Duplicate detection in Q&A platforms
* FAQ deduplication systems
* Search query normalization
* Chatbots and conversational AI

---

## 🔮 Future Enhancements

* Add quantitative performance comparison (Accuracy, F1-score)
* Deploy model using FastAPI or Streamlit
* Extend to multilingual duplicate detection
* Integrate approximate nearest neighbor search for large-scale retrieval

---

## 👤 Author

**Chesta Vashishtha**
Machine Learning & NLP Enthusiast | Transformer-based Models

---

⭐ If you find this project useful, feel free to star the repository!
