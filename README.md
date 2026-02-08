# Automated Email Classification Using GenAI-Enhanced Models

## 📌 Problem Statement
Enterprise email systems receive massive volumes of emails every day. Traditional rule-based filtering systems fail to handle evolving language patterns, while classical machine learning models struggle with adaptability and scalability.  
This project implements a **multi-class automated email classification system** using **traditional NLP + GenAI-enhanced representations**.

---

## 🎯 Objective
To automatically classify incoming emails into:
- **Spam**
- **Promotions**
- **Support**
- **Personal**

using TF-IDF, Transformer embeddings, and machine learning classifiers.

---

## 🧠 Approach

### 1. Dataset
- Based on **Enron-style email dataset**
- Columns:
  - `label`:  
    - `1` → Spam  
    - `0` → Non-Spam
  - `text`: Email content
- Non-spam emails are further categorized using keyword-based heuristics.

---

### 2. Data Preprocessing
- Lowercasing
- Null value removal
- Noise handling (URLs, numbers, tokens)
- Label transformation (Binary → Multi-class)

---

### 3. Feature Engineering
- **TF-IDF Vectorization** (Baseline)
- **Transformer-based Embeddings**  
  - Model: `sentence-transformers/all-MiniLM-L6-v2`

---

### 4. Models Used
- Logistic Regression (TF-IDF features)
- Logistic Regression (Transformer embeddings)

---

### 5. Evaluation Metrics
- Confusion Matrix
- Precision
- Recall
- F1-Score

---

### 6. GenAI Enhancement
Transformer embeddings capture **semantic meaning** and **contextual relationships**, significantly improving classification robustness compared to TF-IDF.

---

### 7. Real-Time Prediction
The system supports real-time email classification using the trained GenAI-enhanced model.

---

## 🛠️ Tech Stack
- Python
- Google Colab
- Scikit-learn
- Hugging Face Transformers
- fastText
- Pandas
- NumPy
- PyTorch

---

## 🚀 How to Run (Google Colab)

1. Upload the dataset (`emails.csv`)
2. Install dependencies
3. Run notebook cells sequentially
4. Train baseline and GenAI models
5. Test real-time email classification

---
