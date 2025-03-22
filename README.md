# 🤖 Machine Learning Projects

Welcome to my collection of machine learning assignments, developed during the course of my ML studies.  
Each assignment explores different machine learning techniques — from unsupervised learning to advanced ensemble methods — with clear implementations and real-world data challenges.

---

## 📂 Assignments Overview

### 📁 Assignment 1: Clustering & Dimensionality Reduction (Unsupervised Learning)

**Objective**: Discover natural groupings in data without labels.

**What’s Inside**:
- 🧪 **K-Means Clustering** – applied with multiple `k` values
- 📐 **Principal Component Analysis (PCA)** – used to reduce dimensionality for better visualization
- 📊 **Silhouette Score** – used to evaluate clustering quality
- 🗂️ Dataset: Generated synthetic datasets for clustering (2D, 3D)

📌 You’ll learn how to:
- Choose the number of clusters (`k`)
- Visualize clusters in 2D and 3D
- Interpret PCA-transformed data

---

### 📁 Assignment 2: Supervised Learning – Linear Regression, Naive Bayes & Decision Trees

**Objective**: Train models to predict numeric and categorical outcomes using labeled data.

**Key Components**:
- 📈 **Linear Regression**: Applied to predict continuous values with performance metrics like MSE and R²
- 📊 **Naive Bayes**: Used for classification with text features (TF-IDF preprocessing included)
- 🌳 **Decision Trees**: Built and visualized for both regression and classification tasks

**Extras**:
- Feature encoding (LabelEncoder, TF-IDF)
- Train-test split, cross-validation
- Evaluation metrics: accuracy, confusion matrix, precision, recall

📌 Great for understanding:
- Difference between regression & classification
- Model evaluation strategies
- Handling both numeric and text data

---

### 📁 Assignment 3: Data Preprocessing & AdaBoost (Ensemble Learning)

**Objective**: Prepare messy data and improve predictions using ensemble methods.

**Steps Taken**:
- 🧹 **Data Cleaning**:
  - Null values handled via:
    - KNN imputation for numerical features
    - Mode imputation for categorical features
- 🏗️ **Feature Encoding & Scaling** (OneHotEncoder + StandardScaler)
- 🚀 **AdaBoost**:
  - Implemented **from scratch**
  - Used decision stumps as weak learners
  - Evaluated performance across boosting iterations

📌 Key Takeaways:
- How boosting improves model performance
- How to write your own ensemble algorithm
- Importance of good preprocessing in real-world data

---

## 🧠 Tools & Libraries Used

- **Python 3.x**
- **Jupyter Notebook**
- **scikit-learn** – Models, metrics, preprocessing
- **Pandas / NumPy** – Data manipulation
- **Matplotlib / Seaborn** – Visualizations

---

## 🚀 Getting Started

### 📦 Requirements

Install the required libraries:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
