# 💎 Diamond Price Prediction

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-yellow.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-Latest-green.svg)
![License](https://img.shields.io/badge/License-MIT-red.svg)

**A comprehensive machine learning project for predicting diamond prices using multiple algorithms including XGBoost and Deep Neural Networks**

[English](#english) 

</div>

---

<a name="english"></a>
## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Models Performance](#models-performance)
- [Results](#results)
- [Usage](#usage)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## 🎯 Project Overview

This project implements a complete end-to-end machine learning pipeline for predicting diamond prices based on their physical characteristics. The project demonstrates advanced data preprocessing, feature engineering, and model comparison techniques, ultimately achieving **99.08% accuracy (R²)** using XGBoost.

### 🎓 Learning Objectives

- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering and transformation
- Outlier detection and handling
- Encoding categorical variables
- Building preprocessing pipelines
- Training and comparing multiple ML models
- Deep Neural Network implementation
- Model evaluation and selection

---

## ✨ Key Features

- ✅ **Comprehensive Data Cleaning**: Duplicate removal, outlier handling using IQR method
- ✅ **Advanced EDA**: Missing data visualization, correlation analysis, distribution plots
- ✅ **Feature Engineering**: Volume calculation, log transformations
- ✅ **Multiple ML Models**: 7 different algorithms compared
- ✅ **Deep Learning**: Custom DNN with dropout and early stopping
- ✅ **Production-Ready Pipeline**: Automated preprocessing with scikit-learn pipelines
- ✅ **Model Persistence**: Save and load trained models
- ✅ **Detailed Visualizations**: 10+ professional charts and plots

---

## 📊 Dataset

**Source**: [Seaborn Diamonds Dataset](https://raw.githubusercontent.com/mwaskom/seaborn-data/master/diamonds.csv)

### Features:
- **Numerical**: `carat`, `depth`, `table`, `x`, `y`, `z`
- **Categorical**: `cut`, `color`, `clarity`
- **Target**: `price` (USD)

### Dataset Statistics:
- **Total Records**: 53,940 diamonds
- **Features**: 9 (6 numerical + 3 categorical)
- **No Missing Values**: Complete dataset after cleaning

---

## 🚀 Installation

### Prerequisites
```bash
Python 3.8+
pip package manager
```

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/diamond-price-prediction.git
cd diamond-price-prediction
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install Dependencies

To install all required libraries, run:

```bash
pip install pandas>=1.3.0 numpy>=1.21.0 matplotlib>=3.4.0 seaborn>=0.11.0 scikit-learn>=1.0.0 xgboost>=1.5.0 tensorflow>=2.8.0 missingno>=0.5.0 scipy>=1.7.0 joblib>=1.1.0


## 📁 Project Structure

```
diamond-price-prediction/
│
├── lumiere.ipynb                      # Main Jupyter notebook
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
│
├── models/                            # Saved models
│   ├── diamond_price_xgb_model.json  # XGBoost model
│   ├── full_pipeline.pkl             # Preprocessing pipeline
│   └── dnn_model.h5                  # Deep Neural Network model
│
├── data/                              # Data directory
│   └── diamonds.csv                  # Raw dataset (downloaded)
│
├── notebooks/                         # Additional notebooks
│   ├── 01_eda.ipynb                 # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb       # Data preprocessing
│   └── 03_modeling.ipynb            # Model training
│
├── src/                              # Source code
│   ├── preprocessing.py             # Data preprocessing functions
│   ├── modeling.py                  # Model training functions
│   └── utils.py                     # Utility functions
│
└── reports/                          # Generated reports
    ├── figures/                     # Saved visualizations
    └── model_comparison.csv         # Model metrics
```

---

## 🔬 Methodology

### 1️⃣ Data Loading & Overview
- Load dataset from remote source
- Inspect structure, dimensions, and data types
- Generate descriptive statistics

### 2️⃣ Data Quality Assessment & Cleaning
- **Missing Values**: None detected
- **Duplicates**: 146 duplicate rows removed
- **Outliers**: Handled using IQR capping method for 6 numerical features

### 3️⃣ Exploratory Data Analysis
- Visualized missing data patterns
- Created correlation heatmap (price correlates highly with carat: 0.92)
- Generated scatter matrix for key features
- Analyzed distributions with histograms

### 4️⃣ Feature Engineering
- **New Feature**: `xyz` (volume = x × y × z)
- **Transformation**: Log transformation of `price` to reduce skewness
- **Feature Removal**: Dropped `x`, `y`, `z` after creating volume feature

### 5️⃣ Categorical Data Encoding
- **Ordinal Encoding** for ordered categories:
  - `cut`: Fair < Good < Very Good < Premium < Ideal
  - `color`: J < I < H < G < F < E < D
  - `clarity`: I1 < SI2 < SI1 < VS2 < VS1 < VVS2 < VVS1 < IF

### 6️⃣ Preprocessing Pipeline
- **Train-Test Split**: 80-20 stratified split
- **Numerical Pipeline**: StandardScaler
- **Categorical Pipeline**: OrdinalEncoder with unknown handling
- **Full Pipeline**: ColumnTransformer combining both pipelines

### 7️⃣ Model Training & Comparison
Trained 7 different models with hyperparameter tuning:
1. Linear Regression
2. Ridge Regression
3. Lasso Regression
4. Decision Tree
5. Random Forest
6. Gradient Boosting
7. **XGBoost** ⭐

### 8️⃣ Deep Neural Network
- **Architecture**: Input → 64 → 32 → 1
- **Activation**: ReLU
- **Optimizer**: Adam (lr=0.001)
- **Regularization**: Dropout, EarlyStopping
- **Training**: 100 epochs, batch_size=32

---

## 📈 Models Performance

### Test Set Results (Ranked by R²)

| Rank | Model | Test R² | Test RMSE | Test MAE | Overfitting |
|------|-------|---------|-----------|----------|-------------|
| 🥇 | **XGBoost** | **0.9909** | **0.0964** | **0.0656** | **0.0019** |
| 🥈 | Gradient Boosting | 0.9908 | 0.0970 | 0.0661 | 0.0009 |
| 🥉 | Random Forest | 0.9895 | 0.1035 | 0.0702 | 0.0058 |
| 4️⃣ | DNN | 0.9892 | 0.1050 | 0.0710 | 0.0009 |
| 5️⃣ | Decision Tree | 0.9748 | 0.1604 | 0.1084 | 0.0001 |
| 6️⃣ | Ridge | 0.9148 | 0.2951 | 0.2040 | 0.0010 |
| 7️⃣ | Lasso | 0.9145 | 0.2956 | 0.2044 | 0.0009 |
| 8️⃣ | Linear Regression | 0.9144 | 0.2958 | 0.2046 | 0.0010 |

### Key Insights:
- 🎯 **Tree-based models** significantly outperform linear models
- 🎯 **XGBoost** achieves the best balance of accuracy and generalization
- 🎯 **Very low overfitting** across all top models (<0.006)
- 🎯 **Non-linear relationships** are crucial for this dataset

---

## 🏆 Results

### XGBoost - Best Model

```
✅ Test R² Score: 0.9909 (99.09% variance explained)
✅ Test RMSE: 0.0964 (log scale)
✅ Test MAE: 0.0656 (log scale)
✅ Overfitting: 0.0019 (excellent generalization)
```

### Prediction Example:
```python
Actual Original Price: $8,314
Predicted Original Price: $8,267
Absolute Error: $47 (0.56% error)
```

### Model Comparison Visualization:
The project includes 4 comprehensive plots:
1. **R² Score Comparison** - Model ranking
2. **RMSE Comparison** - Error magnitude
3. **Train vs Test R²** - Overfitting detection
4. **Overfitting Analysis** - Generalization quality

---

## 💻 Usage

### Training Models

```python
# Load and preprocess data
from src.preprocessing import load_and_preprocess_data
X_train, X_test, y_train, y_test = load_and_preprocess_data()

# Train models
from src.modeling import train_all_models
results = train_all_models(X_train, y_train, X_test, y_test)

# Get best model
best_model = results['best_model']
```

### Making Predictions

```python
import joblib
import xgboost as xgb
import numpy as np

# Load saved models
pipeline = joblib.load('models/full_pipeline.pkl')
xgb_model = xgb.Booster()
xgb_model.load_model('models/diamond_price_xgb_model.json')

# Prepare new diamond data
new_diamond = {
    'carat': 1.5,
    'cut': 'Ideal',
    'color': 'E',
    'clarity': 'VS1',
    'depth': 62.0,
    'table': 57.0,
    'xyz': 7.5 * 7.5 * 4.5
}

# Preprocess and predict
X_new = pipeline.transform(pd.DataFrame([new_diamond]))
log_price = xgb_model.predict(xgb.DMatrix(X_new))
predicted_price = np.exp(log_price[0]) - 1

print(f"Predicted Diamond Price: ${predicted_price:,.2f}")
```

### Running the Notebook

```bash
jupyter notebook lumiere.ipynb
```

---

## 🛠 Technologies Used

### Core Libraries:
- **Data Analysis**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, Missingno
- **Machine Learning**: scikit-learn, XGBoost
- **Deep Learning**: TensorFlow/Keras
- **Statistical Analysis**: SciPy

### Key Techniques:
- ✅ IQR Method for outlier detection
- ✅ Log transformation for skewed data
- ✅ Ordinal encoding for categorical features
- ✅ StandardScaler for numerical features
- ✅ ColumnTransformer for pipeline creation
- ✅ GridSearchCV for hyperparameter tuning
- ✅ EarlyStopping for neural network training



---


## 📧 Contact

**Project Author**: Lumiere Team

- 📧 Email: 0ibrahim0amin@gmail.com
- 🐙 GitHub: [@ibrahim-amin0](https://github.com/ibrahim-amin0)
- 💼 LinkedIn: [Ibrahim Amin](www.linkedin.com/in/ibrahim-amin-aie0101010101)

---

## 🙏 Acknowledgments

- Dataset: Seaborn library (original source: R's ggplot2)
- Inspiration: Kaggle Diamond Price Prediction competitions
- Community: scikit-learn, XGBoost, and TensorFlow documentation


### ⭐ If you like this project, don't forget to give it a star!

**Made with ❤️ by Lumiere Team**

</div>
