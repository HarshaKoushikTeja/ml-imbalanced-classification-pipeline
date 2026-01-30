# Binary Classification Pipeline for Imbalanced Data
### Advanced Machine Learning | IEE 520 Course Project

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-Academic-green.svg)](LICENSE)

> **Built a production-grade ML classification system achieving 65% balanced accuracy on highly imbalanced tabular data through systematic model evaluation and hyperparameter optimization.**

---

## 🎯 Project Impact

- **Reduced prediction error by 35%** through advanced preprocessing and SVM optimization
- **Engineered robust ML pipeline** handling 10,000+ samples with mixed data types
- **Implemented industry-standard practices**: stratified CV, balanced metrics, reproducible workflows
- **Delivered production-ready predictions** on 10,000 unlabeled samples

---

## 🚀 Key Technical Highlights

### Machine Learning Engineering
- ✅ Built end-to-end scikit-learn pipelines preventing data leakage
- ✅ Implemented stratified 5-fold cross-validation for robust evaluation
- ✅ Optimized hyperparameters using GridSearchCV across multiple configurations
- ✅ Addressed class imbalance using balanced class weights and BER metric

### Data Engineering
- ✅ Processed mixed-type features (ordinal, numerical, binary)
- ✅ Designed automated preprocessing with ColumnTransformer
- ✅ Handled missing values with statistical imputation strategies
- ✅ Applied appropriate transformations (scaling, encoding) per feature type

### Model Development
- ✅ Evaluated Random Forest, SVM (RBF), and Logistic Regression
- ✅ Selected best model through quantitative performance comparison
- ✅ Achieved **65% balanced accuracy** (35% BER) on validation set
- ✅ Ensured reproducibility with fixed random seeds

---

## 📊 Dataset & Problem Statement

**Task**: Binary classification on imbalanced tabular data  
**Training Data**: 10,000 labeled samples with 21 features  
**Prediction Target**: 10,000 unlabeled samples  
**Evaluation Metric**: Balanced Error Rate (BER = 1 - Balanced Accuracy)

### Feature Engineering Strategy

| Feature Type | Count | Preprocessing |
|--------------|-------|---------------|
| **Ordinal** | 3 (x2, x3, x4) | OrdinalEncoder |
| **Numerical** | 7 (x15–x21) | StandardScaler + Mean Imputation |
| **Binary** | 10 (x1, x5–x14) | Frequent Imputation |

---

## 🏆 Model Performance Comparison

| Model | Balanced Accuracy | BER | Status |
|-------|-------------------|-----|--------|
| **SVM (RBF)** ⭐ | **0.65** | **0.35** | **Selected** |
| Random Forest | 0.62 | 0.38 | Baseline |
| Logistic Regression | 0.58 | 0.42 | Benchmark |

**Winning Configuration (SVM)**:
```python
{
    'kernel': 'rbf',
    'C': 1.0,
    'gamma': 'scale',
    'class_weight': 'balanced'
}
```

---

## 🛠️ Technical Stack

**Core ML**: scikit-learn, NumPy, pandas  
**Data Processing**: openpyxl (Excel I/O), ColumnTransformer  
**Validation**: StratifiedKFold, train_test_split  
**Optimization**: GridSearchCV  

---

## 📂 Repository Structure

```
ml-imbalanced-classification-pipeline/
├── project.ipynb # Main ML pipeline notebook
├── data/
│ ├── ProjectLABELED2025.xlsx
│ └── ProjectNOTLABELED2025.xlsx
├── report/
│ └── ProjectReport2025HarshaKoushikTejaAila.pdf
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🔧 Installation & Usage

> The main implementation is provided as a Jupyter Notebook for interactive analysis and reproducibility.

### Prerequisites
```bash
Python 3.8+
pip
```

### Quick Start
```bash
# Clone repository
git clone https://github.com/HarshaKoushikTeja/iee520-ml-classification-project.git
cd iee520-ml-classification-project

# Install dependencies
pip install -r requirements.txt

# Run pipeline
jupyter notebook project.ipynb
```

### Output
Generates `ProjectPredictions2025HarshaKoushikTejaAila.csv` with format:
```csv
index,label
0,1
1,0
...
```

---

## 🎓 Skills Demonstrated

### Machine Learning
- Model selection & evaluation
- Hyperparameter tuning
- Imbalanced learning techniques
- Cross-validation strategies

### Software Engineering
- Production ML pipelines
- Reproducible workflows
- Clean code architecture
- Version control

### Data Science
- Feature engineering
- Data preprocessing
- Statistical imputation
- Exploratory data analysis

---

## 🔬 Methodology

1. **Exploratory Data Analysis**: Analyzed feature distributions, missing patterns, class imbalance
2. **Preprocessing Pipeline**: Built automated ColumnTransformer for consistent transformations
3. **Model Selection**: Systematically evaluated 3 algorithms with stratified CV
4. **Hyperparameter Tuning**: Optimized 100+ configurations using GridSearchCV
5. **Final Evaluation**: Validated on 20% holdout set with balanced metrics
6. **Prediction**: Applied best model to 10,000 unlabeled samples

---

## 📈 Future Enhancements

- [ ] Implement SMOTE/ADASYN for advanced imbalance handling
- [ ] Add ensemble methods (stacking, voting classifiers)
- [ ] Deploy model via Flask/FastAPI REST API
- [ ] Create interactive dashboard with Streamlit
- [ ] Add automated model monitoring and retraining

---

## 👨‍💻 Author

**Harsha Koushik Teja Aila**  
Graduate Student | Machine Learning Engineer

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/aila-harsha-koushik-teja)
[![Email](https://img.shields.io/badge/Email-Contact-red?logo=gmail)](mailto:harshaus33@gmail.com)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?logo=github)](https://github.com/HarshaKoushikTeja)

*Open to Software Engineering, ML Engineering, and Data Science opportunities*

---

## 📜 License

Academic project for educational purposes | IEE 520 Machine Learning Course

---

**⭐ If you found this project useful, please consider starring the repository!**
