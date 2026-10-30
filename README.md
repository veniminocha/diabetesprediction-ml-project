# Diabetes Risk Predictor

A Predictive Analytics mini-project that explores diabetes classification using the Pima Indians Diabetes dataset (768 records, 8 clinical input features).

## Objectives
- Clean and explore clinical measurement data.
- Compare Logistic Regression, Linear Discriminant Analysis (LDA), and Quadratic Discriminant Analysis (QDA).
- Evaluate models with stratified 5-fold cross-validation.
- Tune Logistic Regression using Grid Search and Randomized Search.
- Evaluate the selected model on an untouched test set.
- Interpret Logistic Regression coefficients using odds ratios.

## Repository structure
```text
diabetes-risk-predictor/
├── data/                 # Dataset is downloaded here when notebook runs
├── images/               # Generated plots
├── notebooks/
│   └── diabetes_prediction.ipynb
├── results/              # Generated metric tables
├── README.md
├── requirements.txt
└── .gitignore
```

## How to run
1. Clone this repository.
2. Create and activate a virtual environment.
3. Install dependencies: `pip install -r requirements.txt`
4. Start Jupyter: `jupyter notebook`
5. Open `notebooks/diabetes_prediction.ipynb` and run cells from top to bottom.

The notebook downloads the public Pima dataset automatically if `data/diabetes.csv` is not present.

## Dataset
Pima Indians Diabetes dataset, commonly distributed as 768 observations with eight predictors and a binary `Outcome` column. The notebook uses the public Brownlee dataset mirror and saves a local copy under `data/`.

## Methods
Missing/invalid zero measurements are treated as missing for Glucose, BloodPressure, SkinThickness, Insulin, and BMI. Imputation and scaling are performed inside Scikit-learn pipelines to reduce data leakage. Zero pregnancies and the binary outcome are retained as valid values.

## Disclaimer
This is an educational machine-learning project, not a medical diagnostic tool. It should not be used for healthcare decisions.
