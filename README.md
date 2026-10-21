    # Machine Learning-Based Network Intrusion Detection and Multi-Class Attack Classification

    Module: Special Topics in CyberSecurity (CSCY413_SEP26)
    Team: Anannya Das (23000036), Bhagya Sijumon (23000742)

    ## Aim
    This project develops and evaluates a leakage-aware, explainable machine learning pipeline for network intrusion detection and multiclass attack classification on the CIC-IDS2017 dataset. Logistic Regression, Decision Tree, Random Forest, XGBoost and SVM classifiers are compared for binary (benign vs attack) and multiclass (attack type) detection. Imbalance-aware metrics (MCC, Cohen's Kappa, macro-F1) are used, SHAP explains the best models, and results are shown in an interactive dashboard.

    ## Dataset
    CIC-IDS2017 (Parquet version from Kaggle). The data is not stored in this repository. See data/README.md.

    ## Workflow
    Load, clean and preprocess, explore (EDA), split, feature selection and balancing (training data only), models (binary then multiclass), evaluation, SHAP, dashboard.

    ## Repository structure
    - data/ : dataset instructions (data not committed)
    - notebooks/ : Jupyter notebooks
    - src/ : preprocessing, models, evaluation, explainability code
    - dashboard/ : interactive dashboard
    - results/ : tables and figures
    - reports/ : deliverable reports
    - tests/ : tests
    - docs/ : methodology and ethics notes

    ## Team and responsibilities
    - Bhagya Sijumon: pipeline design, leakage removal, Random Forest/XGBoost/SVM, resampling, SHAP, statistical tests, final integration.
    - Anannya Das: repository and issue board, EDA, Logistic Regression/Decision Tree baselines, metrics, dashboard, documentation.

    ## Ethics
    Public data only. No scanning, exploitation or traffic generation against any live system.

    ## Status
    Week 4: repository, project structure and task board set up.
