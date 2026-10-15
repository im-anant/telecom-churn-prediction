# PRD: Telecom Customer Churn Prediction (scikit-learn, Beginner Level)

> **Instructions for the agent:** This is a class project for a student who has learned only the topics in Section 4. Read the whole document, give me a short plan, then implement the tasks in Section 8 in order. Check each "Done when" line before moving on.
>
> **Hard rule:** use ONLY the tools in the "Allowed" list in Section 4. Do not add anything from the "Not allowed" list. If you think something outside the list is truly needed, ask me first. Write simple, readable code with a plain-English markdown explanation before every section.

---

## 1. Project Summary

| Item | Value |
|---|---|
| Project | Predict whether a telecom customer will churn (leave the company) |
| Problem type | Binary classification (Churn = 1, No churn = 0) |
| Tools | Python, pandas, matplotlib/seaborn, scikit-learn |
| Deliverable | One Colab notebook, `churn_prediction.ipynb`, that runs top to bottom |
| Level | Beginner: follows the workflow taught in class |

**Problem statement:** Customer churn is a critical challenge in the telecom industry. Customers discontinue their services and move to competitors, which reduces revenue and loyalty. Build a classification model that predicts whether a customer is likely to churn, based on their demographic details, usage patterns, billing information, and service features.

## 2. Workflow to Follow (the same pipeline taught in class)

DATA → EDA → PREPROCESSING → MODEL → EVALUATION → GENERALIZATION → PROJECT

The notebook should follow this order. For the evaluation and results sections, also follow the style of my reference notebook (the Malaria classifier): metrics, ROC curve, confusion matrix, saved model, and a final sample-predictions display.

## 3. Dataset

- **Name:** Telco Customer Churn (about 7,043 rows, 21 columns)
- **Kaggle slug (verify first):** `blastchar/telco-customer-churn`
- **Loading:** use `kagglehub.dataset_download(...)` as in the reference notebook. If that fails, read the CSV from `data/WA_Fn-UseC_-Telco-Customer-Churn.csv` and print which source was used.
- **Target:** `Churn` (Yes/No), about 26% Yes.

| Feature group | Columns |
|---|---|
| Demographics | `gender`, `SeniorCitizen`, `Partner`, `Dependents` |
| Usage / services | `tenure`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` |
| Billing | `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges` |
| Service feedback | Not in this dataset. Use `TechSupport` and `OnlineSecurity` as a stand-in and say so in a markdown cell. |
| Drop | `customerID` |

**Cleaning steps (one small cell each):**
1. `TotalCharges` is stored as text and has 11 blank values (new customers with `tenure` 0). Convert with `pd.to_numeric(..., errors='coerce')` and fill the blanks with 0.
2. Drop `customerID`.
3. Convert the target: `df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})`.
4. Check for duplicates and missing values, and print the results.

## 4. Allowed and Not Allowed Tools

### Allowed (covered in the class playlist or the reference notebook)
- **EDA and pandas:** `read_csv`, `head`, `info`, `describe`, `isnull`, `duplicated`, `value_counts`, `groupby`, `map`, `fillna`, `drop`
- **Plots:** matplotlib and seaborn (bar chart, histogram, boxplot, countplot, heatmap)
- **Preprocessing:** numerical vs categorical features, `StandardScaler`, `OneHotEncoder`, `ColumnTransformer`, scikit-learn `Pipeline`
- **Feature engineering and selection:** simple new columns (for example, tenure groups) and dropping unhelpful columns
- **Models:** `LogisticRegression`, `KNeighborsClassifier`, `DecisionTreeClassifier`, `RandomForestClassifier`
- **Evaluation:** `train_test_split`, confusion matrix, accuracy, precision, recall, F1-score, `classification_report`, ROC curve and AUC (as in the reference notebook)
- **Generalization:** overfitting vs underfitting (compare train and test scores), `cross_val_score`, `GridSearchCV`
- **Saving:** `joblib.dump` for the trained pipeline (the reference notebook also saves its model)
- Plain `for` loops and simple functions

### Not allowed (not covered, so leave it out)
- PyTorch, TensorFlow, or neural networks
- SMOTE or other resampling libraries (`imbalanced-learn`)
- XGBoost, LightGBM, or other boosting libraries
- SHAP or other explainability libraries
- Clustering (this is a classification project)
- Streamlit, Flask, or deployment
- Advanced Python (decorators, type hints, heavy lambda or nested comprehensions)

### Optional (use only if it was taught)
- `class_weight='balanced'` in the models, to improve recall for churners. If used, show results with and without it.

## 5. Models to Build

Build these four models inside scikit-learn `Pipeline`s (preprocessing + model), so each is trained the same way:

| # | Model | Notes |
|---|---|---|
| 1 | Logistic Regression | Main baseline (the first model, as taught in class) |
| 2 | K-Nearest Neighbors | Use `n_neighbors=5` first |
| 3 | Decision Tree | Try a limited `max_depth` to reduce overfitting |
| 4 | Random Forest | `n_estimators=100` first |

Then:
- Compare the four models in one table (accuracy, precision, recall, F1, AUC on the test set).
- Use `cross_val_score` (5-fold) on each model and add the mean score to the table.
- Pick the best one, run `GridSearchCV` on it with a small parameter grid (2 parameters, 2–3 values each), and compare before and after.
- Explain overfitting or underfitting by comparing train and test scores of the Decision Tree and Random Forest.

## 6. Conventions

- Churn = 1 is the positive class everywhere. Precision, recall, and F1 describe how well we find churners.
- Do `train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)` first, before any scaling or encoding. All preprocessing lives in the `Pipeline` so it is fitted on the training data only.
- Use `random_state=42` everywhere.
- Report final metrics on the test set only.
- Confusion matrix labels: `['No Churn', 'Churn']`.
- Accuracy alone is not enough because about 74% of customers do not churn. Always show recall and F1 for the churn class.

## 7. Notebook Structure

Put a markdown cell (1–3 plain sentences) before every code section saying what it does and why. Comment non-obvious lines.

1. Title, problem statement, dataset description
2. Download data, imports
3. Load and inspect the data (`head`, `info`, `describe`, missing values, duplicates)
4. Clean the data
5. Class distribution bar chart (Churn vs No Churn)
6. EDA: charts and 4–5 written insights
7. Separate X and y, identify numerical and categorical columns
8. Train/test split (80/20, stratified)
9. Preprocessing with `ColumnTransformer` (`StandardScaler` for numbers, `OneHotEncoder` for categories)
10. Model 1: Logistic Regression (pipeline, train, predict, evaluate)
11. Models 2–4: KNN, Decision Tree, Random Forest (same steps)
12. Confusion matrix and classification report for the best model
13. ROC curve with AUC for all models on one plot
14. Model comparison table
15. Overfitting check (train vs test scores) and 5-fold cross-validation
16. Hyperparameter tuning with `GridSearchCV` on the best model
17. Feature engineering (optional, simple): for example tenure groups or number of services, then re-test
18. Save the final pipeline with `joblib`
19. Sample predictions table: 16 test customers with true label, predicted label, and churn probability (green = correct, red = wrong), as in the reference notebook
20. Conclusion: results in plain words, what drives churn, business suggestions, limitations

## 8. Task List for the Agent

### Task 1: Setup
- Create the notebook with imports and the data download.
- **Done when:** imports and data loading run with no errors.

### Task 2: Inspect and clean
- **Done when:** no missing values remain, `TotalCharges` is numeric, `customerID` is gone, `Churn` is 0/1, and duplicates have been checked.

### Task 3: EDA
- Required charts: churn distribution, churn rate by `Contract`, `InternetService`, and `PaymentMethod`; `tenure` and `MonthlyCharges` by churn (histogram or boxplot); correlation heatmap of the numeric columns.
- **Done when:** every chart has a title and axis labels, and 4–5 insights are written below the charts.

### Task 4: Split and preprocess
- **Done when:** train and test shapes are printed, the churn percentage in each is printed (they should be similar), and the `ColumnTransformer` is built.

### Task 5: Logistic Regression
- **Done when:** accuracy, precision, recall, F1, and the confusion matrix are shown, each with a one-sentence plain explanation.

### Task 6: Other models and comparison
- **Done when:** KNN, Decision Tree, and Random Forest are trained, and one comparison table and one ROC plot cover all four.

### Task 7: Generalization
- **Done when:** the train vs test score is shown for each model, 5-fold cross-validation scores are in the table, and a markdown cell says which models overfit and why.

### Task 8: Tuning
- **Done when:** `GridSearchCV` finishes, the best parameters are printed, and before and after scores are compared.

### Task 9: Save and sample predictions
- **Done when:** `churn_model.pkl` exists, it can be reloaded in a new cell, and it gives the same predictions. The 16-row table shows green and red rows.

### Task 10: Conclusion and final check
- **Done when:** Restart and Run All completes with no errors, and the conclusion states limitations honestly.

## 9. Success Targets (be realistic)

Report the real numbers honestly, even if they miss these ranges.

| Metric (test set) | Realistic range |
|---|---|
| Accuracy | about 78–82% |
| AUC-ROC | about 0.82–0.86 |
| Recall (churn) | about 0.50–0.60 (higher, around 0.75, if `class_weight='balanced'` is used, with lower precision) |
| F1 (churn) | about 0.55–0.62 |

Logistic Regression and Random Forest usually perform similarly on this dataset. Do not worry if the simple model matches the complex one. Say so in the conclusion.

## 10. Quality Checklist

- [ ] Runs end to end after Restart and Run All
- [ ] Only tools from the Allowed list are used
- [ ] Every section has a plain-English markdown explanation
- [ ] Preprocessing is inside the `Pipeline` (no leakage)
- [ ] Metrics are on the test set, with Churn = 1 as the positive class
- [ ] Charts have titles and axis labels
- [ ] Train vs test comparison and cross-validation are included
- [ ] Limitations stated (no real feedback column, imbalanced data, one dataset)

## 11. Insights to Check Against the Data

Check each against the actual charts before writing it. Do not copy without verifying.

- Month-to-month customers seem to churn more than one- or two-year customers.
- Customers with short tenure seem to churn more.
- Fiber optic customers seem to churn more.
- Customers without tech support or online security seem to churn more.

**Possible suggestions:** offer discounts for longer contracts, give extra help to new customers, promote tech support and security add-ons.
