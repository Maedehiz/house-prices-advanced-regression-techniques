# House Prices Prediction - Kaggle Competition

## Overview

This project is a machine learning solution for the Kaggle competition **House Prices: Advanced Regression Techniques**. The goal is to predict residential property prices in Ames, Iowa, using 79 explanatory variables describing various aspects of the houses.

The project covers the complete machine learning workflow, including data cleaning, feature engineering, exploratory data analysis, model selection, hyperparameter tuning, and competition submission.

---

## Dataset

The dataset contains information about residential homes, including:

* Property size and dimensions
* Construction quality and condition
* Basement and garage characteristics
* Exterior and interior features
* Neighborhood information
* Sale details

Target Variable:

* **SalePrice** – the final sale price of each house

Dataset source:

* Kaggle House Prices: Advanced Regression Techniques Competition

---

## Project Workflow

### 1. Data Cleaning

Missing values were handled according to the meaning of each feature:

* Categorical features representing the absence of a characteristic were filled with `"None"`.
* Numerical features representing missing physical structures (e.g., garage area when no garage exists) were filled with `0`.
* Remaining categorical features were filled using the mode.
* Remaining numerical features were filled using the median.

---

### 2. Feature Engineering

Several additional features were created to improve model performance:

#### TotalSF

Combines the total living and basement area:

```python
TotalSF = TotalBsmtSF + GrLivArea
```

#### TotalBath

Represents the total number of bathrooms:

```python
TotalBath = FullBath + BsmtFullBath + 0.5*(HalfBath + BsmtHalfBath)
```

These engineered features became some of the most important predictors in the final model.

---

### 3. Feature Encoding

#### Ordinal Encoding

Applied to features with a natural ranking:

Examples:

* ExterQual
* KitchenQual
* BsmtQual
* GarageQual
* FireplaceQu
* HeatingQC
* GarageFinish
* PavedDrive
* Basement finishing quality features

#### One-Hot Encoding

Applied to nominal categorical features using:

```python
pd.get_dummies()
```

---

### 4. Target Transformation

The target variable was positively skewed.

A logarithmic transformation was applied:

```python
y = np.log1p(SalePrice)
```

Predictions were transformed back using:

```python
np.expm1(predictions)
```

---

## Models Evaluated

Multiple regression algorithms were evaluated using LazyPredict and cross-validation:

* Linear Regression
* Random Forest Regressor
* Gradient Boosting Regressor
* HistGradientBoostingRegressor
* XGBoost Regressor
* CatBoost Regressor
* LightGBM Regressor

After experimentation and tuning, **XGBoost Regressor** produced the best overall results.

---

## Hyperparameter Tuning

RandomizedSearchCV was used to optimize XGBoost parameters.

Best parameters:

```python
{
    "subsample": 0.7,
    "n_estimators": 500,
    "max_depth": 3,
    "learning_rate": 0.05,
    "colsample_bytree": 0.8
}
```

---

## Results

### Best Kaggle Score

```text
0.12422
```

### Latest Kaggle Score

```text
0.12422
```

### Submission Version

```text
V2
```

### Competition

House Prices: Advanced Regression Techniques

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* XGBoost
* CatBoost
* LightGBM
* Matplotlib
* Seaborn
* Jupyter Notebook

---

## Key Lessons Learned

This project provided practical experience in:

* Handling missing values based on domain meaning
* Feature engineering
* Encoding categorical variables
* Regression modeling
* Hyperparameter optimization
* Cross-validation
* Kaggle competition workflows
* Model evaluation and leaderboard analysis

---

## Future Improvements

Potential improvements include:

* Additional feature engineering
* Stacking and blending multiple models
* Advanced ensemble methods
* More extensive hyperparameter optimization
* Feature selection techniques

---

## Author

**Maedeh Izadi**

Software Engineering Student

Interested in:

* Artificial Intelligence
* Machine Learning
* Explainable AI (XAI)
* Medical AI
