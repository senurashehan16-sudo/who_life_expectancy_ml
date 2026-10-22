# Explainable Machine Learning for WHO Life Expectancy Prediction

## Introduction

This project develops an end-to-end machine learning workflow for predicting the life expectancy of men and women. It uses an adapted World Health Organization life expectancy dataset containing health, economic, social, and demographic information from 193 countries between 2000 and 2015.

The project covers exploratory data analysis, feature engineering, preprocessing, Principal Component Analysis (PCA), model development, hyperparameter tuning, AutoML-assisted comparison, performance evaluation, and Explainable AI (XAI). Because both male and female life expectancy are predicted together, this is a **multi-output regression** problem.

## Project Objectives

- Explore the dataset and identify patterns, missing values, distributions, and outliers.
- Build a preprocessing workflow that avoids using test-set information during training.
- Engineer useful health and economic features.
- Reduce dimensionality using PCA fitted only on the training data.
- Develop and compare at least two suitable regression models.
- Use AutoML to support model selection and comparison.
- Tune model hyperparameters using cross-validation.
- Evaluate predictions using MAE, RMSE, and R².
- Interpret model behaviour using feature importance, PCA loadings, and SHAP.

## Dataset Overview

| Item | Description |
|---|---|
| Records | 2,938 |
| Original columns | 25 |
| Countries | 193 |
| Time period | 2000–2015 |
| Target 1 | Life expectancy of men |
| Target 2 | Life expectancy of women |
| Problem type | Multi-output regression |

The input file used by the notebook is named `Life Expectancy Data v2.csv`. Add the exact dataset source and licence before publishing the repository.

## Suggested Repository Structure

```text
who-life-expectancy-ml/
├── Next_Gen_DS_Assessment_01_Senura_Shehan.ipynb
├── Life Expectancy Data v2.csv
├── presentation/
│   └── WHO_Life_Expectancy_Model_Building_Presentation.pptx
└── README.md
```

## Project Workflow

1. Load and inspect the dataset.
2. Perform exploratory data analysis.
3. Separate the two target columns from the input features.
4. Split the data into training and testing sets.
5. Engineer additional health-related features.
6. Impute missing values, encode categorical variables, and scale numerical variables.
7. Fit PCA on the training data and transform both training and testing data.
8. Train Linear Regression, Random Forest, and Gradient Boosting models.
9. Tune the tree-based models using `RandomizedSearchCV`.
10. Use FLAML AutoML to support model comparison.
11. Evaluate all models using MAE, RMSE, and R².
12. Explain model predictions using feature importance, PCA loadings, and SHAP.

## Feature Engineering

The notebook creates the following features:

- `Health_expenditure_per_GDP`: compares health spending with GDP.
- `Mortality_gap`: measures the difference between adult and infant mortality indicators.
- `thinness_average`: combines the two thinness-related variables into one feature.

Feature engineering must be performed without using either target column.

## Preprocessing Pipeline

The preprocessing approach uses separate transformations for numerical and categorical data.

| Data type | Processing methods |
|---|---|
| Numerical features | Median imputation followed by standard scaling |
| Categorical features | Most-frequent imputation followed by one-hot encoding |
| Unknown categories | Ignored safely during transformation |

The data is divided into 70% training data and 30% testing data. The preprocessing objects must be fitted on the training set only and then used to transform the test set. In the current notebook, preprocessing produces 216 features before PCA.

## Principal Component Analysis

PCA is fitted only on the processed training data, with `n_components=0.95`. It reduces the feature space from 216 transformed features to 20 principal components while retaining approximately 95.3% of the variance. The first principal component explains approximately 22.9% of the variance.

The fitted PCA object is then used to transform the test data. It must never be refitted on the test set.

## Machine Learning Models

The following regression models are evaluated:

- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

`MultiOutputRegressor` is used where necessary so that one model can predict male and female life expectancy.

## Hyperparameter Tuning

Random Forest and Gradient Boosting are tuned using `RandomizedSearchCV` with 10 parameter combinations, three-fold cross-validation, and mean absolute error as the optimisation objective.

The best parameter combinations found in the notebook were:

| Model | Best parameters |
|---|---|
| Random Forest | `n_estimators=300`, `max_depth=30`, `min_samples_split=2`, `min_samples_leaf=1` |
| Gradient Boosting | `n_estimators=200`, `max_depth=2`, `learning_rate=0.05` |

## AutoML

FLAML AutoML is used separately for the two target variables to support model selection.

| Target | Selected estimator | MAE | RMSE | R² |
|---|---:|---:|---:|---:|
| Men | LightGBM | 6.0161 | 7.3497 | 0.4947 |
| Women | XGBoost | 6.0170 | 7.3845 | 0.4626 |

AutoML is used as supporting evidence rather than as a replacement for the manually developed and explained modelling workflow.

## Model Evaluation

The following metrics are used:

- **MAE:** the average absolute prediction error in years. It is easy to interpret and less sensitive to large errors than RMSE.
- **RMSE:** gives more weight to large prediction errors and helps identify models that make serious mistakes.
- **R²:** measures the proportion of target variation explained by the model.

### Current Reported Results

The table shows the average score across the two targets.

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 5.7644 | 7.0930 | 0.5167 |
| Gradient Boosting | 6.0031 | 7.3742 | 0.4775 |
| Tuned Gradient Boosting | 6.0353 | 7.3735 | 0.4776 |
| Random Forest | 6.3713 | 7.7711 | 0.4197 |
| Tuned Random Forest | 6.3723 | 7.7632 | 0.4209 |

Linear Regression has the strongest reported average performance. However, these results are **provisional** because of the validation issue explained below.

## Explainable AI

The project uses several interpretation methods:

- Random Forest feature importance identifies influential principal components.
- PCA loading analysis connects important principal components back to the original transformed features.
- SHAP explains how principal components affect predictions for both male and female life expectancy.

These methods make the dimensionality-reduced models easier to interpret, although PCA means that importance is first expressed in terms of components rather than original variables.

## Important Validation Note

The notebook creates a column named `Life expectancy` by averaging the male and female target columns for exploratory analysis. That column is not removed when the feature matrix is later created. It therefore contains direct target information and causes **target leakage**. PCA loading analysis also shows that this derived target strongly contributes to an important component.

Before treating the reported metrics as final:

1. Remove the derived `Life expectancy` column before creating the feature matrix, or keep it only in a separate EDA copy of the data.
2. Rerun preprocessing, PCA, model training, tuning, evaluation, and XAI.
3. Update the results in this README and the presentation.
4. Place preprocessing and PCA inside the scikit-learn pipeline passed to cross-validation so that each validation fold is processed independently.
5. Consider a country-grouped or time-based validation strategy if the model is intended to generalise to unseen countries or future years.

This correction is essential for obtaining trustworthy estimates of real-world model performance.

## How to Run the Project

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/who-life-expectancy-ml.git
   cd who-life-expectancy-ml
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

   On Windows, use:

   ```powershell
   .venv\Scripts\activate
   ```

3. Install the required packages:

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn flaml shap lightgbm xgboost jupyter
   ```

4. Place `Life Expectancy Data v2.csv` in the same directory expected by the notebook.

5. Start Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

6. Open the notebook and run all cells from the beginning.

The notebook can also be uploaded to Google Colab. In that case, upload or mount the dataset and update its file path before running the analysis.

## Main Technologies

- Python
- pandas and NumPy
- Matplotlib and Seaborn
- scikit-learn
- FLAML AutoML
- LightGBM and XGBoost
- SHAP
- Jupyter Notebook or Google Colab

## Key Lessons Learned

- Split the data before fitting imputers, scalers, encoders, or PCA.
- Never include a feature calculated from the target values.
- Keep all learned transformations inside the cross-validation pipeline.
- Use more than one metric because each metric describes model quality differently.
- AutoML can support model selection, but the final model still needs validation and explanation.
- PCA improves compactness but makes interpretation more difficult, so component loadings should be inspected.

## Future Improvements

- Correct the target leakage and regenerate all results.
- Use a complete scikit-learn pipeline during hyperparameter tuning.
- Compare random, country-grouped, and time-based validation strategies.
- Analyse errors by country, year, and development status.
- Explain the best validated model rather than selecting an XAI model only for convenience.
- Add automated tests for feature construction and leakage prevention.

## Author

**Senura Shehan**  
BSc (Hons) Data Science, PSB Academy

## Acknowledgements

This educational project uses an adapted WHO life expectancy dataset and open-source Python machine learning libraries.

## Licence

This repository is intended for educational use. Add an appropriate open-source licence, such as the MIT License, only after confirming that the dataset can be redistributed under compatible terms.
