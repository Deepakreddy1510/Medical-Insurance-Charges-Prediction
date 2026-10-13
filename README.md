# Medical Insurance Charges Prediction

Predicting recorded medical insurance charges using multiple linear regression, a BMI × smoking interaction, and reproducible model evaluation.

The project compares three OLS specifications on identical validation folds, evaluates the selected model on a reserved test set, and examines residuals and influential observations. The selected **interaction OLS** model achieves **test R² of 0.8862**, **MAE of 2,828.53**, and **RMSE of 4,572.84** in charge units.

## Project files

| File | Description |
|---|---|
| [Medical_Insurance_Charges_Regression.ipynb](Medical_Insurance_Charges_Regression.ipynb) | Analysis notebook with embedded data, executed outputs, plots, and validation checks |
| [Medical-Insurance-Charges-slides.pptx](Medical-Insurance-Charges-slides.pptx) | Presentation covering the objective, dataset, methods, results, and conclusions |
| README.md | Project overview and reproduction instructions |

## Objective

Estimate a person's recorded charges from age, sex, BMI, number of children, smoking status, and region. The emphasis is on prediction for unseen observations, fair comparison of candidate models, and transparent reporting of limitations.

This is an academic observational-data project. It does not establish causal effects or provide a validated insurance-pricing system.

## Dataset

The dataset contains **1,338 rows**, **six predictors**, and one target column.

| Column | Description | Values / range |
|---|---|---|
| `age` | Age | 18–64 |
| `sex` | Recorded sex category | female, male |
| `bmi` | Body mass index | 15.96–53.13 |
| `children` | Number of children | 0–5 |
| `smoker` | Smoking status | no, yes |
| `region` | Recorded region | northeast, northwest, southeast, southwest |
| `charges` | Prediction target | Approximately 1,121.87–63,770.43 |

There are no missing values and one exact duplicate pair. The duplicate records are retained but grouped so they cannot cross training/test or fitting/validation boundaries. The CSV does not specify currency or observation period, so results are labelled in **charge units**.

The CSV is embedded in the notebook; no separate dataset download or local CSV path is required.

## Methodology

### Data preparation and splitting

- Reserve approximately 20% of unique-record groups using `GroupShuffleSplit`, with seed **42**: **1,070 training rows** and **268 test rows**.
- Perform exploratory analysis using training data only.
- Encode sex and smoking as binary indicators and region using three dummy variables. Reference categories are female, non-smoker, and northeast.
- Centre age, BMI, and children using fitting-data means. Recalculate these means within each cross-validation fitting fold.

### Candidate models

| Model | Specification |
|---|---|
| Additive OLS | All six inputs, with a common BMI slope for smokers and non-smokers |
| Interaction OLS | Adds centred BMI × smoker, allowing the BMI slope to differ by smoking status |
| Log interaction OLS | Fits log charges with the interaction and returns predictions to the original scale using fitting-residual smearing |

For the interaction model, writing centred BMI as `bmi_c` and smoking as `S` (0 or 1), the BMI-related contribution is:

```text
beta_BMI × bmi_c + beta_smoker × S + beta_interaction × bmi_c × S
```

The non-smoker BMI slope is `beta_BMI`; the smoker slope is `beta_BMI + beta_interaction`. The smoking difference therefore depends on BMI.

The log candidate is considered when baseline training residual standard deviations across fitted-value quartiles have a maximum-to-minimum ratio of at least **1.5**. The observed ratio was **1.799**. This is a pragmatic eligibility rule, not proof that logging is appropriate.

All candidates retain their main effects. Age enters linearly in the original-scale models; age-squared and spline specifications are not tested.

### Validation and selection

- Use five shuffled, group-preserving folds within training, identical for all candidates.
- Fit on four folds and predict the fifth, giving each training row one out-of-fold prediction.
- Estimate centring and log smearing from each fitting fold only.
- Compare predictions in original charge units using MAE, RMSE, and R².
- Select the simplest candidate within **1% of the lowest mean fold RMSE**.
- Refit the selected specification on all training rows, then evaluate it once on the reserved test set.

The test set does not guide model selection. Mean fold RMSE and pooled out-of-fold RMSE are reported separately because they are not generally equal.

## Results

### Five-fold model comparison

| Model | Mean validation RMSE | Mean validation MAE |
|---|---:|---:|
| Additive OLS | 6,107.20 | 4,210.73 |
| **Interaction OLS** | **4,909.40** | **2,986.02** |
| Log interaction OLS | 10,622.51 | 5,193.15 |

Interaction OLS reduced mean validation RMSE by **19.61%** relative to additive OLS and was selected. The log candidate performed worse on the original charge scale.

### Selected model: reserved test set

| Metric | Result |
|---|---:|
| Test observations | 268 |
| MAE | 2,828.53 |
| RMSE | 4,572.84 |
| R² | 0.8862 |

Predicting the training mean for every test row gives an RMSE of **13,613.40**. No negative test predictions occurred, and no predictions were clipped.

**R² of 0.8862 is not “88.62% accuracy.”** It compares squared prediction error with variation around the test-set mean; MAE and RMSE describe the size of prediction errors.

### Test subgroup performance

| Smoking status | Rows | MAE | RMSE | R² |
|---|---:|---:|---:|---:|
| Non-smoker | 208 | 2,282.29 | 3,920.01 | 0.516 |
| Smoker | 60 | 4,722.17 | 6,334.95 | 0.714 |

Smokers have larger absolute prediction errors. Their higher subgroup R² reflects a different target spread and does not establish smaller errors.

## Diagnostics and calculation checks

The notebook includes exploratory plots, out-of-fold residual comparisons, actual-versus-predicted plots, and training residual diagnostics:

- Q–Q and scale-location plots show remaining tail departures and uneven residual spread.
- Leverage, externally studentized residuals, and Cook's distance screen unusual or influential observations.
- **104 training rows** trigger at least one screen. All are retained; a diagnostic flag does not establish a data error.
- VIF checks assess collinearity, and HC3 standard errors support cautious coefficient inference.

Executable checks verify group separation, fold coverage, metric formulas, least-squares agreement across statsmodels/NumPy/scikit-learn, leverage, residual scaling, and Cook's distance. A temporary deleted-observation refit checks calculations for the most influential training observation.

The stored notebook contains executed outputs and no error outputs. Its delivery record states that all cells ran in a fresh IPython process and all nine figures were visually reviewed. Standard Jupyter kernel transport could not be tested in that environment; local **Restart Kernel and Run All** is recommended to confirm Jupyter integration.

## Run locally

Use **Python 3.11 or later**.

### 1. Clone the repository

```bash
git clone https://github.com/Deepakreddy1510/Medical-Insurance-Charges-Prediction.git
cd Medical-Insurance-Charges-Prediction
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows Command Prompt:

```bat
.venv\Scripts\activate
```

macOS / Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install numpy pandas scipy scikit-learn statsmodels matplotlib seaborn ipython jupyterlab
```

The notebook displays the package versions used for its recorded execution. Small numerical differences may occur with different versions.

### 4. Open and execute the notebook

```bash
jupyter lab Medical_Insurance_Charges_Regression.ipynb
```

Select **Kernel → Restart Kernel and Run All Cells** and run cells in order. Data and plot outputs are embedded; plot-file export is disabled by default through `EXPORT_PLOTS=False`.

## Limitations and future work

- Findings describe associations, not causal effects.
- Residual structure remains; normal errors and constant variance are not established.
- Diagnoses, policy details, identifiers, and timestamps are unavailable, limiting explanation and provenance checks.
- Grouping exact duplicates does not establish that all observations are independent.
- The comparison covers only three specifications. Polynomial age terms, splines, and other model families remain possible extensions.
- The log eligibility gate uses the overall training data, so internal validation is not an unbiased assessment of the complete selection procedure. Nested validation and external data would strengthen future evaluation.
- One reserved split does not establish performance in other populations or time periods.

## References

- [Insurance dataset attributed by the source repository](https://www.kaggle.com/datasets/mirichoi0218/insurance)
- [Original reference repository](https://github.com/jishen-harilal/insurance-charges-prediction-linear-regression)
- Course material on multiple OLS, leverage, studentized residuals, and Cook's distance, referenced in the notebook.

The reported metrics come from this project's notebook outputs rather than previously reported reference-repository scores.
