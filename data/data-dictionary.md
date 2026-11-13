# Data

## Source

The **Cleveland Heart Disease** dataset from the
[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/45/heart+disease),
collected at the Cleveland Clinic Foundation. It is the most complete of the
four databases in that collection and the one almost all published work uses.

`processed.cleveland.data` is headerless CSV — column names are assigned in the
notebook.

| Property | Value |
| --- | --- |
| Rows as supplied | 303 patients |
| Rows after cleaning | **297** (6 dropped for missing values) |
| Columns | 14 (13 predictors + outcome) |
| Collected | late 1980s |
| Missing values | 6 `?` placeholders across `ca` and `thal` |

## Columns

| # | Column | Type | Meaning |
| --- | --- | --- | --- |
| 1 | `age` | continuous | Age in years |
| 2 | `sex` | binary | 1 = male, 0 = female |
| 3 | `cp` | categorical | Chest pain type: 1 typical angina, 2 atypical angina, 3 non-anginal pain, **4 asymptomatic** |
| 4 | `trestbps` | continuous | Resting blood pressure, mm Hg on admission |
| 5 | `chol` | continuous | Serum cholesterol, mg/dl |
| 6 | `fbs` | binary | Fasting blood sugar > 120 mg/dl |
| 7 | `restecg` | categorical | Resting ECG: 0 normal, 1 ST-T abnormality, 2 probable LV hypertrophy |
| 8 | `thalach` | continuous | Maximum heart rate achieved |
| 9 | `exang` | binary | Exercise-induced angina |
| 10 | `oldpeak` | continuous | ST depression induced by exercise relative to rest |
| 11 | `slope` | categorical | Slope of the peak exercise ST segment: 1 up, 2 flat, 3 down |
| 12 | `ca` | continuous | Number of major vessels (0–3) coloured by fluoroscopy |
| 13 | `thal` | categorical | 3 normal, 6 fixed defect, **7 reversible defect** |
| 14 | `num` | outcome | Diagnosis, 0–4 |

Bold values are the levels that turned out to carry the most predictive weight.

## Target construction

`num` records disease severity from 0 to 4. The notebook collapses it to a
binary outcome, which is the standard treatment for this dataset:

```python
df['target'] = (df['num'] > 0).astype(int)
```

| target | Meaning | Patients |
| --- | --- | ---: |
| 0 | No heart disease | 160 |
| 1 | Heart disease present | 137 |

Roughly 54 / 46 — close enough to balanced that no resampling or class weighting
was needed, though the train/test split is stratified to preserve the ratio.

## Cleaning

`ca` and `thal` arrive as text because of `?` placeholders. They are coerced to
numeric, which turns the placeholders into `NaN`, and those 6 rows are dropped —
2% of the data, too few to be worth imputing.

```python
df['ca']   = pd.to_numeric(df['ca'],   errors='coerce')
df['thal'] = pd.to_numeric(df['thal'], errors='coerce')
df = df.dropna().reset_index(drop=True)      # 303 -> 297
```

## Licence

The UCI Heart Disease dataset is distributed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Cite:

> Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989).
> *Heart Disease*. UCI Machine Learning Repository.
> https://doi.org/10.24432/C52P4X
