# TravelInsurancePrediction
# Travel Insurance Purchase Prediction

A machine learning project that predicts whether a customer is likely to purchase travel insurance, based on demographic, financial, and travel-history attributes. The goal is both **predictive** (classify likely buyers) and **explanatory** (identify which factors drive the purchase decision).

---

## 1. Dataset

- **Source file:** `TravelInsurancePrediction.csv`
- **Size:** 1,987 rows × 9 usable columns (an `Unnamed: 0` index column was dropped)
- **Target variable:** `TravelInsurance` (1 = purchased, 0 = did not purchase)

| Column | Type | Description |
|---|---|---|
| Age | int | Customer age (25–35 in this dataset) |
| Employment Type | categorical | Government Sector / Private Sector or Self-Employed |
| GraduateOrNot | categorical | Whether the customer is a graduate |
| AnnualIncome | int | Annual income (₹300,000–₹1,800,000) |
| FamilyMembers | int | Number of family members |
| ChronicDiseases | binary | Has a chronic disease (1/0) |
| FrequentFlyer | categorical | Flies frequently (Yes/No) |
| EverTravelledAbroad | categorical | Has travelled abroad before (Yes/No) |
| TravelInsurance | binary (target) | Purchased travel insurance (1/0) |

Class balance: ~35.7% of customers purchased insurance, ~64.3% did not — a moderate class imbalance, not severe enough to require resampling but worth noting when interpreting recall.

---

## 2. Data Preprocessing & Cleaning

- Dropped the redundant `Unnamed: 0` index column.
- Checked for missing values with `isnull().sum()` — **the dataset has zero missing values**, so no imputation was required.
- Checked for duplicate rows with `.duplicated()` — **173 duplicate rows were identified**. *(Note: in the current notebook these are detected but not yet dropped — see Limitations below.)*
- No formal outlier treatment (e.g., IQR capping) was applied. `AnnualIncome` and `FamilyMembers` were reviewed via `.describe()` and histograms; values stayed within plausible real-world ranges, so no rows were removed as outliers.
- Categorical variables (`FrequentFlyer`, `EverTravelledAbroad`, `GraduateOrNot`, `Employment Type`) were label-encoded to 0/1 for model compatibility.

---

## 3. Exploratory Data Analysis (EDA)

Key visualizations produced:
- Countplots for `Employment Type`, `FamilyMembers` (overall, and split by `ChronicDiseases` and `Employment Type`)
- Pie charts for `GraduateOrNot`, `FrequentFlyer`, `EverTravelledAbroad` distribution
- A heatmap of `AnnualIncome` vs `FamilyMembers` proportions
- A full correlation heatmap across numeric features
- Bar plots of `AnnualIncome` by `FrequentFlyer` and by `EverTravelledAbroad`
- Histograms of `Age`, `AnnualIncome`, and `FamilyMembers` split by `TravelInsurance` outcome

**Key insights from EDA:**
- **Annual income is the strongest visible driver.** Customers who purchased insurance have a noticeably higher average `AnnualIncome` than those who didn't, and this holds up in the correlation heatmap.
- **Travel history matters.** Customers who have flown frequently or travelled abroad before purchase insurance at a visibly higher rate than those who haven't — makes intuitive sense, since these customers already understand travel-related risk.
- **Employment type shows an income overlap**, but Government Sector employees skew toward more moderate, consistent incomes vs. the wider spread in Private Sector/Self-Employed.
- Family size and chronic disease status show weaker, more diffuse relationships with the target compared to income and travel history.

---

## 4. Feature Engineering

Two engineered features were added on top of the raw columns:

- **`IncomePerFamilyMember`** = `AnnualIncome / FamilyMembers` — captures disposable income per dependent rather than raw household income, which is arguably more relevant to a discretionary purchase like travel insurance.
- **`TravelScore`** = `FrequentFlyer + EverTravelledAbroad` — a simple 0–2 composite signal of "how much of a traveler" this customer already is.

---

## 5. Model Building

**Train/test split:** 80/20 (`test_size=0.2`, `random_state=42`), stratification not applied explicitly (relies on the split being representative given moderate class balance).

**Models trained and compared:**

| Model | Why it was included |
|---|---|
| Logistic Regression | Simple, interpretable linear baseline |
| K-Nearest Neighbors | Captures local, non-linear patterns |
| Decision Tree | Captures non-linear interactions, easy to interpret |
| Random Forest | Ensemble of trees, typically more robust and higher-accuracy than a single tree |

All models were trained with **default scikit-learn hyperparameters** — no tuning or cross-validation was performed in this iteration (see Limitations).

---

## 6. Results

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| **Random Forest** | **0.8241** | **0.8142** | 0.6525 | **0.7244** |
| Decision Tree | 0.8040 | 0.7520 | **0.6667** | 0.7068 |
| KNN | 0.7990 | 0.7850 | 0.5957 | 0.6774 |
| Logistic Regression | 0.7487 | 0.7158 | 0.4823 | 0.5763 |

**Random Forest was selected as the final/best model.** It has the highest accuracy, precision, and F1 score of the four, indicating it balances correctly identifying purchasers against not over-predicting purchases. Its recall (65%) is respectable but not the strongest metric — meaning it still misses roughly a third of actual buyers, which is a reasonable trade-off if the business cost of a false positive (marketing to a non-buyer) is lower than the cost of extra model complexity.

### Sample predictions (unseen data)

Three synthetic customer profiles were run through the trained Random Forest model:


This confirms the pattern seen in EDA: higher income + prior travel experience strongly pushes the prediction toward "will purchase."

---

## 7. Key Factors Influencing the Purchase Decision

Based on the correlation analysis and EDA (not yet formally confirmed via Random Forest `feature_importances_` — see Limitations), the factors most associated with purchasing travel insurance are, in approximate order of influence:

1. **Annual Income** — the single strongest differentiator between buyers and non-buyers.
2. **Ever Travelled Abroad** — prior international travel experience correlates with insurance uptake.
3. **Frequent Flyer status** — similarly correlates, reinforcing that travel behavior (not just income) matters.
4. **Employment Type** — a secondary factor, likely acting as a proxy for income stability.
5. **Family Members / Chronic Diseases** — weaker, less consistent relationships with the target in this dataset.

-
