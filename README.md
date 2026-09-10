
# Travel Insurance Prediction

##  Project Overview

This project focuses on predicting whether a customer is likely to purchase **Travel Insurance** based on their demographic, employment, income, family, health, and travel-related information.

The project uses **Exploratory Data Analysis (EDA)** and Machine Learning techniques to understand customer characteristics and build a predictive model for travel insurance purchase decisions.

The analysis helps identify the factors associated with travel insurance purchase and provides a foundation for businesses to target potential customers more effectively.

---

# Project Objectives

The main objectives of this project are:

1. Understand the customer dataset.
2. Clean and preprocess the data.
3. Perform Exploratory Data Analysis (EDA).
4. Identify important customer characteristics.
5. Analyze factors related to travel insurance purchase.
6. Prepare categorical and numerical features for machine learning.
7. Build a machine learning classification model.
8. Evaluate the prediction performance.
9. Identify potential customers who are more likely to purchase travel insurance.
10. Provide actionable business solutions.

---

# Problem Statement

Travel insurance companies have customer information such as age, income, employment type, family size, health conditions, frequent-flyer status, and international travel history.

However, it can be difficult to determine which customers are most likely to purchase travel insurance.

### Problem:

> **To develop a machine learning-based Travel Insurance Prediction system that analyzes customer characteristics and predicts whether a customer is likely to purchase travel insurance, helping insurance businesses improve customer targeting and marketing strategies.**

---

# 🛠️ Approach / Methodology

The project follows the workflow below:

```text
Travel Insurance Dataset
          ↓
Data Understanding
          ↓
Data Cleaning
          ↓
Exploratory Data Analysis
          ↓
Feature Preparation
          ↓
Categorical Encoding
          ↓
Feature & Target Selection
          ↓
Train-Test Split
          ↓
Machine Learning Model
          ↓
Model Evaluation
          ↓
Prediction
          ↓
Business Insights
          ↓
Business Solution
````

---

#  1. Dataset

The project uses:

```text
TravelInsurancePrediction.csv
```

The dataset contains customer information related to travel insurance purchase behavior.

### Main Features

| Feature             | Description                                   |
| ------------------- | --------------------------------------------- |
| Age                 | Age of the customer                           |
| Employment Type     | Customer's employment category                |
| GraduateOrNot       | Whether the customer is a graduate            |
| AnnualIncome        | Annual income of the customer                 |
| FamilyMembers       | Number of family members                      |
| ChronicDiseases     | Whether the customer has chronic disease      |
| FrequentFlyer       | Whether the customer is a frequent flyer      |
| EverTravelledAbroad | Whether the customer has travelled abroad     |
| TravelInsurance     | Target variable indicating insurance purchase |

The notebook contains **1,987 records and 9 columns after preprocessing**. 

---

#  2. Data Preprocessing & Cleaning

The first step is to inspect and clean the dataset.

### Removing Unnecessary Column

The column:

```text
Unnamed: 0
```

is removed because it represents an unnecessary index column.

```python
TrIns.drop(['Unnamed: 0'], axis=1, inplace=True)
```

This leaves the relevant customer features for analysis. 

---

#  Missing Value Analysis

Missing values are checked using:

```python
TrIns.isnull().sum()
```

The analysis shows **0 missing values across all nine columns** after the unnecessary index column is removed. 

Therefore, no missing-value imputation is required for the dataset in the notebook.

---

# 📋 Data Types

The dataset contains both numerical and categorical variables.

### Numerical Features

* Age
* AnnualIncome
* FamilyMembers
* ChronicDiseases
* TravelInsurance

### Categorical Features

* Employment Type
* GraduateOrNot
* FrequentFlyer
* EverTravelledAbroad

The dataset contains **5 integer columns and 4 object/categorical columns**. 

---

#  3. Exploratory Data Analysis

EDA is performed to understand customer characteristics and identify relationships between customer attributes and travel insurance purchase.

The analysis considers factors such as:

* Age
* Annual income
* Employment type
* Graduation status
* Family size
* Chronic diseases
* Frequent-flyer status
* Overseas travel experience
* Travel insurance purchase

---

# Customer Demographics

The dataset contains customers between:

```text
Minimum Age = 25
Maximum Age = 35
```

The average customer age is approximately:

```text
29.65 years
```

The median age is:

```text
29 years
```

This indicates that the dataset primarily represents relatively young adult customers. 

---

#  Annual Income Analysis

Customer annual income ranges from approximately:

```text
₹300,000
```

to:

```text
₹1,800,000
```

The average annual income is approximately:

```text
₹932,763
```

Income can therefore be considered an important customer characteristic when studying travel insurance purchase behavior. 

---

#  Family Members

The number of family members ranges from:

```text
2 to 9
```

with an average of approximately:

```text
4.75 family members
```

Family size can be considered when designing travel insurance packages for individuals and families. 

---

#  Chronic Disease

The dataset contains a binary feature:

```text
ChronicDiseases
```

where:

```text
0 → No
1 → Yes
```

The analysis can be used to understand whether health-related characteristics are associated with travel insurance purchase behavior.

---

# ✈️ Frequent Flyer

The `FrequentFlyer` feature indicates whether a customer frequently travels by air.

Frequent flyers can represent an important customer segment for travel insurance companies because they may have greater exposure to travel-related risks.

---

#  International Travel

The feature:

```text
EverTravelledAbroad
```

indicates whether the customer has travelled abroad previously.

Customers with international travel experience may represent an important target group for travel insurance marketing.

---

#  4. Target Variable

The target variable is:

```text
TravelInsurance
```

It represents whether the customer purchased travel insurance.

The target is binary:

```text
0 → Did not purchase Travel Insurance

1 → Purchased Travel Insurance
```

The notebook uses this variable as the prediction target. 

---

#  Machine Learning Approach

Since `TravelInsurance` is a binary target variable, this is a:

> **Binary Classification Problem**

The goal is to predict one of two possible outcomes:

```text
Purchase Insurance
        OR
Do Not Purchase Insurance
```

The customer features are used as input variables to predict the target.

---

# Feature Preparation

The dataset contains categorical variables such as:

* Employment Type
* GraduateOrNot
* FrequentFlyer
* EverTravelledAbroad

These variables need to be converted into numerical form before being provided to a machine learning algorithm.

Numerical variables such as:

* Age
* AnnualIncome
* FamilyMembers
* ChronicDiseases

can be used directly after appropriate preprocessing.

---

#  Model Evaluation

The classification model should be evaluated using appropriate classification metrics.

Important metrics include:

### Accuracy

Measures the overall percentage of correctly classified customers.

```text
Accuracy =
Correct Predictions / Total Predictions
```

### Precision

Measures how many customers predicted as insurance buyers actually purchased insurance.

### Recall

Measures how many actual insurance buyers were correctly identified.

### F1-Score

Provides a balance between precision and recall.

For an insurance marketing application, recall can be particularly useful because missing potential customers can reduce marketing opportunities.

---

#  Business Solution

The machine learning solution can support an insurance company through the following workflow:

```text
Customer Data
      ↓
Customer Analysis
      ↓
Machine Learning Prediction
      ↓
Identify Potential Buyers
      ↓
Customer Segmentation
      ↓
Targeted Marketing
      ↓
Personalized Insurance Offers
      ↓
Higher Conversion
      ↓
Improved Business Revenue
```

---

#  Actionable Business Solutions

## 1. Target High-Potential Customers

The prediction model can identify customers who have a higher probability of purchasing travel insurance.

The company can prioritize these customers for marketing campaigns.

---

## 2. Personalized Marketing

Customer characteristics can be used to create personalized campaigns.

For example:

* Frequent flyers → Frequent traveller insurance plans
* International travellers → Overseas travel coverage
* Families → Family travel insurance packages
* Higher-income customers → Premium travel insurance plans

---

## 3. Customer Segmentation

Customers can be divided into groups based on:

* Age
* Income
* Employment
* Family size
* Travel behavior
* International travel history

This allows businesses to design different offers for different customer groups.

---

## 4. Digital Marketing Campaigns

Potential customers can be targeted through:

* Email campaigns
* SMS
* Online advertisements
* Travel websites
* Mobile applications

The model can help reduce unnecessary marketing toward customers with a low likelihood of purchase.

---

## 5. Cross-Selling Opportunities

Travel insurance can be promoted alongside:

* Flight bookings
* International travel packages
* Hotel bookings
* Holiday packages
* Travel memberships

This can increase the probability of insurance purchase.

---

# Business Benefits

The project can help insurance companies:

* Identify potential insurance buyers
* Improve marketing efficiency
* Reduce unnecessary promotional costs
* Personalize customer offers
* Improve customer targeting
* Increase insurance conversions
* Improve customer engagement
* Support data-driven decision making

---

#  Key Insights

The dataset provides several useful observations:

### Customer Age

Customers are primarily between 25 and 35 years old, with an average age of approximately 29.65 years. 

### Income

Annual income varies considerably, ranging from ₹300,000 to ₹1,800,000. 

### Family Size

Customers have between 2 and 9 family members, with an average of approximately 4.75. 

### Customer Attributes

The dataset combines demographic, financial, health, and travel-related information, making it suitable for studying travel insurance purchase behavior. 

---

#  Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Plotly
* Scikit-learn
* Google Colab / Jupyter Notebook

The notebook imports Pandas, NumPy, Matplotlib, Seaborn and Plotly for data analysis and visualization. 


---

#  Future Enhancements

The project can be improved by:

* Testing multiple classification algorithms
* Hyperparameter tuning
* Cross-validation
* Feature importance analysis
* ROC-AUC analysis
* Creating an interactive dashboard
* Developing a web-based prediction application
* Deploying the model using Flask or Streamlit
* Integrating the model with an insurance CRM system
* Automating customer-targeted marketing campaigns

---

#  Conclusion

This project demonstrates how customer demographic, financial, health, and travel-related information can be analyzed to predict travel insurance purchase behavior.

The dataset contains **1,987 customer records**, with no missing values after preprocessing.  

By applying data preprocessing, EDA, feature preparation, and classification techniques, businesses can identify customers who are more likely to purchase travel insurance.

The resulting insights can support:

* Targeted marketing
* Personalized insurance offers
* Customer segmentation
* Cross-selling
* Better marketing resource allocation
* Increased insurance conversion

Overall, the project demonstrates how **Data Science and Machine Learning can support data-driven decision making in the insurance industry.**

---


If your notebook contains the **actual ML model + accuracy/precision/recall/F1**, those should be added to the **Model Evaluation** section rather than making up numbers.
```
