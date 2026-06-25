

#  Loan Approval Prediction using Logistic Regression

##  Project Overview

This project aims to predict whether a loan application will be **approved or rejected** using a **Logistic Regression** machine learning model. The model is trained on a loan dataset containing applicant financial information, employment details, and credit-related attributes.

The project demonstrates a complete machine learning workflow—from data preprocessing and exploratory analysis to model development, evaluation, and deployment using **Streamlit**.

---

#  Objective

The objective of this project is to build a binary classification model capable of predicting loan approval decisions based on an applicant's financial and credit profile. The project also demonstrates how a trained machine learning model can be deployed as an interactive web application for real-time predictions.

---

# Dataset Information

The dataset consists of **45,000 loan applications** with demographic, financial, employment, and credit history information.

### Features Used

| Feature                    | Description                                  |
| -------------------------- | -------------------------------------------- |
| person_age                 | Age of the applicant                         |
| person_income              | Annual income of the applicant               |
| person_emp_exp             | Employment experience (years)                |
| loan_amnt                  | Requested loan amount                        |
| loan_int_rate              | Loan interest rate (%)                       |
| loan_percent_income        | Ratio of loan amount to annual income        |
| cb_person_cred_hist_length | Length of credit history                     |
| credit_score               | Applicant's credit score                     |
| loan_status                | Target variable (0 = Rejected, 1 = Approved) |

---

#  Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib
* Streamlit

---

#  Project Workflow

## 1. Data Loading

The loan dataset was loaded into a Pandas DataFrame for analysis and preprocessing.

---

## 2. Data Exploration (EDA)

Exploratory Data Analysis (EDA) was performed to understand the dataset and identify important patterns.

The following analyses were carried out:

* Dataset shape
* Dataset information
* Statistical summary
* Missing value analysis
* Distribution of numerical variables
* Loan approval distribution
* Feature relationship visualization

Various plots were generated using Matplotlib and Seaborn for better understanding of the data.

---

## 3. Data Cleaning

The dataset was examined for missing values using:

```python
df.isnull().sum()
```

Result:

* No missing values were present.
* No duplicate records required removal.
* The dataset was already clean and ready for modeling.

---

## 4. Feature Selection

Since Logistic Regression requires numerical inputs, only numerical features were selected for training.

```python
df_numeric = df.select_dtypes(include=["int64", "float64"])
```

The target variable:

```python
loan_status
```

was separated from the input features.

---

## 5. Feature Scaling

Since Logistic Regression is sensitive to feature magnitudes, feature scaling was applied using **StandardScaler**.

```python
from sklearn.preprocessing import StandardScaler
```

The scaler standardizes every feature by transforming them to have:

* Mean = 0
* Standard Deviation = 1

This improves model convergence and prediction performance.

---

## 6. Train-Test Split

The dataset was divided into training and testing sets.

* Training Data : 80%
* Testing Data : 20%

```python
train_test_split()
```

The training data was used for learning the model parameters, while the testing data was used to evaluate model performance.

---

#  Machine Learning Model

The model used is:

## Logistic Regression

Logistic Regression is a supervised machine learning algorithm used for binary classification problems.

In this project, it predicts:

* **1 → Loan Approved**
* **0 → Loan Rejected**

The model estimates the probability of loan approval using the logistic (sigmoid) function.

---

# Model Evaluation

The trained model was evaluated using multiple classification metrics.

### Accuracy Score

Measures the overall percentage of correctly classified loan applications.

---

### Confusion Matrix

Used to analyze:

* True Positives
* True Negatives
* False Positives
* False Negatives

This provides a detailed understanding of classification performance.

---

### Classification Report

The classification report includes:

* Precision
* Recall
* F1-Score
* Support


---

#  Model Saving

After training, the model and scaler were saved using **Joblib**.

```python
joblib.dump(output,"loan_model3.pkl")
joblib.dump(scaler,"scaler3.pkl")
```

Saving the model eliminates the need to retrain it every time predictions are required.

---

# Streamlit Deployment

A Streamlit web application was developed to provide an interactive interface for loan prediction.

Users can enter:

* Age
* Annual Income
* Employment Experience
* Loan Amount
* Interest Rate
* Loan Percent Income
* Credit History Length
* Credit Score

The application:

* Scales the input data
* Loads the saved Logistic Regression model
* Predicts loan approval
* Displays prediction probabilities
* Shows applicant details

---

# Project Structure

```
Loan-Approval-Prediction
│
├── app.py
├── loan_model.pkl
├── scaler.pkl
├── loan.csv
├── loan_approval.ipynb
├── requirements.txt
├── README.md

```

---

#  How to Run

## Clone the Repository

```bash
git clone https://github.com/yourusername/Loan-Approval-Prediction.git
```

---

## Install Required Libraries

```bash
pip install -r requirements.txt
```

---

## Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open automatically in your web browser.


#  Libraries Used

```
streamlit
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
```



