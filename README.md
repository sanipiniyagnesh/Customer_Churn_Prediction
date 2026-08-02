# 📊 Customer Churn Prediction & Risk Analytics

An end-to-end Machine Learning pipeline that predicts customer churn probability for subscription services using **Logistic Regression** and deploys an interactive prediction microservice using **Streamlit**.

---

## 📌 Project Overview
Customer attrition (churn) directly impacts business revenue. This project processes 7,000+ customer records to identify high-risk accounts before they cancel, allowing retention teams to intervene proactively.

---

## 🎯 Key Metrics & Results
- **Overall Model Accuracy:** **80.41%** on unseen test data (`N = 1,409`).
- **Primary Retention Protectors:** `Tenure` ($-1.22$) and `Two-year Contracts` ($-0.59$).
- **Top Churn Risk Driver:** `Fiber Optic Internet Service` ($+0.78$).

---

## 🛠️ Technical Stack & Dependencies
- **Language:** Python
- **Data Preprocessing & Math:** `pandas`, `numpy`
- **Machine Learning:** `scikit-learn` (`LogisticRegression`, `StandardScaler`, `train_test_split`)
- **Visualization:** `matplotlib`, `seaborn`
- **Deployment:** `streamlit`, `pickle`

---

## 🚀 How to Run Locally

```bash
# 1. Clone repository
git clone [https://github.com/sanipiniyagnesh/Customer_Churn_Prediction.git](https://github.com/sanipiniyagneshr/Customer_Churn_Prediction.git)
cd customer-churn-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run Streamlit App
streamlit run app_churn.py
