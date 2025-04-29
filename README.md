# 📊 ANN-Based Customer Churn Classification

A Streamlit web application that predicts the likelihood of a customer churning using an Artificial Neural Network (ANN) model trained on the **Churn Modelling Dataset**. The app enables interactive input of customer features to instantly obtain predictions based on a pre-trained deep learning model.

🔗 **Live App**: [Visit the Streamlit App](https://ann-classification-churn-3kq5m6c96mw9lz48bszpqv.streamlit.app/)

---

## 🚀 Project Overview

This project addresses the business-critical problem of **customer churn prediction**. It uses a fully connected ANN trained on a dataset of bank customers. The app allows users to enter various customer attributes like age, credit score, and account balance, and outputs the probability of churn in real time.

---

## 🧠 Model Details

- **Architecture**: Fully Connected ANN with multiple Dense layers
- **Framework**: TensorFlow / Keras
- **Objective**: Binary classification (Churn or Not Churn)
- **Preprocessing**:
  - Label Encoding for Gender
  - One-Hot Encoding for Geography
  - Standard Scaling for numerical features

---

## 🛠️ Features

- Real-time prediction via interactive UI
- Trained deep learning model (`.h5` file)
- Cleaned and preprocessed dataset (`Churn_Modelling.csv`)
- Pickled encoders and scaler for deployment
- Streamlit frontend with sliders, input fields, and dropdowns

---

## 📂 Repository Structure
```bash
ann-classification-churn/
│ ├── app.py # Streamlit app script
├── churn_model.h5 # Trained ANN model
├── Churn_Modelling.csv # Dataset used for training
├── experiments.ipynb # Model building and tuning experiments
├── prediction.ipynb # Final prediction pipeline notebook
├── label_encoder_gender.pkl # Label Encoder for Gender
├── ohe_geo.pkl # One-Hot Encoder for Geography
├── scaler.pkl # Standard Scaler
├── requirements.txt # Python dependencies
├── LICENSE # License file
└── README.md # You're here!

```

---

## ⚙️ How to Run Locally

### 1. Clone the Repo

```bash
git clone https://github.com/your-username/mohammedsaim-quadri-ann-classification-churn.git
cd mohammedsaim-quadri-ann-classification-churn
```

### 2.Set Up Environment
Recommend using a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

### 3.Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the App
```bash
streamlit run app.py
```

---

## 📈 Dataset
The model is trained on the Churn Modelling Dataset which includes:
- Demographic and financial details of 10,000+ bank customers
- Target: Exited (1 = churned, 0 = retained)

Features used:
- Credit Score
- Geography
- Gender
- Age
- Tenure
- Balance
- Num of Products
- Has Credit Card
- Is Active Member
- Estimated Salary

---

## 🔍 Demo Screenshot


---

## 📌 Deployment
The model is deployed using Streamlit Community Cloud with all encoders and scalers serialized via pickle.

🔗 [Visit the Streamlit App](https://ann-classification-churn-3kq5m6c96mw9lz48bszpqv.streamlit.app/)

---

## 📜 License
This project is licensed under the terms of the GNU License. See the LICENSE file for details.

---
👨‍💻 Author
Mohammed Saim Ahmed Quadri
📫 [LinkedIn](https://www.linkedin.com/in/msaquadri)
📧 mohammedsaimquadri@gmail.com

