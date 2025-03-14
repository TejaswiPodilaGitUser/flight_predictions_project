# Flight Price and Customer Satisfaction Prediction


##Streamlit app

- **Flight Price Prediction**:
<img width="1685" alt="image" src="https://github.com/user-attachments/assets/884c918d-9df9-43e2-afbd-40be2070f2c8" />

<img width="1659" alt="image" src="https://github.com/user-attachments/assets/5252913d-d2e5-498a-8c46-5d852b319dad" />

- **Customer Satisfaction Prediction**:
  
<img width="1665" alt="image" src="https://github.com/user-attachments/assets/74b4d548-8a87-4af3-990e-d59aed24eb16" />

<img width="1655" alt="image" src="https://github.com/user-attachments/assets/680a9ec8-4375-45d9-bc2a-df906e617252" />



This repository contains a **Flight Price Prediction** and **Customer Satisfaction Prediction** system built using machine learning models and deployed as a Streamlit application. The project includes data preprocessing, feature engineering, model training, evaluation, and visualization.

## 📌 Features
- **Flight Price Prediction**: Predicts the price of flight tickets based on historical data.
- **Customer Satisfaction Prediction**: Classifies whether a passenger is satisfied or not.
- **Multiple Machine Learning Models**: Compares 5-6 models for better performance.
- **EDA & Visualizations**: Displays key insights using Seaborn & Matplotlib.
- **Feature Engineering**: Extracts meaningful features for better model performance.
- **Data Cleaning & Preprocessing**: Handles missing values, outliers, and transformations.
- **SMOTE for Imbalanced Data**: Synthetic Minority Over-sampling Technique applied for balancing datasets.
- **Streamlit Web App**: Interactive UI for predictions.

## 📁 Project Structure
```bash
flight_prediction_project/
│── data/
│   ├── raw/  # Original dataset
│   ├── processed/  # Cleaned dataset
│── models/
│   ├── best_flight_price_model.pkl
│   ├── best_satisfaction_model.pkl
│── src/
│   ├── eda.py  # Exploratory Data Analysis
│   ├── model_training.py  # Model training and evaluation
│   ├── main.py  # Streamlit app
│── notebooks/
│   ├── EDA.ipynb  # Jupyter notebook for EDA
│── requirements.txt
│── README.md
│── .gitignore
```

## 📊 Datasets Used
- **Flight Price Dataset**: Contains airline, source, destination, duration, and ticket prices.
- **Customer Satisfaction Dataset**: Includes passenger demographics, flight experience, and satisfaction labels.

## 🛠️ Installation & Setup
### 1️⃣ Clone the Repository
```bash
git clone git@github.com:TejaswiPodilaGitUser/flight_predictions_project.git
cd flight_prediction_project
```

### 2️⃣ Create Virtual Environment & Install Dependencies
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

### 3️⃣ Run Exploratory Data Analysis (EDA)
```bash
python src/eda.py
```

### 4️⃣ Train Machine Learning Models
```bash
python src/model_training.py
```

### 5️⃣ Run Streamlit App
```bash
streamlit run src/main.py
```

## 📈 Model Training & Evaluation
- **Data Cleaning & Feature Engineering**: 
  - Handled missing values, outliers, and categorical encoding.
  - Created new features for improved model accuracy.
- **Models Used**: XGBoost, LightGBM, Random Forest, Decision Tree, SVM, and Logistic Regression.
- **SMOTE Applied**: Balanced dataset using Synthetic Minority Over-sampling Technique.
- **Best models are saved as:**
  - `models/best_flight_price_model.pkl`
  - `models/best_satisfaction_model.pkl`


  ![alt text](image.png)


Model Training Scores


![alt text](image-1.png)


![alt text](image-2.png)


- **Plots & Predictions**:
  - Feature importance visualization.
  - Prediction trend analysis.
  - Model performance comparison plots.
 
- **EDA**:
<img width="976" alt="image" src="https://github.com/user-attachments/assets/5d76492f-1dd0-4969-98eb-7db1ec8e5013" />

<img width="994" alt="image" src="https://github.com/user-attachments/assets/c0796c09-f7e8-4e53-b3e3-a48c2206ddbd" />


## 🖥️ Web App Features
- **Flight Price Prediction**: Users enter flight details to get a price estimate.
- **Customer Satisfaction Prediction**: Users enter passenger details to check satisfaction.
- **Data Insights & Visualizations**: Displays key patterns from the dataset.
- **Prediction Plots**: Visualizes model outputs and trends.

## 🔧 Future Enhancements
- Deploy on cloud (AWS/GCP/Heroku)
- Add real-time API integration
- Improve model accuracy with hyperparameter tuning

## 🤝 Contributing
Feel free to fork this repo and submit pull requests with improvements.

## 📜 License
This project is open-source 

---
**Author**: Tejaswi Podila
