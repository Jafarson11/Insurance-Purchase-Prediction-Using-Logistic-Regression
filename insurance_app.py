import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv('insurance.csv')
    return data

# Preprocess the dataset
def preprocess_data(data):
    # Convert categorical columns to numeric
    data['sex'] = data['sex'].apply(lambda x: 1 if x == 'male' else 0)
    data['smoker'] = data['smoker'].apply(lambda x: 0 if x == 'yes' else 1)
    data = pd.get_dummies(data, columns=['region'], drop_first=True)
    
    # Create a new binary target column 'purchase' based on a condition (example: expenses > threshold)
    threshold = data['expenses'].median()
    data['purchase'] = (data['expenses'] > threshold).astype(int)
    
    return data

# Build and train logistic regression model
def build_and_train_model(data):
    X = data.drop(['expenses', 'purchase'], axis=1)
    y = data['purchase']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, scaler, accuracy, X.columns

# User input for prediction
def get_user_input(feature_columns):
    age = st.sidebar.slider('Age', 18, 100, 30)
    sex = st.sidebar.selectbox('Sex', ['male', 'female'])
    bmi = st.sidebar.slider('BMI', 10.0, 50.0, 25.0)
    children = st.sidebar.slider('Children', 0, 10, 0)
    smoker = st.sidebar.selectbox('Smoker', ['yes', 'no'])
    region = st.sidebar.selectbox('Region', ['northwest', 'northeast', 'southeast', 'southwest'])
    
    user_data = {
        'age': age,
        'sex': 1 if sex == 'male' else 0,
        'bmi': bmi,
        'children': children,
        'smoker': 1 if smoker == 'yes' else 0,
        'region_northwest': 1 if region == 'northwest' else 0,
        'region_northeast': 1 if region == 'northeast' else 0,
        'region_southeast': 1 if region == 'southeast' else 0,
        'region_southwest': 1 if region == 'southwest' else 0
    }
    
    # Ensure all columns are present
    user_input_df = pd.DataFrame(user_data, index=[0])
    user_input_df = user_input_df.reindex(columns=feature_columns, fill_value=0)
    
    return user_input_df

# Main function to run the app
def main():
    st.title("Insurance Purchase Prediction System")
    
    # Load and preprocess data
    data = load_data()
    data = preprocess_data(data)
    
    # Display the data
    st.write("### Insurance Data")
    st.write(data.head())
    
    # Data visualization
    st.write("### Data Visualization")
    
    fig, ax = plt.subplots()
    sns.countplot(x='purchase', data=data, ax=ax)
    ax.set_title('Distribution of Purchase Decision')
    st.pyplot(fig)
    
    fig, ax = plt.subplots()
    sns.histplot(data['age'], kde=True, ax=ax)
    ax.set_title('Age Distribution')
    st.pyplot(fig)
    
    fig, ax = plt.subplots()
    sns.boxplot(x='purchase', y='bmi', data=data, ax=ax)
    ax.set_title('BMI vs Purchase Decision')
    st.pyplot(fig)
    
    # Build and train model
    model, scaler, accuracy, feature_columns = build_and_train_model(data)
    st.write(f"Model Accuracy: {accuracy:.2f}")
    
    # Get user input
    user_input = get_user_input(feature_columns)
    st.write("### User Input Parameters")
    st.write(user_input)
    
    # Scale user input
    user_input_scaled = scaler.transform(user_input)
    
    # Predict and display result
    prediction = model.predict(user_input_scaled)
    prediction_proba = model.predict_proba(user_input_scaled)
    
    st.write("### Prediction")
    purchase_decision = "will" if prediction[0] == 1 else "will not"
    color = "green" if prediction[0] == 1 else "red"
    st.markdown(f"<h3 style='color: {color};'>The model predicts that the customer {purchase_decision} purchase insurance.</h3>", unsafe_allow_html=True)
    
    st.write("### Prediction Probability")
    st.write(f"Probability of not purchasing: {prediction_proba[0][0]:.2f}")
    st.write(f"Probability of purchasing: {prediction_proba[0][1]:.2f}")

    # Plot probability of purchasing
    fig, ax = plt.subplots()
    probabilities = prediction_proba[0]
    bars = plt.bar(['Not Purchase', 'Purchase'], probabilities, color=[color if i == prediction[0] else 'grey' for i in range(2)])
    ax.set_title('Probability of Purchase')
    ax.set_ylim([0, 1])
    ax.bar_label(bars, fmt='%.2f')
    st.pyplot(fig)

if __name__ == "__main__":
    main()
