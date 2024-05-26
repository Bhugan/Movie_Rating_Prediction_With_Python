import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import chardet

# Function to detect file encoding
def detect_encoding(file):
    raw_data = file.read()
    result = chardet.detect(raw_data)
    return result['encoding']

# File uploader widget
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Detect the encoding of the uploaded file
    encoding = detect_encoding(uploaded_file)
    uploaded_file.seek(0)  # Reset file pointer to the beginning
    df = pd.read_csv(uploaded_file, encoding=encoding)
    
    # Display the dataframe
    st.write("DataFrame Preview:")
    st.write(df.head())

    # Handle missing values
    df = df.dropna(subset=['Rating'])  # Drop rows where the target variable is missing
    df['Votes'] = pd.to_numeric(df['Votes'], errors='coerce')
    df = df.dropna(subset=['Votes'])  # Ensure 'Votes' is numeric and drop rows where it's not

    # Define features and target variable
    X = df[['Genre', 'Director', 'Votes']]
    y = df['Rating']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocess the data (OneHotEncoding for categorical features)
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['Genre', 'Director'])
        ],
        remainder='passthrough'  # Keep the Votes column as it is
    )

    # Create a pipeline with preprocessing and regression
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

    # Train the model
    pipeline.fit(X_train, y_train)

    # Make predictions
    y_pred = pipeline.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Display evaluation metrics
    st.write(f'Mean Squared Error: {mse}')
    st.write(f'R^2 Score: {r2}')

    st.write("## Predict Movie Rating")
    
    # Input form for new movie data
    genre = st.text_input("Genre")
    director = st.text_input("Director")
    votes = st.number_input("Votes", min_value=0)

    if st.button("Predict Rating"):
        # Make a prediction based on user input
        input_data = pd.DataFrame([[genre, director, votes]], columns=['Genre', 'Director', 'Votes'])
        prediction = pipeline.predict(input_data)
        st.write(f"Predicted Rating: {prediction[0]:.2f}")

# Run the Streamlit app
if __name__ == '__main__':
    st.title('IMDb Movies Rating Predictor')
    st.write("Upload your IMDb Movies India CSV file to get started.")
