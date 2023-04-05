import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from data_preprocessing import preprocess_data

def train_model(data_file):
    # Load raw data
    data = pd.read_csv(data_file)

    #Pre-process it
    data = preprocess_data(data)

    # Split data into training and test sets
    X = data.drop(['total_revenue'], axis=1)
    y = data['total_revenue']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define preprocessing steps
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, ['total_adspend']),
        ('cat', categorical_transformer, ['network_id', 'country_id', 'month'])
    ])

    # Define the model
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    score = model.score(X_test, y_test)
    print(f'R-squared score: {score:.2f}')
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean squared error: {mse:.2f}')
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f'Root Mean Squared Error:", {rmse:.2f}')

    return model





