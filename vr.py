import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.svm import SVR
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Define the model dictionary
models = {
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'Support Vector Machine': SVR(),
    'XGBoost': xgb.XGBRegressor(random_state=42),
    'LightGBM': lgb.LGBMRegressor(random_state=42)
}

# Streamlit UI
st.title('Model Selection for Predictions')

# Load the data
@st.cache_data
def load_data():
    return pd.read_csv('train.csv')

df = load_data()

# Select features and target
X = df.drop(['SalePrice'], axis=1)
y = df['SalePrice']

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(exclude=['object']).columns

# Preprocessing for numerical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

# User selects models
selected_models = st.multiselect('Select models', list(models.keys()))

if selected_models:
    if len(selected_models) > 1:
        # Create a Voting Regressor with the selected models
        voting_regressor = VotingRegressor(
            estimators=[(name, models[name]) for name in selected_models]
        )
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', voting_regressor)
        ])
    else:
        # Single model case
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', models[selected_models[0]])
        ])

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Define MAPE function
    def mean_absolute_percentage_error(y_true, y_pred):
        nonzero_indices = y_true != 0
        return np.mean(np.abs((y_true[nonzero_indices] - y_pred[nonzero_indices]) / y_true[nonzero_indices])) * 100

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    # Display metrics
    st.subheader('Model Evaluation Metrics')
    st.write('Mean Absolute Error:', mae)
    st.write('Mean Squared Error:', mean_squared_error(y_test, y_pred))
    st.write('Root Mean Squared Error:', rmse)
    st.write('R^2 Score:', r2)
    st.write('Mean Absolute Percentage Error:', mape)

    # Optional: Display a sample of predictions vs true values
    if st.checkbox('Show predictions vs true values'):
        results_df = pd.DataFrame({'True Values': y_test, 'Predictions': y_pred})
        st.write(results_df.head())
else:
    st.write('Select one or more models to see results.')
