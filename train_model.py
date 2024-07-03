from data_processing import load_data, preprocess_data, omit_empty_columns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
import joblib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Dictionary of models to use
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(),
    'Support Vector Regression': SVR(),
    'Gradient Boosting': GradientBoostingRegressor()
}

def train_model(data, model_name='Linear Regression'):
    """
    Train a model based on the provided model name.

    Parameters:
    - data (pd.DataFrame): The preprocessed data containing the features and target variable.
    - model_name (str): The name of the model to train.

    Returns:
    - model: The trained model.
    - X_test: The test features.
    - y_test: The test target.
    """
    try:
        # Feature engineering: Extract month and day of week from the date
        data['month'] = data['date'].dt.month
        data['day_of_week'] = data['date'].dt.dayofweek

        # Selecting relevant features for prediction
        features = ['month', 'day_of_week']
        X = data[features]
        y = data['nkill']  # Target variable

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Get the selected model
        model = models.get(model_name)
        if model is None:
            raise ValueError(f"Model '{model_name}' is not supported.")
        
        # Train the model
        model.fit(X_train, y_train)

        # Save the model and the feature names
        joblib.dump((model, features), f'terrorism_model_{model_name}.pkl')
        logging.info(f"{model_name} model trained and saved successfully")
        return model, X_test, y_test
    except Exception as e:
        logging.error(f"Error training model: {e}")
        raise
