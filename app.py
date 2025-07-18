from flask import Flask, render_template, request
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

# Load the dataset
file_path = 'Datasets.csv'
df = pd.read_csv(file_path)

# Handle missing values
numerical_columns = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
imputer = SimpleImputer(strategy='mean')
df[numerical_columns] = imputer.fit_transform(df[numerical_columns])

# Drop rows with missing values in the target variable (isFraud)
df.dropna(subset=['isFraud'], inplace=True)

# Separate features (X) and target variable (y)
X = df.drop(columns=['isFraud'])
y = df['isFraud']

# Define preprocessing steps for different types of columns
numeric_features = X.select_dtypes(include=['float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

# Apply preprocessing transformations using ColumnTransformer
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# Preprocess and transform the entire dataset
X_preprocessed = preprocessor.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

# Train models and store them in a dictionary
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Neural Network': MLPClassifier()
}

for name, model in models.items():
    model.fit(X_train, y_train)

# Initialize Flask application
app = Flask(__name__)

# Define endpoint for index page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input data
        features = request.form['features']

        # Split input features into key-value pairs
        feature_values = features.split(',')
        input_dict = {}
        for feature in feature_values:
            key, value = feature.split('=')
            input_dict[key.strip()] = float(value.strip())  # Convert values to appropriate data type

        # Create DataFrame from input data
        input_df = pd.DataFrame([input_dict])

        # Ensure input_df columns match expected columns after preprocessing
        expected_columns = X.columns  # Assuming X is your original DataFrame before preprocessing
        if not set(input_df.columns).issubset(expected_columns):
            raise ValueError(f"Input features contain unexpected columns. Expected columns: {', '.join(expected_columns)}")

        # Preprocess input features
        processed_features = preprocessor.transform(input_df)

        # Make prediction using each model
        predictions = {}
        for name, model in models.items():
            prediction = model.predict(processed_features)
            predictions[name] = int(prediction[0])

        return render_template('index.html', predictions=predictions)

    except Exception as e:
        error_message = str(e)
        return render_template('index.html', error_message=error_message)


# Run Flask application
if __name__ == '__main__':
    app.run(debug=True)
