import pickle
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the best model
best_model_filename = 'trained_models/best_model.pkl'
with open(best_model_filename, 'rb') as f:
    best_model = pickle.load(f)

# Load the CSV file with the data
data_file = '../Adapt/output/transformed_data_with_ner_not_unknown.csv'
df = pd.read_csv(data_file)

# Define the features to match the training data
categorical_features = ['Name', 'Brand']
numerical_features = ['Asking']

# Ensure the data format matches the training data
for col in numerical_features:
    if col not in df.columns:
        raise ValueError(f"Missing numerical feature in the input data: {col}")
    df[col] = df[col].fillna(0)

for col in categorical_features:
    if col not in df.columns:
        raise ValueError(f"Missing categorical feature in the input data: {col}")
    df[col] = df[col].fillna('Unknown')

# Check if 'Model Year' and 'Date' are in the pipeline, and handle their absence
if 'Model Year' not in df.columns:
    df['Model Year'] = 0
if 'Date' not in df.columns:
    df['Date'] = pd.Timestamp('1970-01-01').timestamp()  # Default to Unix epoch start

# Preprocess the data using the preprocessor in the pipeline
data_preprocessed = best_model.named_steps['preprocessor'].transform(df)

# Make predictions
predictions = best_model.named_steps['regressor'].predict(data_preprocessed)

# Add the predictions to the DataFrame
df['Prediction'] = predictions
df['Difference'] = df['Prediction'] - df['Asking']

# Save the results to a new CSV file
output_file = 'output/predictions_with_differences.csv'
df.to_csv(output_file, index=False, float_format='%.2f')

print(f"Predictions complete. Results saved to '{output_file}'")