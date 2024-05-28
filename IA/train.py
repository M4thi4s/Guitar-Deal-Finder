# Necessary Imports
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import linear_model, ensemble, tree
from sklearn.metrics import r2_score
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

# Load in dataset
df = pickle.load(open("data/df.pkl", "rb"))

# Define categorical features to be used
categorical_features = ['Brand', 'Name']
# Define numerical features to be used
numerical_features = ['Asking']

# Display columns in the dataset
print("Columns in the dataset:", df.columns.tolist())

# Display unique values for each categorical feature
for feature in categorical_features:
    print(f"Unique values in '{feature}': {df[feature].unique()}")

# Print ten first entries of the dataset
print(df.head(10))

# Clean the dataset
# Fill missing values for numerical features with the median
for col in numerical_features:
    df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    df[col].fillna(df[col].median(), inplace=True)

# Fill missing values for categorical features with 'Unknown'
for col in categorical_features:
    df[col] = df[col].fillna('Unknown')

# Create X and y
y = df['Final']
X = df[categorical_features + numerical_features]

# Create test train splits
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
y_true = y_test

# Create dataframe to store data of different models
df_models = pd.DataFrame(columns=['Model Name', 
                                  'R-Squared 10-Fold C.V. Train', 
                                  'R-Squared Holdout',
                                  'R-Squared (adj) Holdout'])

# Directory to save models
model_save_dir = 'trained_models'
os.makedirs(model_save_dir, exist_ok=True)

# Check if all columns exist in the training data
missing_categorical_features = [col for col in categorical_features if col not in X_train.columns]
missing_numerical_features = [col for col in numerical_features if col not in X_train.columns]
if missing_categorical_features or missing_numerical_features:
    raise ValueError(f"Missing columns in the dataframe: {missing_categorical_features + missing_numerical_features}")

# Define the preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Running multiple models
models = {}

# Initializing models dict with different regressors
models['Linear Regression'] = linear_model.LinearRegression()
models['Ridge Regression'] = linear_model.Ridge()
models['Lasso Regression'] = linear_model.Lasso(alpha=.5)
models['Huber Regression'] = linear_model.HuberRegressor()
models['Decision Tree Regression'] = tree.DecisionTreeRegressor(max_depth=7)
models['Extra Trees Regression'] = ensemble.ExtraTreesRegressor(max_depth=7)
models['Random Forest Regression'] = ensemble.RandomForestRegressor()
models['AdaBoost Regression'] = ensemble.AdaBoostRegressor()
models['Gradient Boosting Regression'] = ensemble.GradientBoostingRegressor()

# Initialize variables to track the best model
best_model = None
best_model_name = ""
best_r2_score = -np.inf

# Using tqdm to display progress
for key, value in tqdm(models.items(), desc="Training Models"):
    # Create a pipeline that includes the preprocessor and the model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', value)
    ])
    
    # Running 10-fold cross validation and calculating mean R2 of the 10 trials
    scores = cross_val_score(pipeline, X_train, y_train, cv=10, scoring='r2')
    R2_CV = np.mean(scores)
    
    # Training the pipeline on the entire train set, and testing it on the holdout set
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    R2_holdout = r2_score(y_true, y_pred)
    
    # Calculating adjusted R^2 on holdout set
    n = len(y_train)
    p = X_train.shape[1]
    adj_R2_holdout = 1 - (1 - R2_holdout) * (n - 1) / (n - p - 1)
    
    # Round numbers to 3 decimals
    R2_CV = round(R2_CV, 3)
    R2_holdout = round(R2_holdout, 3)
    adj_R2_holdout = round(adj_R2_holdout, 3)
    
    # Printing output
    print(f'Model: {key}')
    print(f'C.V. R^2: {R2_CV}')
    print(f'Holdout R^2: {R2_holdout}')
    print(f'Holdout adj R^2: {adj_R2_holdout}')
    print()
    
    # Creating model dictionary to add to a dataframe row
    model_dict = {'Model Name': key,
                  'R-Squared 10-Fold C.V. Train': R2_CV,
                  'R-Squared Holdout': R2_holdout,
                  'R-Squared (adj) Holdout': adj_R2_holdout
                 }
    
    # Converting from dictionary to a pandas dataframe
    model_df = pd.DataFrame(model_dict, index=[0])
    
    # Appending to the overall dataframe
    df_models = df_models.append(model_df, ignore_index=True)

    # Check if this model is the best so far
    if R2_holdout > best_r2_score:
        best_r2_score = R2_holdout
        best_model = pipeline
        best_model_name = key

# Save the best model and preprocessor
best_model_filename = os.path.join(model_save_dir, 'best_model.pkl')
with open(best_model_filename, 'wb') as f:
    pickle.dump(best_model, f)

print(f"The best model is {best_model_name} with R^2 score of {best_r2_score}")

# Reorder for easy viewing
df_models = df_models[['Model Name', 
                       'R-Squared 10-Fold C.V. Train', 
                       'R-Squared Holdout', 
                       'R-Squared (adj) Holdout']]

# Sort by R-Squared (adj)
df_models.sort_values(by=['R-Squared (adj) Holdout'], inplace=True, ascending=False)

# Reset the indices
df_models = df_models.reset_index(drop=True)

# Print the dataframe
print(df_models)