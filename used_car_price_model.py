"""
Used Car Price Prediction - Project 2
Run: python used_car_price_model.py
The script trains and evaluates four regression models, then saves the
best model (selected by lowest test MAE).
"""
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_FILE = "car data.csv"
MODEL_FILE = "best_used_car_price_model.joblib"
REFERENCE_YEAR = 2026

df = pd.read_csv(DATA_FILE)
print("Original dataset shape:", df.shape)
print("Missing values before cleaning:\\n", df.isna().sum())
df = df.drop_duplicates().copy()
df["Vehicle_Age"] = REFERENCE_YEAR - df["Year"]

X = df.drop(columns=["Selling_Price", "Year"])
y = df["Selling_Price"]
categorical = X.select_dtypes(include=["object"]).columns.tolist()
numeric = X.select_dtypes(exclude=["object"]).columns.tolist()

preprocessor = ColumnTransformer([
    ("numeric", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numeric),
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

models = {
    "Multiple Linear Regression": LinearRegression(),
    "Decision Tree Regressor": DecisionTreeRegressor(
        random_state=42, max_depth=5, min_samples_leaf=2
    ),
    "Random Forest Regressor": RandomForestRegressor(
        n_estimators=300, random_state=42, min_samples_leaf=2
    ),
    "Gradient Boosting Regressor": GradientBoostingRegressor(
        random_state=42, n_estimators=200, learning_rate=0.05, max_depth=2
    )
}

results = []
for name, estimator in models.items():
    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("regressor", estimator)
    ])
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)
    results.append({
        "Model": name,
        "MAE": mean_absolute_error(y_test, pred),
        "MSE": mean_squared_error(y_test, pred),
        "RMSE": mean_squared_error(y_test, pred) ** 0.5,
        "R2": r2_score(y_test, pred)
    })

results_df = pd.DataFrame(results).sort_values("MAE")
print("\\nModel comparison (lower MAE/RMSE is better; higher R2 is better):")
print(results_df.to_string(index=False))
best_name = results_df.iloc[0]["Model"]
print("\\nBest model by test MAE:", best_name)

# Refit the selected model on all cleaned data before saving.
best_pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("regressor", models[best_name])
])
best_pipeline.fit(X, y)
joblib.dump(best_pipeline, MODEL_FILE)
print("Saved trained model to:", MODEL_FILE)

def predict_used_car_price(car_name, present_price, kms_driven,
                           fuel_type, seller_type, transmission,
                           owner, manufacturing_year):
    """Predict a price in the same units as Selling_Price in the dataset."""
    trained_model = joblib.load(MODEL_FILE)
    row = pd.DataFrame([{
        "Car_Name": car_name,
        "Present_Price": present_price,
        "Kms_Driven": kms_driven,
        "Fuel_Type": fuel_type,
        "Seller_Type": seller_type,
        "Transmission": transmission,
        "Owner": owner,
        "Vehicle_Age": REFERENCE_YEAR - manufacturing_year
    }])
    return float(trained_model.predict(row)[0])

# Example (replace these values with a real car's details):
# estimate = predict_used_car_price(
#     "ritz", 5.59, 27000, "Petrol", "Dealer", "Manual", 0, 2014
# )
# print("Estimated selling price:", estimate)
