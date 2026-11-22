# Used Car Price Prediction - Project 2

## Dataset
Place `car data.csv` in the same folder as the Python script. The dataset used here
contains 299 records after duplicate removal and predicts `Selling_Price`.

## Run
1. Install Python 3.10+.
2. Open a terminal in this folder.
3. Install packages: `pip install -r requirements.txt`
4. Run: `python used_car_price_model.py`

## Models compared
- Multiple Linear Regression (baseline)
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor

The selected model is chosen using the lowest MAE on a held-out 20% test set.
The saved model file in this package was fitted on all cleaned records after
the comparison. For a formal report, quote the test-set comparison, not training
performance.

## Important interpretation
The target values are kept in the same units as the dataset's `Selling_Price`.
Do not label them as Ghana cedis unless the dataset is converted and validated
against Ghana-market prices. The dataset is small (299 records), so
results are preliminary and should not be treated as a guaranteed market quote.

`Vehicle_Age` is calculated as 2026 minus the manufacturing year. If you update
the reference year, update it consistently in training and prediction.
