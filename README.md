# Sales Forecasting Automation

End-to-end pipeline that cleans retail sales data, stores it in SQL,
trains regression models to forecast monthly sales, and serves
predictions through a REST API with a web dashboard.

## Tech Stack
Python, pandas, scikit-learn, SQLite, FastAPI, HTML/JS, Chart.js

## Pipeline
1. `generate_data.py` - builds a simulated Superstore-style dataset (6,000 orders)
2. `clean_and_store.py` - removes duplicates, fixes types, handles nulls, loads into SQLite, runs SQL aggregation
3. `train_models.py` - feature engineering (lags, rolling mean, seasonality), trains Linear, Ridge, Lasso with GridSearchCV and TimeSeriesSplit
4. `app.py` - FastAPI endpoints: `/forecast`, `/api/summary`, `/dashboard`

## Results
| Model | MAE | RMSE | R2 | MAPE |
|-------|-----|------|----|------|
| Linear | 20332 | 27148 | 0.501 | 8.89% |
| Ridge | 19093 | 25663 | 0.554 | 8.29% |
| Lasso | 19860 | 26562 | 0.522 | 8.58% |

Ridge performed best (time-based split, last 6 months as test).

## Run
pip install pandas numpy scikit-learn matplotlib fastapi uvicorn joblib
python generate_data.py
python clean_and_store.py
python train_models.py
uvicorn app:app --reload

Open http://127.0.0.1:8000 and http://127.0.0.1:8000/dashboard

## Note
Data is simulated to mirror the Superstore dataset structure.