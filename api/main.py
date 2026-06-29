from fastapi import FastAPI
from api.routes import predict, insights

app = FastAPI(title="Customer Churn & CLV Prediction API")

app.include_router(predict.router, prefix="/api/v1/predict", tags=["Predict"])
app.include_router(insights.router, prefix="/api/v1/insights", tags=["Insights"])

@app.get("/")
def root():
    return {"message": "Welcome to the Customer Churn & CLV Prediction API"}
