from fastapi import FastAPI
from api.routes import predict, insights, mlops

app = FastAPI(
    title="RetainIQ Enterprise Inference API",
    description="High-performance customer retention scoring and lifetime value prediction service powered by RetainIQ.",
    version="2.1.0",
)

app.include_router(predict.router, prefix="/api/v1/predict", tags=["Predict"])
app.include_router(insights.router, prefix="/api/v1/insights", tags=["Insights"])
app.include_router(mlops.router)

@app.get("/")
def root():
    return {"message": "Welcome to the RetainIQ Customer Churn & CLV Prediction API"}

