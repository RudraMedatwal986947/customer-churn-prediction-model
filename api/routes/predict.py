from fastapi import APIRouter

router = APIRouter()

@router.post("/churn")
def predict_churn():
    return {"message": "Churn prediction endpoint"}

@router.post("/clv")
def predict_clv():
    return {"message": "CLV prediction endpoint"}
