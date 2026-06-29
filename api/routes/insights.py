from fastapi import APIRouter

router = APIRouter()

@router.get("/segmentation")
def get_segmentation():
    return {"message": "Segmentation insights endpoint"}
