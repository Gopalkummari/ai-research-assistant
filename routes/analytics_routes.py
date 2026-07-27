from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from src.database.base import get_db
from src.vector_store.manager import VectorStoreManager
from src.analytics.metrics import AnalyticsEngine

router = APIRouter(prefix="/analytics", tags=["System Analytics"])
vector_store = VectorStoreManager()

@router.get("/stats")
def get_analytics(db: Session = Depends(get_db)):
    analytics_engine = AnalyticsEngine(db, vector_store)
    stats = analytics_engine.get_system_stats()
    return stats
