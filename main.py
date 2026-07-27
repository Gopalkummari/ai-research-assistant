import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config.settings import settings
from src.database.base import Base, engine
from routes.document_routes import router as doc_router
from routes.search_routes import router as search_router
from routes.analysis_routes import router as analysis_router
from routes.analytics_routes import router as analytics_router

# Initialize database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title=settings.APP_NAME,
    description="Enterprise AI Research & Knowledge Assistant with RAG, Citations, TensorFlow Document Classification, and Analytics.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Routers
app.include_router(doc_router)
app.include_router(search_router)
app.include_router(analysis_router)
app.include_router(analytics_router)

@app.get("/", tags=["Health"])
def root_health_check():
    return {
        "status": "healthy",
        "app_name": settings.APP_NAME,
        "docs": "/docs",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
