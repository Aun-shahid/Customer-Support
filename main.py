import fastapi
from routes.search import router as search_router

app = fastapi.FastAPI()


app.include_router(search_router, prefix="/api", tags=["search"])


@app.get("/")
async def read_root():
    
    return {"Hello": "World"}


@app.get("/health")
def health_check():
    return {"status": "ok"}