from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.endpoints import api_router

app = FastAPI(
    title='AG News Classification (CBOW + MLP)',
    description='Full-Stack Academic ML Project using PyTorch CBOW embeddings, MLP Classifier, and TF-IDF Baseline.',
    version='1.0.0'
)

# Enable CORS for Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:5173', 'http://127.0.0.1:5173', '*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

app.include_router(api_router)

@app.get('/')
def root():
    return {
        'project': 'ANN-Based Classification of AG News Using CBOW Embedding + MLP',
        'status': 'online',
        'docs': '/docs'
    }

if __name__ == '__main__':
    import uvicorn
    uvicorn.run('main:app', host='0.0.0.0', port=8000, reload=True)
