"""API FastAPI do classificador de gêneros cinematográficos.

Endpoints:
  GET  /api/health       — status do serviço e informações do modelo
  GET  /api/movies       — lista de filmes do dataset de modelagem
  GET  /api/movies/{id}  — detalhe de um filme
  POST /api/predict      — prevê o gênero por sinopse ou por id de filme
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .model import carregar_ou_treinar, prever

FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"

estado = {}


@asynccontextmanager
async def lifespan(app):
    modelos, filmes, info = carregar_ou_treinar()
    estado["modelos"] = modelos
    estado["filmes"] = filmes
    estado["info"] = info
    estado["por_id"] = {f["id"]: f for f in filmes}
    yield


app = FastAPI(title="Classificador de Gêneros por Sinopse", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    movie_id: int | None = Field(default=None, description="ID do filme no dataset")
    overview: str | None = Field(default=None, description="Sinopse em inglês")
    algorithm: str | None = Field(
        default=None,
        description="Algoritmo: 'Naive Bayes' ou 'KNN'. Usa o vencedor se omitido.",
    )


@app.get("/api/health")
def health():
    return {"status": "ok", "modelo": estado["info"]["modelo_vencedor"]}


@app.get("/api/movies")
def listar_filmes():
    """Lista enxuta para o seletor do frontend (título, id e gênero real)."""
    return [
        {"id": f["id"], "title": f["title"], "genero_real": f["genero_real"]}
        for f in estado["filmes"]
    ]


@app.get("/api/movies/{movie_id}")
def detalhar_filme(movie_id: int):
    filme = estado["por_id"].get(movie_id)
    if filme is None:
        raise HTTPException(status_code=404, detail="Filme não encontrado.")
    return filme


@app.post("/api/predict")
def predict(req: PredictRequest):
    nome_modelo = req.algorithm or estado["info"]["modelo_vencedor"]
    modelo = estado["modelos"].get(nome_modelo)
    if modelo is None:
        disponiveis = ", ".join(estado["modelos"])
        raise HTTPException(
            status_code=400,
            detail=f"Algoritmo '{nome_modelo}' inválido. Disponíveis: {disponiveis}.",
        )

    if req.movie_id is not None:
        filme = estado["por_id"].get(req.movie_id)
        if filme is None:
            raise HTTPException(status_code=404, detail="Filme não encontrado.")
        overview = filme["overview"]
    elif req.overview:
        overview = req.overview
    else:
        raise HTTPException(status_code=400, detail="Informe movie_id ou overview.")

    try:
        genero, probabilidades = prever(modelo, overview)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    resposta = {
        "algoritmo": nome_modelo,
        "genero_previsto": genero,
        "probabilidades": probabilidades,
        "overview": overview,
    }
    if req.movie_id is not None:
        resposta["filme"] = estado["por_id"][req.movie_id]
    return resposta


@app.get("/api/model")
def info_modelo():
    return estado["info"]


app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
