# Classificador de Gêneros por Sinopse

Monorepo com backend (FastAPI) e frontend simples para prever o gênero de um
filme a partir da sinopse, usando o pipeline do notebook
`Classificação de Gêneros por Sinopse.ipynb` (TF-IDF + Naive Bayes/KNN).

## Estrutura

    genre-classifier/
    ├── backend/            API FastAPI + treinamento do modelo
    │   ├── app/
    │   │   ├── main.py     endpoints e servidor
    │   │   └── model.py    higienização, TF-IDF, GridSearchCV, previsão
    │   └── modelo_genero.pkl  (gerado no primeiro treino)
    ├── frontend/           página estática (seleção de filme + resultado)
    └── data/
        └── tmdb_5000_movies.csv

## Pré-requisitos

- **Python 3.11+** instalado.
- **uv** (gerenciador de pacotes/ambientes). Instalação, conforme o sistema:

      # Linux/macOS
      curl -LsSf https://astral.sh/uv/install.sh | sh

      # Windows (PowerShell)
      powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

      # ou via pip
      pip install uv

- O dataset `data/tmdb_5000_movies.csv` presente no repositório.

## Como rodar

    cd backend
    uv venv .venv
    uv pip install --python .venv/bin/python fastapi "uvicorn[standard]" scikit-learn pandas joblib
    .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000

No primeiro boot o modelo é treinado (a partir do CSV em `data/`) e salvo em
`backend/modelo_genero.pkl`; nos boots seguintes é carregado do disco.

Abra http://localhost:8000 no navegador.

### Troubleshooting

- Se você alterar o código de higienização, treino ou parâmetros em
  `backend/app/model.py`, apague `backend/modelo_genero.pkl` antes de subir o
  servidor — caso contrário o modelo antigo em cache continuará sendo servido.

## Endpoints

    GET  /api/movies              lista de filmes do conjunto de TESTE (20%,
                                  nunca usados no treinamento — previsões
                                  honestas, com erros reais possíveis)
    GET  /api/movies/{id}         detalhe de um filme (sinopse + gênero real)
    POST /api/predict             body: {"movie_id": 123} ou {"overview": "..."},
                                  com "algorithm": "Naive Bayes" ou "KNN" (opcional;
                                  usa o vencedor se omitido)
    GET  /api/model               info do modelo vencedor, parâmetros e métricas
    GET  /api/health              status

## Notas

- O recorte de modelagem segue o notebook: 781 filmes de gênero único nas
  classes Comedy, Documentary, Drama e Horror (mínimo de 60 exemplos/classe),
  divididos em 80% treino (624) e 20% teste (157), estratificados.
- Os modelos servidos são treinados APENAS com o treino. O frontend lista
  somente os filmes de teste, então cada previsão é fora da amostra e os
  erros exibidos correspondem às métricas do notebook (NB ~68% de acurácia,
  KNN ~59%).
- O modelo vencedor é escolhido por F1 macro no conjunto de teste.
