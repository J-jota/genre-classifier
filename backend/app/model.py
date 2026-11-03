"""Treinamento do classificador de gêneros, seguindo o pipeline do notebook.

Replica as partes 2 a 8 do notebook: higienização do dataset, recorte de
rótulo único (Drama, Comedy, Documentary, Horror), TF-IDF, GridSearchCV com
Naive Bayes e KNN, e seleção do melhor modelo por F1 macro.
"""

import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42
MINIMO_POR_GENERO = 60
CLASSES_ESPERADAS = 4

CAMINHO_DATASET = Path(__file__).resolve().parent.parent.parent / "data" / "tmdb_5000_movies.csv"
CAMINHO_MODELO = Path(__file__).resolve().parent.parent / "modelo_genero.pkl"


def extrair_generos(conteudo):
    """Converte o campo JSON em uma lista de nomes de gêneros."""
    if pd.isna(conteudo):
        return []
    try:
        dados = json.loads(conteudo) if isinstance(conteudo, str) else conteudo
        return [
            str(item["name"]).strip()
            for item in dados
            if isinstance(item, dict) and item.get("name")
        ]
    except (json.JSONDecodeError, TypeError, KeyError):
        return []


def higienizar(caminho_dataset):
    """Reproduz a higienização do notebook (partes 2 e 3)."""
    df_original = pd.read_csv(caminho_dataset)

    df = df_original[["id", "title", "overview", "genres"]].copy()
    df["title"] = df["title"].fillna("").astype(str).str.strip()
    df["overview"] = df["overview"].fillna("").astype(str).str.strip()
    df = df[(df["title"] != "") & (df["overview"] != "")].copy()

    df["lista_generos"] = df["genres"].apply(extrair_generos)
    df = df[df["lista_generos"].str.len() > 0].copy()

    # Rótulo único: somente filmes com exatamente um gênero.
    df = df[df["lista_generos"].str.len() == 1].copy()
    df["genero"] = df["lista_generos"].str[0]

    df = df[df["overview"].str.len() >= 20].copy()
    df = df.drop_duplicates(subset=["title", "overview", "genero"]).copy()

    contagem = df["genero"].value_counts()
    generos_selecionados = contagem[contagem >= MINIMO_POR_GENERO].index.tolist()
    df = df[df["genero"].isin(generos_selecionados)].copy()

    df_modelagem = df.reset_index(drop=True)

    # Conferências de consistência (mesmas asserts do notebook).
    assert not df_modelagem[["title", "overview", "genero"]].isna().any().any()
    assert (df_modelagem["overview"].str.strip() != "").all()
    assert (df_modelagem["overview"].str.len() >= 20).all()
    assert not df_modelagem.duplicated(["title", "overview", "genero"]).any()
    assert df_modelagem["genero"].nunique() == CLASSES_ESPERADAS

    return df_modelagem[["id", "title", "overview", "genero"]]


def criar_tfidf():
    return TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        max_df=0.95,
        sublinear_tf=True,
    )


def calcular_metricas(y_real, y_previsto):
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_real, y_previsto, average="macro", zero_division=0
    )
    _, _, f1_weighted, _ = precision_recall_fscore_support(
        y_real, y_previsto, average="weighted", zero_division=0
    )
    return {
        "acuracia": round(float(accuracy_score(y_real, y_previsto)), 4),
        "precision_macro": round(float(precision_macro), 4),
        "recall_macro": round(float(recall_macro), 4),
        "f1_macro": round(float(f1_macro), 4),
        "f1_weighted": round(float(f1_weighted), 4),
    }


def treinar():
    """Treina e compara NB e KNN; retorna o melhor modelo e metadados."""
    df_modelagem = higienizar(CAMINHO_DATASET)

    X = df_modelagem["overview"]
    y = df_modelagem["genero"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    pipeline_nb = Pipeline([
        ("tfidf", criar_tfidf()),
        ("modelo", MultinomialNB()),
    ])
    pipeline_knn = Pipeline([
        ("tfidf", criar_tfidf()),
        ("modelo", KNeighborsClassifier(algorithm="brute", metric="cosine", n_jobs=-1)),
    ])

    busca_nb = GridSearchCV(
        estimator=pipeline_nb,
        param_grid={
            "tfidf__ngram_range": [(1, 1), (1, 2)],
            "tfidf__min_df": [1, 2],
            "modelo__alpha": [0.1, 0.5, 1.0],
        },
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
    )
    busca_knn = GridSearchCV(
        estimator=pipeline_knn,
        param_grid={
            "tfidf__ngram_range": [(1, 1), (1, 2)],
            "tfidf__min_df": [1, 2],
            "modelo__n_neighbors": [3, 5, 7, 11, 15],
            "modelo__weights": ["uniform", "distance"],
        },
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
    )

    buscas = {"Naive Bayes": busca_nb, "KNN": busca_knn}
    resultados = {}
    for nome, busca in buscas.items():
        busca.fit(X_train, y_train)
        previsto = busca.best_estimator_.predict(X_test)
        resultados[nome] = {
            "busca": busca,
            "metricas_teste": calcular_metricas(y_test, previsto),
            "melhores_params": {k: str(v) for k, v in busca.best_params_.items()},
            "f1_macro_cv": round(float(busca.best_score_), 4),
        }

    melhor_nome = max(resultados, key=lambda n: resultados[n]["metricas_teste"]["f1_macro"])
    melhor = resultados[melhor_nome]

    # Os modelos servidos ficam treinados APENAS nos 80% de treino. Assim,
    # as previsões sobre os filmes da lista (que são os 20% de teste) são
    # honestas: o modelo nunca viu essas sinopses, e os erros exibidos
    # refletem o desempenho real reportado nas métricas.
    modelos = {nome: r["busca"].best_estimator_ for nome, r in resultados.items()}

    # A lista de filmes do frontend contém somente o conjunto de teste.
    df_teste = df_modelagem.loc[X_test.index, ["id", "title", "overview", "genero"]]
    filmes = [
        {
            "id": int(row.id),
            "title": row.title,
            "overview": row.overview,
            "genero_real": row.genero,
        }
        for row in df_teste.itertuples()
    ]

    info = {
        "modelo_vencedor": melhor_nome,
        "algoritmos_disponiveis": list(modelos.keys()),
        "melhores_params": melhor["melhores_params"],
        "metricas_teste": melhor["metricas_teste"],
        "f1_macro_cv": melhor["f1_macro_cv"],
        "comparacao": {
            nome: {
                "metricas_teste": r["metricas_teste"],
                "f1_macro_cv": r["f1_macro_cv"],
                "melhores_params": r["melhores_params"],
            }
            for nome, r in resultados.items()
        },
        "total_filmes": len(df_modelagem),
        "total_treino": int(len(X_train)),
        "total_teste": int(len(X_test)),
        "generos": sorted(y.unique().tolist()),
        "distribuicao": y.value_counts().to_dict(),
    }

    return modelos, filmes, info


def carregar_ou_treinar():
    """Carrega o modelo salvo ou treina do zero e persiste em disco."""
    if CAMINHO_MODELO.exists():
        with open(CAMINHO_MODELO, "rb") as f:
            return pickle.load(f)
    pacote = treinar()
    with open(CAMINHO_MODELO, "wb") as f:
        pickle.dump(pacote, f)
    return pacote


def prever(modelo, sinopse):
    """Prevê o gênero de uma sinopse e retorna probabilidades por classe."""
    if not isinstance(sinopse, str) or not sinopse.strip():
        raise ValueError("Informe uma sinopse textual não vazia.")
    sinopse = sinopse.strip()
    genero = modelo.predict([sinopse])[0]
    if hasattr(modelo, "predict_proba"):
        proba = modelo.predict_proba([sinopse])[0]
        probabilidades = {
            classe: round(float(p), 4)
            for classe, p in zip(modelo.classes_, proba)
        }
    else:
        probabilidades = {genero: 1.0}
    return genero, probabilidades
