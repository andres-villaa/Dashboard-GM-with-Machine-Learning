from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import uvicorn, os

app = FastAPI(title="GM Toluca — Predictor de Ruptura")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── CARGAR Y ENTRENAR MODELO AL INICIAR ───────────────────────────────────
print("Cargando datos y entrenando modelo...")

DATA_PATH = os.path.join(os.path.dirname(__file__), "data_clean_v2.csv")
df = pd.read_csv(DATA_PATH)

FEATURES = [
    "VIDA TOTAL DE HTTA",
    "PIEZAS MAQUINADAS",
    "VIDA DESPERDICIADA",
    "% DE VIDA ULTILIZADA",
]
TARGET = "DESCRIPCION DE LA FALLA EN LA HERRAMIENTA"

X = df[FEATURES]
y = (df[TARGET] == "Rota").astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=3,
    class_weight="balanced",
    random_state=42,
)
rf.fit(X_train_sc, y_train)
print(f"Modelo entrenado con {len(X_train)} registros.")


# ── SCHEMAS ───────────────────────────────────────────────────────────────
class PredictInput(BaseModel):
    vida_total: float
    piezas_maquinadas: float


# ── ENDPOINTS ─────────────────────────────────────────────────────────────
@app.get("/")
def serve_dashboard():
    return FileResponse(
        os.path.join(os.path.dirname(__file__), "static", "index.html")
    )


@app.post("/predict")
def predict(data: PredictInput):
    vt = data.vida_total
    pm = min(data.piezas_maquinadas, vt)

    vida_desperdiciada = vt - pm
    pct_vida = (pm / vt * 100) if vt > 0 else 0

    entrada = pd.DataFrame([{
        "VIDA TOTAL DE HTTA":   vt,
        "PIEZAS MAQUINADAS":    pm,
        "VIDA DESPERDICIADA":   vida_desperdiciada,
        "% DE VIDA ULTILIZADA": pct_vida,
    }])

    entrada_sc = scaler.transform(entrada)
    prob = float(rf.predict_proba(entrada_sc)[0][1])
    pct  = round(prob * 100, 1)

    if prob >= 0.70:
        nivel = "Alto"
        recomendacion = "Revisar la herramienta antes del siguiente ciclo."
    elif prob >= 0.45:
        nivel = "Medio"
        recomendacion = "Realizar inspección visual antes del próximo turno."
    else:
        nivel = "Bajo"
        recomendacion = "No se requiere acción inmediata. Continuar con normalidad."

    return {
        "probabilidad": pct,
        "nivel": nivel,
        "recomendacion": recomendacion,
        "vida_desperdiciada": round(vida_desperdiciada),
        "pct_vida": round(pct_vida, 1),
    }


@app.get("/historico")
def historico():
    records = []
    for _, row in df.iterrows():
        vt = float(row["VIDA TOTAL DE HTTA"])
        pm = float(row["PIEZAS MAQUINADAS"])
        pm = min(pm, vt)
        vd  = vt - pm
        pct = (pm / vt * 100) if vt > 0 else 0

        entrada = pd.DataFrame([{
            "VIDA TOTAL DE HTTA":   vt,
            "PIEZAS MAQUINADAS":    pm,
            "VIDA DESPERDICIADA":   vd,
            "% DE VIDA ULTILIZADA": pct,
        }])
        prob  = float(rf.predict_proba(scaler.transform(entrada))[0][1])
        nivel = "Alto" if prob >= 0.70 else "Medio" if prob >= 0.45 else "Bajo"

        records.append({
            "falla":      str(row[TARGET]),
            "vida_total": int(vt),
            "piezas":     int(pm),
            "pct_vida":   round(pct, 1),
            "prob":       round(prob * 100, 1),
            "nivel":      nivel,
        })
    return records


@app.get("/stats")
def stats():
    total = len(df)
    rota  = int((df[TARGET] == "Rota").sum())
    pct_vida_media = round(float(df["% DE VIDA ULTILIZADA"].mean()), 1)
    return {
        "total_fallas":    total,
        "pct_rota":        round(rota / total * 100, 1),
        "vida_prom":       pct_vida_media,
        "registros_train": len(X_train),
        "registros_test":  len(X_test),
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
