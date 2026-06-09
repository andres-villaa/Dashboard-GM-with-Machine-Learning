# GM Toluca — Predictor de Ruptura de Herramienta
## Random Forest · FastAPI · Dashboard

---

## Estructura del proyecto

```
gm_dashboard/
├── main.py              ← Backend FastAPI + modelo RF
├── requirements.txt     ← Dependencias Python
├── data_clean_v2.csv    ← Datos limpios de GM (coloca aquí)
└── static/
    └── index.html       ← Dashboard frontend
```

---

## Setup (una sola vez)

```bash
# 1. Crear entorno virtual
python -m venv .venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate

# 2. Instalar dependencias
pip install -r requirements.txt
```

---

## Correr el servidor

```bash
python main.py
```

El servidor arranca en: **http://localhost:8000**

Abre esa URL en tu navegador y el dashboard carga automáticamente.

---

## Endpoints disponibles

| Endpoint | Método | Descripción |
|---|---|---|
| `/` | GET | Dashboard HTML |
| `/predict` | POST | Predicción del RF dado vida_total y piezas_maquinadas |
| `/historico` | GET | Predicciones del RF sobre todos los registros del CSV |
| `/stats` | GET | Estadísticas generales del dataset |

---

## Ejemplo de llamada al predictor

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"vida_total": 10000, "piezas_maquinadas": 5000}'
```

Respuesta:
```json
{
  "probabilidad": 64.3,
  "nivel": "Alto",
  "recomendacion": "Revisar la herramienta antes del siguiente ciclo.",
  "vida_desperdiciada": 5000,
  "pct_vida": 50.0
}
```

---

## Notas importantes

- El modelo se entrena automáticamente al iniciar el servidor con `data_clean_v2.csv`
- No hay nada hardcodeado — cada predicción la calcula el Random Forest real
- El historial muestra las predicciones del modelo sobre los 283 registros reales
