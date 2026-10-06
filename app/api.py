"""API REST para predecir el riesgo de enfermedad cardíaca en tiempo real (Etapa 3).

Se ejecuta con:  uvicorn app.api:app --host 0.0.0.0 --port 8000
Documentación interactiva automática en:  http://localhost:8000/docs
"""

from pathlib import Path  # Construcción de rutas independiente del sistema operativo
from typing import Literal  # Permite restringir un campo a un conjunto fijo de valores

import joblib  # Carga del modelo entrenado guardado en disco
import pandas as pd  # El modelo espera un DataFrame con los nombres de las columnas
from fastapi import FastAPI  # Framework web para construir la API
from pydantic import BaseModel, Field  # Validación automática de los datos de entrada

RUTA_MODELO = Path(__file__).resolve().parent / "model.joblib"  # Ruta del modelo, relativa a este archivo (funciona igual en local y en Docker)
UMBRAL = 0.5  # Probabilidad a partir de la cual se predice "enfermedad"

model = joblib.load(RUTA_MODELO)  # Carga una sola vez, al iniciar el servidor, el Pipeline completo (preprocesamiento + clasificador)

app = FastAPI(  # Crea la aplicación web
    title="API de predicción de enfermedad cardíaca",  # Título que aparece en la documentación /docs
    description="Devuelve la probabilidad de enfermedad cardíaca a partir de 11 variables clínicas.",  # Descripción de la API
    version="1.0.0",  # Versión de la API
)


class Paciente(BaseModel):  # Esquema de los datos de entrada: FastAPI rechaza con error 422 lo que no lo cumpla
    """Variables clínicas de un paciente (mismos nombres que el dataset de entrenamiento)."""

    Age: int = Field(..., ge=1, le=120, description="Edad en años")  # Entero obligatorio entre 1 y 120
    Sex: Literal["M", "F"] = Field(..., description="Sexo: M o F")  # Solo acepta M o F
    ChestPainType: Literal["TA", "ATA", "NAP", "ASY"] = Field(..., description="Tipo de dolor torácico")  # Una de las cuatro categorías del dataset
    RestingBP: int = Field(..., ge=0, le=300, description="Presión arterial en reposo (mm Hg); 0 = no medida")  # 0 se interpreta como dato faltante
    Cholesterol: int = Field(..., ge=0, le=1000, description="Colesterol sérico (mg/dl); 0 = no medido")  # 0 se interpreta como dato faltante
    FastingBS: Literal[0, 1] = Field(..., description="Glucosa en ayunas > 120 mg/dl: 1 = sí, 0 = no")  # Indicador binario
    RestingECG: Literal["Normal", "ST", "LVH"] = Field(..., description="Resultado del electrocardiograma en reposo")  # Una de tres categorías
    MaxHR: int = Field(..., ge=30, le=250, description="Frecuencia cardíaca máxima alcanzada")  # Entero en un rango fisiológico amplio
    ExerciseAngina: Literal["Y", "N"] = Field(..., description="Angina inducida por ejercicio: Y o N")  # Solo acepta Y o N
    Oldpeak: float = Field(..., ge=-10, le=10, description="Depresión del segmento ST inducida por ejercicio")  # Número decimal
    ST_Slope: Literal["Up", "Flat", "Down"] = Field(..., description="Pendiente del segmento ST en ejercicio")  # Una de tres categorías

    model_config = {  # Configuración del esquema
        "json_schema_extra": {  # Ejemplo que se muestra en la documentación interactiva
            "example": {  # Paciente de ejemplo listo para probar en /docs
                "Age": 54, "Sex": "M", "ChestPainType": "ASY", "RestingBP": 140, "Cholesterol": 239,  # Datos demográficos y de reposo
                "FastingBS": 0, "RestingECG": "Normal", "MaxHR": 120, "ExerciseAngina": "Y",  # Glucosa, ECG y prueba de esfuerzo
                "Oldpeak": 1.5, "ST_Slope": "Flat",  # Segmento ST
            }
        }
    }


class Prediccion(BaseModel):  # Esquema de la respuesta de la API
    """Resultado de la predicción."""

    heart_disease_probability: float  # Probabilidad estimada de enfermedad cardíaca (entre 0 y 1)
    prediction: int  # 1 si la probabilidad supera el umbral, 0 en caso contrario


@app.get("/health")  # Ruta GET /health: usada por Docker y Kubernetes para saber si el servicio está vivo
def health():  # Función que atiende la ruta /health
    """Comprobación de estado del servicio."""
    return {"status": "ok", "model": type(model.named_steps["clf"]).__name__}  # Responde que funciona e indica qué clasificador está cargado


@app.post("/predict", response_model=Prediccion)  # Ruta POST /predict: recibe un paciente y devuelve la predicción
def predict(data: Paciente):  # FastAPI convierte y valida el JSON recibido contra el esquema Paciente
    """Predice la probabilidad de enfermedad cardíaca de un paciente."""
    X = pd.DataFrame([data.model_dump()])  # Convierte el paciente en un DataFrame de una fila con los nombres de columna originales
    proba = float(model.predict_proba(X)[0][1])  # El Pipeline preprocesa y devuelve la probabilidad de la clase 1 (enfermedad)
    return {"heart_disease_probability": proba, "prediction": int(proba > UMBRAL)}  # Respuesta en formato JSON
