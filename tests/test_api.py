"""Pruebas automáticas de la API (se ejecutan con `pytest tests/` y en GitHub Actions)."""

from fastapi.testclient import TestClient  # Cliente que llama a la API en memoria, sin levantar un servidor

from app.api import app  # Aplicación FastAPI que se va a probar

client = TestClient(app)  # Crea el cliente de pruebas conectado a la aplicación

PACIENTE_ALTO_RIESGO = {  # Perfil clínico típico de enfermedad cardíaca
    "Age": 65, "Sex": "M", "ChestPainType": "ASY", "RestingBP": 150, "Cholesterol": 280,  # Hombre mayor, dolor asintomático
    "FastingBS": 1, "RestingECG": "ST", "MaxHR": 105, "ExerciseAngina": "Y",  # Glucosa alta, baja frecuencia máxima, angina de esfuerzo
    "Oldpeak": 2.5, "ST_Slope": "Flat",  # Depresión ST marcada y pendiente plana
}
PACIENTE_BAJO_RIESGO = {  # Perfil clínico típico de persona sana
    "Age": 35, "Sex": "F", "ChestPainType": "ATA", "RestingBP": 118, "Cholesterol": 190,  # Mujer joven, angina atípica
    "FastingBS": 0, "RestingECG": "Normal", "MaxHR": 175, "ExerciseAngina": "N",  # Glucosa normal, alta frecuencia máxima, sin angina
    "Oldpeak": 0.0, "ST_Slope": "Up",  # Sin depresión ST y pendiente ascendente
}


def test_health():  # Prueba que el servicio responde y tiene un modelo cargado
    respuesta = client.get("/health")  # Llama a la ruta de estado
    assert respuesta.status_code == 200  # El código HTTP debe ser 200 (correcto)
    assert respuesta.json()["status"] == "ok"  # El cuerpo debe indicar que el servicio está bien


def test_predict_devuelve_probabilidad_valida():  # Prueba el formato de la respuesta de /predict
    respuesta = client.post("/predict", json=PACIENTE_ALTO_RIESGO)  # Envía un paciente válido
    assert respuesta.status_code == 200  # La petición debe ser aceptada
    cuerpo = respuesta.json()  # Convierte la respuesta JSON en diccionario
    assert 0.0 <= cuerpo["heart_disease_probability"] <= 1.0  # La probabilidad debe estar entre 0 y 1
    assert cuerpo["prediction"] in (0, 1)  # La predicción debe ser binaria
    assert cuerpo["prediction"] == int(cuerpo["heart_disease_probability"] > 0.5)  # La predicción debe ser coherente con el umbral 0.5


def test_predict_ordena_bien_el_riesgo():  # Prueba que el modelo tiene sentido clínico
    alto = client.post("/predict", json=PACIENTE_ALTO_RIESGO).json()  # Predicción del paciente de alto riesgo
    bajo = client.post("/predict", json=PACIENTE_BAJO_RIESGO).json()  # Predicción del paciente de bajo riesgo
    assert alto["heart_disease_probability"] > bajo["heart_disease_probability"]  # El de alto riesgo debe tener mayor probabilidad
    assert alto["prediction"] == 1  # Debe clasificarse como enfermedad
    assert bajo["prediction"] == 0  # Debe clasificarse como sano


def test_predict_acepta_colesterol_no_medido():  # Prueba el caso de dato faltante codificado como 0
    paciente = {**PACIENTE_ALTO_RIESGO, "Cholesterol": 0}  # Mismo paciente sin medición de colesterol
    respuesta = client.post("/predict", json=paciente)  # Envía la petición
    assert respuesta.status_code == 200  # El modelo imputa internamente y debe responder


def test_predict_rechaza_categoria_invalida():  # Prueba la validación de valores categóricos
    paciente = {**PACIENTE_ALTO_RIESGO, "Sex": "X"}  # Sexo con un valor no permitido
    respuesta = client.post("/predict", json=paciente)  # Envía la petición
    assert respuesta.status_code == 422  # FastAPI debe rechazarla con error de validación


def test_predict_rechaza_campos_faltantes():  # Prueba la validación de campos obligatorios
    paciente = {k: v for k, v in PACIENTE_ALTO_RIESGO.items() if k != "Age"}  # Paciente sin la edad
    respuesta = client.post("/predict", json=paciente)  # Envía la petición
    assert respuesta.status_code == 422  # Debe rechazarse por faltar un campo
