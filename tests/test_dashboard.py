"""Pruebas del dashboard: verifican que las funciones de actualización (callbacks) responden sin errores."""

import pytest  # Marco de pruebas (permite repetir una prueba con varios valores)

from dashboard import app as dash_app  # Aplicación Dash con sus funciones de actualización
from dashboard import modelos as md  # Nombres de los modelos y métricas

PACIENTE = [54, "M", "ASY", 140, 239, 0, "Normal", 120, "Y", 1.5, "Flat"]  # Paciente de ejemplo en el orden de las columnas de entrada


def test_los_cuatro_modelos_estan_entrenados():  # Comprueba que existen los modelos pedidos
    assert set(dash_app.RES) == set(md.NOMBRES_MODELOS)  # LogisticRegression, RandomForest, KNN y GradientBoosting
    for resultado in dash_app.RES.values():  # Recorre los resultados de cada modelo
        assert 0.85 < resultado["auc_test"] <= 1.0  # El AUC en test debe ser alto y válido
        assert abs(resultado["auc_cv"] - resultado["auc_test"]) < 0.05  # Validación y test deben coincidir (sin sobreajuste oculto)


@pytest.mark.parametrize("nombre", md.NOMBRES_MODELOS)  # Repite la prueba con cada uno de los cuatro modelos
def test_actualizar_modelos(nombre):  # Prueba la actualización de la sección ML Models
    salida = dash_app.actualizar_modelos(nombre, 0.5)  # Simula elegir el modelo con umbral 0.5
    assert len(salida) == 13  # Debe devolver los 13 elementos que espera la interfaz
    assert all(elemento is not None for elemento in salida)  # Ninguno puede venir vacío


def test_el_umbral_cambia_la_sensibilidad():  # Prueba el efecto del control deslizante
    y, proba = dash_app.Y_TEST, dash_app.RES["RandomForest"]["proba_test"]  # Etiquetas y probabilidades de test
    bajo = md.metricas_con_umbral(y, proba, 0.2)  # Métricas con umbral bajo
    alto = md.metricas_con_umbral(y, proba, 0.8)  # Métricas con umbral alto
    assert bajo["sensibilidad"] >= alto["sensibilidad"]  # Un umbral más bajo detecta al menos tantos enfermos
    assert bajo["especificidad"] <= alto["especificidad"]  # ...a costa de clasificar peor a los sanos
    assert bajo["tp"] + bajo["tn"] + bajo["fp"] + bajo["fn"] == len(y)  # Las cuatro celdas suman el total de pacientes


def test_figuras_del_eda():  # Prueba las figuras interactivas del análisis exploratorio
    assert dash_app.actualizar_numerica("Cholesterol", "violin", ["sin_ceros"]).data  # Distribución numérica con datos
    assert dash_app.actualizar_categorica("ST_Slope", "tasa").data  # Tasa de enfermedad por categoría
    assert dash_app.actualizar_categorica("Sex", "conteo").data  # Conteo por clase
    assert dash_app.actualizar_dispersion("Age", "MaxHR").data  # Diagrama de dispersión


def test_prediccion_de_paciente():  # Prueba el formulario de predicción individual
    figura, texto = dash_app.predecir_paciente("RandomForest", 0.5, *PACIENTE)  # Predicción para el paciente de ejemplo
    assert 0 <= figura.data[0].value <= 100  # El medidor muestra un porcentaje válido
    assert "Riesgo" in texto  # El texto indica el nivel de riesgo
    _, aviso = dash_app.predecir_paciente("RandomForest", 0.5, None, *PACIENTE[1:])  # Mismo paciente sin la edad
    assert "Completa" in aviso  # Debe pedir que se completen los campos
