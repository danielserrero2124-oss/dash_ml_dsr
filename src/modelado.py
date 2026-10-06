"""Funciones reutilizables de carga, preprocesamiento, entrenamiento y evaluación.

Los dos notebooks del proyecto importan este módulo para no repetir código
(requisito "separar código en funciones reutilizables" de la Etapa 1).
"""

from pathlib import Path  # Manejo de rutas de archivos independiente del sistema operativo

import pandas as pd  # Lectura del CSV y manejo de tablas de datos
from sklearn.compose import ColumnTransformer  # Aplica transformaciones distintas a grupos de columnas
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier  # Modelos de ensamble de árboles
from sklearn.impute import SimpleImputer  # Imputación de valores faltantes
from sklearn.linear_model import LogisticRegression  # Regresión logística
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score  # Métricas de clasificación
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split  # Búsqueda de hiperparámetros, validación cruzada y partición
from sklearn.naive_bayes import GaussianNB  # Naive Bayes gaussiano
from sklearn.neighbors import KNeighborsClassifier  # K vecinos más cercanos
from sklearn.pipeline import Pipeline  # Encadena preprocesamiento y modelo en un solo objeto
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder  # Escalado a [0, 1] y codificación one-hot
from sklearn.svm import SVC  # Máquina de vectores de soporte
from sklearn.tree import DecisionTreeClassifier  # Árbol de decisión

# ---------------------------------------------------------------------------
# Constantes del proyecto
# ---------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parents[1]  # Carpeta raíz del proyecto (un nivel arriba de src/)
RUTA_DATOS = RAIZ / "data" / "heart.csv"  # Ruta relativa del dataset de Kaggle
RUTA_MODELO_API = RAIZ / "app" / "model.joblib"  # Donde la API espera encontrar el modelo entrenado
RUTA_MODELO_RAIZ = RAIZ / "model.joblib"  # Copia del modelo en la raíz (estructura pedida en la Etapa 0)
RUTA_REPORTE_DRIFT = RAIZ / "drift_report.html"  # Reporte de deriva de datos de Evidently
SEMILLA = 42  # Semilla fija para que las particiones y los modelos sean reproducibles
OBJETIVO = "HeartDisease"  # Nombre de la variable objetivo (1 = enfermedad cardíaca, 0 = sano)

COLUMNAS_CON_CEROS = ["RestingBP", "Cholesterol"]  # Numéricas donde el 0 significa "dato no medido" (fisiológicamente imposible)
COLUMNAS_NUMERICAS = ["Age", "FastingBS", "MaxHR", "Oldpeak"]  # Numéricas sin valores faltantes
COLUMNAS_CATEGORICAS = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]  # Variables de texto
COLUMNAS_ENTRADA = ["Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", "FastingBS",  # Orden original de las 11 variables
                    "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope"]  # predictoras del dataset


def cargar_datos(ruta=RUTA_DATOS):  # Lee el CSV y separa predictoras y objetivo
    """Devuelve (X, y): X con las 11 variables clínicas e y con HeartDisease."""
    df = pd.read_csv(ruta)  # Carga el archivo CSV completo en un DataFrame
    X = df[COLUMNAS_ENTRADA].copy()  # Copia las columnas predictoras (sin la variable objetivo)
    y = df[OBJETIVO].copy()  # Copia la variable objetivo binaria
    return X, y  # Entrega ambas partes por separado


def dividir_datos(X, y, proporcion_test=0.2):  # Partición train/test ANTES de cualquier transformación
    """Partición estratificada 80/20 con semilla fija (evita fuga de información del test)."""
    return train_test_split(  # Devuelve X_train, X_test, y_train, y_test
        X, y,  # Datos a dividir
        test_size=proporcion_test,  # Fracción reservada para la evaluación final
        stratify=y,  # Conserva la proporción de enfermos y sanos en ambos conjuntos
        random_state=SEMILLA,  # Hace que la partición sea siempre la misma
    )


def construir_preprocesador():  # Define las transformaciones que se aprenden SOLO con los datos de entrenamiento
    """ColumnTransformer: imputación de ceros + escalado MinMax + codificación one-hot."""
    rama_ceros = Pipeline([  # Tratamiento de RestingBP y Cholesterol
        ("imputar", SimpleImputer(missing_values=0, strategy="median", add_indicator=True)),  # Cambia los 0 por la mediana y agrega una columna que marca dónde faltaba el dato
        ("escalar", MinMaxScaler()),  # Lleva cada columna al rango [0, 1]
    ])
    return ColumnTransformer([  # Une las tres ramas de preprocesamiento
        ("ceros", rama_ceros, COLUMNAS_CON_CEROS),  # Rama para columnas con ceros que significan "faltante"
        ("numericas", MinMaxScaler(), COLUMNAS_NUMERICAS),  # Rama para el resto de numéricas: solo escalado
        ("categoricas", OneHotEncoder(handle_unknown="ignore"), COLUMNAS_CATEGORICAS),  # Rama categórica: una columna 0/1 por categoría; ignora categorías nuevas
    ])


def train_pipeline(X_train, y_train, model, param_grid, cv=5, scoring="roc_auc"):  # Entrena un modelo dentro de un Pipeline con GridSearchCV
    """Devuelve el GridSearchCV ajustado; el preprocesamiento se reajusta dentro de cada pliegue."""
    pipe = Pipeline([("prep", construir_preprocesador()), ("clf", model)])  # Encadena preprocesamiento y clasificador
    pliegues = StratifiedKFold(n_splits=cv, shuffle=True, random_state=SEMILLA)  # Validación cruzada estratificada y reproducible
    grid = GridSearchCV(pipe, param_grid, cv=pliegues, scoring=scoring, n_jobs=-1)  # Prueba todas las combinaciones de hiperparámetros usando todos los núcleos
    grid.fit(X_train, y_train)  # En cada pliegue ajusta el preprocesamiento solo con la parte de entrenamiento (sin fuga)
    return grid  # Devuelve la búsqueda ya ajustada (incluye el mejor modelo reentrenado con todo train)


def evaluar_modelo(grid, X_test, y_test):  # Calcula las métricas de un modelo ya entrenado sobre el conjunto de prueba
    """Devuelve un diccionario con AUC, accuracy, precisión, recall, F1 y el AUC de validación cruzada."""
    proba = grid.predict_proba(X_test)[:, 1]  # Probabilidad estimada de enfermedad para cada paciente de test
    pred = grid.predict(X_test)  # Clase predicha (umbral 0.5)
    return {  # Métricas de desempeño
        "auc_cv": grid.best_score_,  # AUC promedio de validación cruzada del mejor conjunto de hiperparámetros
        "auc_test": roc_auc_score(y_test, proba),  # Área bajo la curva ROC en test
        "accuracy_test": accuracy_score(y_test, pred),  # Proporción de aciertos en test
        "precision_test": precision_score(y_test, pred),  # De los predichos enfermos, cuántos lo están
        "recall_test": recall_score(y_test, pred),  # De los enfermos reales, cuántos detecta
        "f1_test": f1_score(y_test, pred),  # Media armónica de precisión y recall
        "mejores_parametros": grid.best_params_,  # Hiperparámetros ganadores
    }


def obtener_modelos():  # Catálogo de modelos y sus mallas de hiperparámetros
    """Devuelve {nombre: (estimador, param_grid)} con todos los clasificadores del proyecto."""
    return {  # Cada entrada: nombre → (modelo base, malla de búsqueda con prefijo "clf__")
        "SVC": (SVC(probability=True, random_state=SEMILLA),  # SVM con salida de probabilidades (modelo base del enunciado)
                {"clf__C": [0.1, 1, 10], "clf__gamma": [0.01, 0.1]}),  # Malla del enunciado: regularización y ancho del kernel
        "LogisticRegression": (LogisticRegression(max_iter=1000),  # Regresión logística con suficientes iteraciones para converger
                               {"clf__C": [0.01, 0.1, 1, 10]}),  # Inverso de la fuerza de regularización L2
        "RandomForest": (RandomForestClassifier(random_state=SEMILLA),  # Bosque aleatorio
                         {"clf__n_estimators": [100, 300], "clf__max_depth": [4, 8, None], "clf__min_samples_leaf": [1, 5]}),  # Nº de árboles, profundidad y tamaño mínimo de hoja
        "KNN": (KNeighborsClassifier(),  # K vecinos más cercanos (requiere variables escaladas)
                {"clf__n_neighbors": [5, 11, 21, 31], "clf__weights": ["uniform", "distance"]}),  # Nº de vecinos y ponderación por distancia
        "GradientBoosting": (GradientBoostingClassifier(random_state=SEMILLA),  # Boosting de árboles
                             {"clf__n_estimators": [100, 200], "clf__learning_rate": [0.05, 0.1], "clf__max_depth": [2, 3]}),  # Nº de árboles, tasa de aprendizaje y profundidad
        "DecisionTree": (DecisionTreeClassifier(random_state=SEMILLA),  # Árbol de decisión simple
                         {"clf__max_depth": [3, 5, 8], "clf__min_samples_leaf": [1, 5, 20]}),  # Profundidad y tamaño mínimo de hoja
        "NaiveBayes": (GaussianNB(),  # Naive Bayes gaussiano
                       {"clf__var_smoothing": [1e-9, 1e-7, 1e-5]}),  # Suavizado de las varianzas
    }


def comparar_modelos(X_train, y_train, X_test, y_test, modelos=None):  # Entrena y evalúa todos los modelos
    """Devuelve (ranking, busquedas): tabla ordenada por AUC de validación cruzada y los GridSearchCV ajustados."""
    modelos = modelos or obtener_modelos()  # Usa el catálogo completo si no se indica otro
    filas, busquedas = [], {}  # Acumuladores de resultados y de objetos ajustados
    for nombre, (modelo, malla) in modelos.items():  # Recorre cada clasificador
        grid = train_pipeline(X_train, y_train, modelo, malla)  # Entrena con Pipeline + GridSearchCV
        busquedas[nombre] = grid  # Guarda la búsqueda ajustada
        filas.append({"modelo": nombre, **evaluar_modelo(grid, X_test, y_test)})  # Agrega sus métricas a la tabla
    ranking = pd.DataFrame(filas).sort_values("auc_cv", ascending=False).reset_index(drop=True)  # Ordena de mejor a peor según el AUC de validación cruzada
    ranking.index = ranking.index + 1  # La posición del ranking empieza en 1
    ranking.index.name = "posicion"  # Nombre del índice
    return ranking, busquedas  # Tabla comparativa y modelos ajustados
