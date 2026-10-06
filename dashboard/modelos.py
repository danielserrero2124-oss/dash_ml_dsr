"""Entrenamiento y evaluación de los cuatro modelos que muestra el dashboard.

Todo el flujo evita la fuga de datos (data leakage) y controla el sobreajuste:

1. Los datos se dividen en train/test ANTES de cualquier transformación.
2. El preprocesamiento vive dentro de un Pipeline, así que GridSearchCV lo reajusta en
   cada pliegue usando solo la parte de entrenamiento.
3. Los hiperparámetros se eligen con validación cruzada estratificada (nunca con test).
4. El conjunto de prueba se usa una sola vez, para reportar.

Los resultados se guardan en ``dashboard/artefactos.joblib`` para que el dashboard abra
al instante. Para regenerarlos:  python -m dashboard.modelos
"""

from pathlib import Path  # Manejo de rutas de archivos

import joblib  # Guardado y carga de los artefactos en disco
import numpy as np  # Cálculo numérico
from sklearn.inspection import permutation_importance  # Importancia de variables por permutación
from sklearn.metrics import roc_auc_score, roc_curve  # Área bajo la curva ROC y puntos de la curva
from sklearn.model_selection import StratifiedKFold, learning_curve  # Validación cruzada estratificada y curvas de aprendizaje

from src import modelado as mod  # Funciones reutilizables del proyecto (Pipeline, GridSearchCV, partición)

RUTA_ARTEFACTOS = Path(__file__).resolve().parent / "artefactos.joblib"  # Archivo con modelos y métricas ya calculados
NOMBRES_MODELOS = ["LogisticRegression", "RandomForest", "KNN", "GradientBoosting"]  # Los cuatro modelos pedidos para el dashboard
ETIQUETAS_MODELOS = {  # Nombre completo de cada clasificador para mostrar en pantalla
    "LogisticRegression": "LogisticRegression",  # Regresión logística
    "RandomForest": "RandomForestClassifier",  # Bosque aleatorio
    "KNN": "KNeighborsClassifier",  # K vecinos más cercanos
    "GradientBoosting": "GradientBoostingClassifier",  # Boosting de árboles
}


def entrenar_todo():  # Entrena los cuatro modelos y calcula todo lo que el dashboard necesita
    """Devuelve un diccionario con la partición, los modelos ajustados y sus métricas."""
    X, y = mod.cargar_datos()  # Carga las 11 variables clínicas y el objetivo HeartDisease
    X_train, X_test, y_train, y_test = mod.dividir_datos(X, y)  # Partición 80/20 estratificada ANTES de transformar (sin fuga)
    catalogo = mod.obtener_modelos()  # Catálogo de modelos con sus mallas de hiperparámetros
    pliegues = StratifiedKFold(n_splits=5, shuffle=True, random_state=mod.SEMILLA)  # Mismos pliegues para las curvas de aprendizaje
    resultados = {}  # Diccionario nombre del modelo → resultados
    for nombre in NOMBRES_MODELOS:  # Recorre los cuatro modelos
        modelo, malla = catalogo[nombre]  # Estimador base y malla de búsqueda
        grid = mod.train_pipeline(X_train, y_train, modelo, malla)  # Pipeline + GridSearchCV con validación cruzada de 5 pliegues
        mejor = grid.best_estimator_  # Pipeline completo reentrenado con todo train usando los mejores hiperparámetros
        proba_train = mejor.predict_proba(X_train)[:, 1]  # Probabilidades sobre train (solo para medir el sobreajuste)
        proba_test = mejor.predict_proba(X_test)[:, 1]  # Probabilidades sobre test (evaluación final)
        fpr, tpr, umbrales = roc_curve(y_test, proba_test)  # Puntos de la curva ROC en test
        tamanos, auc_tr, auc_va = learning_curve(  # Curva de aprendizaje: AUC al entrenar con más o menos pacientes
            mejor, X_train, y_train,  # Pipeline ganador y datos de entrenamiento
            train_sizes=np.linspace(0.15, 1.0, 8),  # Ocho tamaños, del 15 % al 100 % de train
            cv=pliegues, scoring="roc_auc", n_jobs=-1,  # Validación cruzada estratificada, métrica AUC, todos los núcleos
        )
        imp = permutation_importance(mejor, X_test, y_test, scoring="roc_auc", n_repeats=20, random_state=mod.SEMILLA)  # Caída de AUC al desordenar cada variable
        indice = grid.best_index_  # Posición de la mejor combinación dentro de los resultados de la búsqueda
        resultados[nombre] = {  # Guarda todo lo relevante del modelo
            "modelo": mejor,  # Pipeline entrenado (se usa para predecir pacientes nuevos)
            "mejores_parametros": {k.replace("clf__", ""): v for k, v in grid.best_params_.items()},  # Hiperparámetros ganadores sin el prefijo del Pipeline
            "auc_cv": float(grid.best_score_),  # AUC medio de validación cruzada
            "auc_cv_desv": float(grid.cv_results_["std_test_score"][indice]),  # Desviación del AUC entre pliegues
            "auc_train": float(roc_auc_score(y_train, proba_train)),  # AUC sobre train (optimista por construcción)
            "auc_test": float(roc_auc_score(y_test, proba_test)),  # AUC sobre test (estimación honesta)
            "proba_test": proba_test,  # Probabilidades de test (para recalcular métricas con cualquier umbral)
            "roc": {"fpr": fpr, "tpr": tpr, "umbrales": umbrales},  # Curva ROC
            "aprendizaje": {  # Curva de aprendizaje resumida
                "n": tamanos,  # Número de pacientes de entrenamiento en cada punto
                "train_media": auc_tr.mean(axis=1), "train_desv": auc_tr.std(axis=1),  # AUC de entrenamiento: media y desviación
                "cv_media": auc_va.mean(axis=1), "cv_desv": auc_va.std(axis=1),  # AUC de validación: media y desviación
            },
            "importancia": dict(zip(X_test.columns, imp.importances_mean)),  # Importancia media de cada variable
        }
    return {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test, "resultados": resultados}  # Artefactos completos


def cargar_artefactos(forzar=False):  # Devuelve los artefactos, entrenando solo si hace falta
    """Lee dashboard/artefactos.joblib; si no existe (o forzar=True) entrena y lo guarda."""
    if RUTA_ARTEFACTOS.exists() and not forzar:  # Si ya hay resultados guardados...
        return joblib.load(RUTA_ARTEFACTOS)  # ...se cargan directamente (arranque inmediato)
    artefactos = entrenar_todo()  # En caso contrario se entrena todo (tarda alrededor de un minuto)
    joblib.dump(artefactos, RUTA_ARTEFACTOS, compress=3)  # Se guarda comprimido para la próxima vez
    return artefactos  # Devuelve los artefactos recién calculados


def metricas_con_umbral(y_real, proba, umbral):  # Métricas de clasificación para un umbral de decisión dado
    """Devuelve la matriz de confusión y las métricas derivadas al clasificar con proba >= umbral."""
    y_real = np.asarray(y_real)  # Etiquetas reales como arreglo
    pred = (np.asarray(proba) >= umbral).astype(int)  # Predicción: 1 si la probabilidad alcanza el umbral
    tp = int(((pred == 1) & (y_real == 1)).sum())  # Verdaderos positivos: enfermos detectados
    tn = int(((pred == 0) & (y_real == 0)).sum())  # Verdaderos negativos: sanos bien clasificados
    fp = int(((pred == 1) & (y_real == 0)).sum())  # Falsos positivos: sanos marcados como enfermos
    fn = int(((pred == 0) & (y_real == 1)).sum())  # Falsos negativos: enfermos no detectados
    return {  # Resultados para el umbral
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,  # Celdas de la matriz de confusión
        "accuracy": (tp + tn) / len(y_real),  # Proporción total de aciertos
        "sensibilidad": tp / (tp + fn) if tp + fn else 0.0,  # Proporción de enfermos detectados (recall)
        "especificidad": tn / (tn + fp) if tn + fp else 0.0,  # Proporción de sanos bien clasificados
        "precision": tp / (tp + fp) if tp + fp else 0.0,  # Proporción de alertas que son enfermos reales
    }


if __name__ == "__main__":  # Permite regenerar los artefactos desde la línea de comandos
    datos = cargar_artefactos(forzar=True)  # Entrena de nuevo y sobrescribe el archivo
    for nombre, r in datos["resultados"].items():  # Recorre los modelos
        print(f"{nombre:<20} AUC train={r['auc_train']:.4f}  CV={r['auc_cv']:.4f}  test={r['auc_test']:.4f}  {r['mejores_parametros']}")  # Resumen por modelo
