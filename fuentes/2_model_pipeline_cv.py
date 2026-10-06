# %% [markdown]
# # Etapa 2 · Modelado con validación segura, exportación del modelo y monitoreo
#
# Este cuaderno contiene:
#
# 1. División de los datos **antes** del escalado.
# 2. `Pipeline` + `GridSearchCV` con validación cruzada estratificada.
# 3. Evaluación del modelo elegido: matriz de confusión, curva ROC y AUC.
# 4. Exportación del modelo para la API (Etapa 3).
# 5. Reporte de deriva de datos con Evidently (Etapa 6).

# %%
import sys  # Acceso a la configuración del intérprete de Python
sys.path.insert(0, "..")  # Agrega la carpeta raíz del proyecto para poder importar el paquete src
import shutil  # Copia de archivos
import warnings  # Control de mensajes de advertencia
import joblib  # Guardado y carga del modelo entrenado
import numpy as np  # Cálculo numérico
import pandas as pd  # Manejo de tablas de datos
import matplotlib.pyplot as plt  # Creación de gráficos
from sklearn.inspection import permutation_importance  # Importancia de variables por permutación
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix, roc_auc_score, roc_curve  # Métricas y gráficos de evaluación
from evidently.metric_preset import DataDriftPreset  # Conjunto de métricas de deriva de datos de Evidently
from evidently.report import Report  # Generador de reportes de Evidently
from src import modelado as mod  # Funciones reutilizables del proyecto

warnings.filterwarnings("ignore")  # Oculta advertencias que no afectan los resultados
pd.set_option("display.width", 160)  # Ancho de impresión de las tablas

# %% [markdown]
# ## 1. División de los datos antes de cualquier transformación

# %%
X, y = mod.cargar_datos()  # Carga las 11 variables clínicas y el objetivo HeartDisease
X_train, X_test, y_train, y_test = mod.dividir_datos(X, y)  # Partición 80/20 estratificada con semilla fija
print(f"Train: {X_train.shape[0]} pacientes ({y_train.mean():.1%} con enfermedad)")  # Tamaño y balance del conjunto de entrenamiento
print(f"Test:  {X_test.shape[0]} pacientes ({y_test.mean():.1%} con enfermedad)")  # Tamaño y balance del conjunto de prueba

# %% [markdown]
# El conjunto de prueba queda apartado desde este momento. La mediana para imputar, los
# mínimos y máximos del escalado y las categorías del *one-hot* se aprenden solo con train
# (y, dentro de la validación cruzada, solo con los pliegues de entrenamiento).

# %% [markdown]
# ## 2. `Pipeline` + `GridSearchCV`
#
# El `Pipeline` tiene dos pasos: `prep` (preprocesamiento) y `clf` (clasificador).

# %%
mod.construir_preprocesador()  # Muestra el diagrama del preprocesador: imputación de ceros, escalado MinMax y one-hot

# %%
ranking, busquedas = mod.comparar_modelos(X_train, y_train, X_test, y_test)  # Entrena los siete modelos con validación cruzada de 5 pliegues
ranking[["modelo", "auc_cv", "auc_test", "accuracy_test", "recall_test", "mejores_parametros"]].round(4)  # Ranking por AUC de validación cruzada

# %%
nombre_mejor = ranking.iloc[0]["modelo"]  # Modelo con mayor AUC de validación cruzada (elegido SIN mirar el test)
grid_mejor = busquedas[nombre_mejor]  # Búsqueda ajustada de ese modelo
modelo_final = grid_mejor.best_estimator_  # Pipeline completo (preprocesamiento + clasificador) reentrenado con todo train
print(f"Modelo elegido: {nombre_mejor}")  # Nombre del modelo ganador
print(f"Hiperparámetros: {grid_mejor.best_params_}")  # Hiperparámetros ganadores
print(f"AUC de validación cruzada: {grid_mejor.best_score_:.4f}")  # Desempeño estimado con train
detalle = pd.DataFrame(grid_mejor.cv_results_)[["params", "mean_test_score", "std_test_score", "rank_test_score"]]  # Resultado de cada combinación de hiperparámetros
detalle.sort_values("rank_test_score").head(5)  # Las cinco mejores combinaciones

# %% [markdown]
# ## 3. Evaluación en el conjunto de prueba
#
# ### AUC y reporte de clasificación

# %%
proba_test = modelo_final.predict_proba(X_test)[:, 1]  # Probabilidad de enfermedad para cada paciente de test
pred_test = modelo_final.predict(X_test)  # Clase predicha con umbral 0.5
auc_test = roc_auc_score(y_test, proba_test)  # Área bajo la curva ROC en test
print(f"AUC en test: {auc_test:.4f}\n")  # Reporta el AUC
print(classification_report(y_test, pred_test, target_names=["Sano (0)", "Enfermedad (1)"]))  # Precisión, recall y F1 por clase

# %% [markdown]
# ### Matriz de confusión

# %%
matriz = confusion_matrix(y_test, pred_test)  # Tabla de aciertos y errores: filas = clase real, columnas = clase predicha
ConfusionMatrixDisplay(matriz, display_labels=["Sano", "Enfermedad"]).plot(cmap="Blues", colorbar=False)  # Dibuja la matriz
plt.title(f"Matriz de confusión – {nombre_mejor} (test)")  # Título
plt.grid(False)  # Quita la cuadrícula superpuesta
plt.show()  # Muestra el gráfico
tn, fp, fn, tp = matriz.ravel()  # Extrae las cuatro celdas
print(f"Verdaderos negativos: {tn} | Falsos positivos: {fp} | Falsos negativos: {fn} | Verdaderos positivos: {tp}")  # Resumen numérico
print(f"Sensibilidad (recall): {tp / (tp + fn):.3f} | Especificidad: {tn / (tn + fp):.3f}")  # Tasas de detección de enfermos y de sanos

# %% [markdown]
# ### Curva ROC

# %%
fig, ax = plt.subplots(figsize=(7.5, 6.5))  # Figura para las curvas ROC
for nombre, grid in busquedas.items():  # Recorre los siete modelos
    p = grid.predict_proba(X_test)[:, 1]  # Probabilidades del modelo en test
    fpr, tpr, _ = roc_curve(y_test, p)  # Tasas de falsos y verdaderos positivos para cada umbral
    grosor, estilo = (3, "-") if nombre == nombre_mejor else (1.2, "--")  # Resalta el modelo elegido
    ax.plot(fpr, tpr, lw=grosor, ls=estilo, label=f"{nombre} (AUC = {roc_auc_score(y_test, p):.3f})")  # Dibuja la curva del modelo
ax.plot([0, 1], [0, 1], color="gray", lw=1, ls=":")  # Diagonal de referencia (clasificador al azar)
ax.set_xlabel("Tasa de falsos positivos (1 − especificidad)")  # Eje X
ax.set_ylabel("Tasa de verdaderos positivos (sensibilidad)")  # Eje Y
ax.set_title("Curvas ROC en test")  # Título
ax.legend(loc="lower right")  # Leyenda con el AUC de cada modelo
plt.show()  # Muestra el gráfico

# %% [markdown]
# ### ¿Qué variables usa el modelo?
#
# Importancia por permutación en test: cuánto cae el AUC al desordenar cada variable.

# %%
imp = permutation_importance(modelo_final, X_test, y_test, scoring="roc_auc", n_repeats=30, random_state=mod.SEMILLA)  # Permuta cada variable 30 veces y mide la caída de AUC
importancia = pd.DataFrame({"media": imp.importances_mean, "desv": imp.importances_std}, index=X_test.columns).sort_values("media")  # Caída media de AUC y su desviación por variable
importancia["media"].plot.barh(figsize=(7, 4), color="#4C72B0", xerr=importancia["desv"])  # Barras horizontales con barra de error
plt.xlabel("Caída media del AUC al permutar la variable")  # Eje X
plt.title("Importancia por permutación (test)")  # Título
plt.show()  # Muestra el gráfico

# %% [markdown]
# **Interpretación de la evaluación.**
#
# * **Modelo elegido:** RandomForest (300 árboles, hojas de al menos 5 pacientes), con AUC de
#   validación cruzada 0,9345 ± 0,028. En test obtiene **AUC = 0,930**, casi idéntico: la
#   validación cruzada estimó bien el desempeño y no hay señales de sobreajuste.
# * **Matriz de confusión (184 pacientes):** 96 verdaderos positivos, 69 verdaderos negativos,
#   13 falsos positivos y 6 falsos negativos. La **sensibilidad es 94,1 %** (detecta 96 de 102
#   enfermos) y la especificidad 84,1 %. En un tamizaje clínico el error más grave es el
#   falso negativo (un enfermo no detectado), así que este balance es razonable; si se quisiera
#   reducirlos aún más, bastaría bajar el umbral de 0,5 a costa de más falsos positivos.
# * **Curva ROC:** todas las curvas, salvo la del árbol individual, se superponen (AUC entre
#   0,91 y 0,94), lo que confirma que la señal está en los datos y no en un algoritmo concreto.
# * **Variables:** `ST_Slope` domina con claridad (permutarla reduce el AUC en 0,11),
#   seguida de `ChestPainType`, `Sex`, `Cholesterol` y `ExerciseAngina`. Coincide con el
#   análisis exploratorio. La edad casi no aporta una vez conocidas las demás variables.

# %% [markdown]
# ## 4. Exportación del modelo (para la Etapa 3)
#
# Se guarda el `Pipeline` completo: la API recibe los datos crudos del paciente y el propio
# modelo aplica el preprocesamiento aprendido en train.

# %%
joblib.dump(modelo_final, mod.RUTA_MODELO_API)  # Guarda el Pipeline en app/model.joblib (lo carga la API)
shutil.copyfile(mod.RUTA_MODELO_API, mod.RUTA_MODELO_RAIZ)  # Copia en la raíz del proyecto (estructura de la Etapa 0)
print(f"Modelo guardado: {mod.RUTA_MODELO_API} ({mod.RUTA_MODELO_API.stat().st_size / 1024:.0f} KB)")  # Confirma la ruta y el tamaño
recargado = joblib.load(mod.RUTA_MODELO_API)  # Vuelve a cargar el archivo para verificar que funciona
assert np.allclose(recargado.predict_proba(X_test)[:, 1], proba_test)  # El modelo recargado da exactamente las mismas probabilidades
paciente = X_test.iloc[[0]]  # Un paciente de ejemplo (DataFrame de una fila)
print(paciente.to_dict(orient="records")[0])  # Datos de entrada tal como los recibirá la API
print(f"Probabilidad de enfermedad: {recargado.predict_proba(paciente)[0, 1]:.3f} | etiqueta real: {y_test.iloc[0]}")  # Predicción de ejemplo

# %% [markdown]
# ## 5. Monitoreo de deriva de datos con Evidently (Etapa 6)
#
# Evidently compara la distribución de cada variable entre un conjunto de **referencia**
# (train) y uno **actual** (los datos que llegan al modelo). Aquí el conjunto actual es
# test, como indica el enunciado; en producción serían las peticiones recibidas por la API.

# %%
reporte = Report(metrics=[DataDriftPreset()])  # Reporte con el conjunto estándar de pruebas de deriva
reporte.run(reference_data=X_train, current_data=X_test)  # Compara cada variable de train contra test
reporte.save_html(str(mod.RUTA_REPORTE_DRIFT))  # Guarda el reporte interactivo en drift_report.html


def resumen_deriva(rep):  # Extrae del reporte una tabla resumida por variable
    datos = rep.as_dict()["metrics"]  # Resultados del reporte como diccionario
    general = datos[0]["result"]  # Resumen global: cuántas columnas derivaron
    por_columna = datos[1]["result"]["drift_by_columns"]  # Detalle de cada variable
    tabla = pd.DataFrame(por_columna).T[["stattest_name", "drift_score", "drift_detected"]]  # Prueba usada, valor p o distancia y decisión
    tabla.columns = ["prueba", "puntuación", "deriva"]  # Nombres en español
    print(f"Columnas con deriva: {general['number_of_drifted_columns']} de {general['number_of_columns']} "  # Conteo de columnas con deriva
          f"| ¿deriva del dataset?: {general['dataset_drift']}")  # Decisión global
    return tabla  # Devuelve la tabla resumida


resumen_deriva(reporte)  # Muestra el resumen de train frente a test

# %% [markdown]
# ### Simulación de deriva
#
# Para comprobar que el monitoreo detecta cambios reales, se simula una población distinta:
# pacientes 8 años mayores, con colesterol 25 % más alto y con una mezcla diferente de tipos de
# dolor torácico.

# %%
rng = np.random.default_rng(mod.SEMILLA)  # Generador aleatorio reproducible
X_derivado = X_test.copy()  # Copia de los datos actuales
X_derivado["Age"] = X_derivado["Age"] + 8  # Población más envejecida
X_derivado["Cholesterol"] = (X_derivado["Cholesterol"] * 1.25).round()  # Colesterol más alto (los ceros siguen en cero)
X_derivado["ChestPainType"] = rng.choice(["ASY", "NAP", "ATA", "TA"], size=len(X_derivado), p=[0.25, 0.25, 0.25, 0.25])  # Tipos de dolor repartidos por igual
reporte_sim = Report(metrics=[DataDriftPreset()])  # Nuevo reporte de deriva
reporte_sim.run(reference_data=X_train, current_data=X_derivado)  # Compara train contra la población simulada
reporte_sim.save_html(str(mod.RAIZ / "drift_report_simulado.html"))  # Guarda el segundo reporte
resumen_deriva(reporte_sim)  # Muestra qué variables detecta como derivadas

# %%
auc_derivado = roc_auc_score(y_test, modelo_final.predict_proba(X_derivado)[:, 1])  # AUC del modelo sobre la población simulada
print(f"AUC en test original: {auc_test:.4f} | AUC con datos derivados: {auc_derivado:.4f}")  # Efecto de la deriva en el desempeño

# %% [markdown]
# **Interpretación del monitoreo.**
#
# * **Train frente a test:** Evidently marca deriva en 1 de 11 variables (`Cholesterol`,
#   p = 0,017) y concluye que **no hay deriva del dataset**. Es lo esperable: ambos conjuntos
#   provienen de la misma partición aleatoria, y al aplicar 11 pruebas al 5 % es normal que alguna
#   resulte significativa por azar. Una sola alerta aislada no justifica reentrenar.
# * **Población simulada:** detecta correctamente las tres variables alteradas (`Age`,
#   `Cholesterol` y `ChestPainType`, con p ≈ 0) y ninguna otra. La regla global de Evidently
#   sigue diciendo "sin deriva del dataset" porque exige que derive al menos el 50 % de las
#   columnas; por eso conviene vigilar también las variables más importantes para el modelo.
# * **Efecto en el modelo:** con los datos derivados el AUC cae de 0,930 a 0,882. La deriva
#   de las entradas es una señal temprana de pérdida de desempeño, útil porque en producción
#   la etiqueta real (si el paciente enfermó) tarda en conocerse.
# * **Uso en producción:** la API debería registrar cada petición; periódicamente se compara
#   ese registro con `X_train` usando este mismo reporte, y se reentrena el modelo cuando la
#   deriva afecta a variables relevantes o cae el desempeño.
