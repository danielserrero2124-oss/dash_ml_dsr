# %% [markdown]
# # Sección ML Models del dashboard: resultados e interpretación
#
# Este capítulo recorre la pestaña **ML Models** del dashboard: los cuatro modelos pedidos
# (`LogisticRegression`, `RandomForestClassifier`, `KNeighborsClassifier` y
# `GradientBoostingClassifier`), sus indicadores y cada una de sus figuras, con la
# interpretación correspondiente. Incluye además análisis que el dashboard solo permite ver
# de uno en uno (el efecto del umbral y la comparación de los cuatro modelos lado a lado).
#
# Los modelos se cargan del mismo archivo que usa el dashboard
# (`dashboard/artefactos.joblib`), generado por `dashboard/modelos.py`.

# %%
import sys  # Acceso a la configuración del intérprete de Python
sys.path.insert(0, "..")  # Agrega la carpeta raíz del proyecto para importar los paquetes dashboard y src
import pandas as pd  # Manejo de tablas de datos
from IPython.display import HTML, display  # Permite incrustar HTML (las figuras interactivas) en el cuaderno
from dashboard import figuras as fg  # Mismas funciones de figuras que usa el dashboard
from dashboard import modelos as md  # Artefactos de los modelos y cálculo de métricas
from src import modelado as mod  # Funciones de entrenamiento del proyecto

pd.set_option("display.width", 180)  # Ancho de impresión de las tablas
ART = md.cargar_artefactos()  # Carga los modelos ya entrenados y sus métricas
RES = ART["resultados"]  # Resultados por modelo
y_test = ART["y_test"]  # Etiquetas reales del conjunto de prueba
MEJOR = max(RES, key=lambda n: RES[n]["auc_cv"])  # Modelo con mayor AUC de validación cruzada


def mostrar(fig):  # Muestra una figura del dashboard dentro del informe
    """Aplica el fondo oscuro de las tarjetas del dashboard y la incrusta como HTML interactivo."""
    fig.update_layout(paper_bgcolor=fg.SUPERFICIE, plot_bgcolor=fg.SUPERFICIE)  # Fondo oscuro (en el dashboard lo aporta la tarjeta)
    display(HTML(fig.to_html(include_plotlyjs="cdn", full_html=False)))  # Inserta la figura con la librería Plotly cargada desde internet


print(f"Train: {len(ART['y_train'])} pacientes | Test: {len(y_test)} pacientes | Modelo elegido: {md.ETIQUETAS_MODELOS[MEJOR]}")  # Resumen de la partición

# %% [markdown]
# ## Cómo se evitan la fuga de datos y el sobreajuste
#
# El dashboard lo resume en cinco pasos. El código que lo implementa es este:
#
# ```python
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)  # 1. Partición ANTES de transformar
#
# preprocesador = ColumnTransformer([                                    # 2. Todo lo que "aprende" va dentro del Pipeline
#     ("ceros", Pipeline([("imputar", SimpleImputer(missing_values=0, strategy="median", add_indicator=True)),
#                         ("escalar", MinMaxScaler())]), ["RestingBP", "Cholesterol"]),
#     ("numericas", MinMaxScaler(), ["Age", "FastingBS", "MaxHR", "Oldpeak"]),
#     ("categoricas", OneHotEncoder(handle_unknown="ignore"), ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]),
# ])
# pipe = Pipeline([("prep", preprocesador), ("clf", modelo)])
#
# pliegues = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)   # 3. Validación cruzada estratificada
# grid = GridSearchCV(pipe, malla, cv=pliegues, scoring="roc_auc")        #    El Pipeline se reajusta en cada pliegue
# grid.fit(X_train, y_train)                                              # 4. El test no interviene en la selección
# ```
#
# | Riesgo | Cómo se evita |
# |---|---|
# | El test influye en la imputación o el escalado | La partición se hace antes; la mediana, los mínimos y máximos y las categorías se aprenden solo con train |
# | Los pliegues de validación influyen en el preprocesamiento | El preprocesamiento está dentro del `Pipeline`, que `GridSearchCV` reajusta en cada pliegue |
# | Elegir hiperparámetros o modelo mirando el test | La selección usa el AUC de validación cruzada; el test se consulta una sola vez |
# | Variables que contienen la respuesta | Las 11 variables se registran antes del diagnóstico; no hay variables posteriores |
# | Sobreajuste por exceso de complejidad | Las mallas incluyen parámetros de regularización y se compara el AUC de train, validación y test |

# %% [markdown]
# ## Mallas de búsqueda e hiperparámetros elegidos
#
# El dashboard muestra los hiperparámetros ganadores del modelo seleccionado. Aquí se ven
# los cuatro, junto con la malla completa que se exploró (detalle que no está en el dashboard).

# %%
catalogo = mod.obtener_modelos()  # Catálogo de modelos con sus mallas de búsqueda
filas = []  # Acumulador de filas
for nombre in md.NOMBRES_MODELOS:  # Recorre los cuatro modelos del dashboard
    malla = {k.replace("clf__", ""): v for k, v in catalogo[nombre][1].items()}  # Malla sin el prefijo del Pipeline
    combinaciones = 1  # Contador de combinaciones de la malla
    for valores in malla.values():  # Recorre cada hiperparámetro
        combinaciones *= len(valores)  # Multiplica por el número de valores probados
    filas.append({"modelo": md.ETIQUETAS_MODELOS[nombre], "malla explorada": str(malla), "combinaciones": combinaciones,  # Malla y tamaño
                  "ajustes (× 5 pliegues)": combinaciones * 5, "elegidos": str(RES[nombre]["mejores_parametros"])})  # Ajustes totales e hiperparámetros ganadores
pd.set_option("display.max_colwidth", 120)  # Permite ver el texto completo de las mallas
pd.DataFrame(filas).set_index("modelo")  # Muestra la tabla

# %% [markdown]
# **Interpretación.** En los cuatro casos la validación cruzada elige configuraciones
# **moderadas o regularizadas**, que es justo lo que limita el sobreajuste:
#
# * **RandomForest:** hojas de al menos 5 pacientes (`min_samples_leaf=5`) en lugar de 1, lo
#   que impide que los árboles memoricen casos individuales.
# * **GradientBoosting:** árboles muy poco profundos (`max_depth=2`) con tasa de aprendizaje
#   baja (0,05) y solo 100 iteraciones.
# * **KNN:** 31 vecinos, el valor más alto de la malla; con pocos vecinos el modelo sería muy
#   sensible al ruido.
# * **LogisticRegression:** `C=1`, la regularización L2 estándar.

# %% [markdown]
# ## Indicadores del modelo seleccionado
#
# El dashboard muestra cinco indicadores para el modelo y el umbral elegidos. Esta tabla los
# reúne para los cuatro modelos con el umbral por defecto (0,5).

# %%
filas = []  # Acumulador de filas
for nombre in sorted(RES, key=lambda n: RES[n]["auc_cv"], reverse=True):  # Recorre los modelos de mayor a menor AUC de validación
    r = RES[nombre]  # Resultados del modelo
    m = md.metricas_con_umbral(y_test, r["proba_test"], 0.5)  # Métricas con umbral 0,5
    filas.append({"modelo": md.ETIQUETAS_MODELOS[nombre], "AUC train": r["auc_train"], "AUC CV": r["auc_cv"], "± CV": r["auc_cv_desv"],  # AUC en train y validación
                  "AUC test": r["auc_test"], "exactitud": m["accuracy"], "sensibilidad": m["sensibilidad"],  # AUC en test, exactitud y sensibilidad
                  "especificidad": m["especificidad"], "precisión": m["precision"]})  # Especificidad y precisión
ranking = pd.DataFrame(filas).set_index("modelo").round(3)  # Tabla de ranking
ranking  # Muestra la tabla

# %% [markdown]
# **Interpretación.**
#
# * **AUC de validación cruzada (criterio de selección):** RandomForest 0,934,
#   GradientBoosting 0,930, LogisticRegression 0,929 y KNN 0,924. La diferencia entre el primero
#   y el último (0,010) es menor que la variación entre pliegues (± 0,03), así que los cuatro
#   modelos son **estadísticamente indistinguibles**. Se elige RandomForest por tener el valor
#   más alto y la menor variación entre pliegues (± 0,028).
# * **AUC en test:** entre 0,930 y 0,940. En test el orden cambia (KNN queda primero), lo que
#   confirma que con 184 pacientes las diferencias de centésimas son ruido. Elegir el modelo
#   por su resultado en test habría sido otra forma de fuga de datos.
# * **Exactitud:** entre 88,6 % y 91,3 %, muy por encima del 55,3 % de un clasificador trivial.
# * **Sensibilidad:** 94,1 % en tres modelos (detectan 96 de 102 enfermos) y 92,2 % en la
#   regresión logística.
# * **Especificidad:** entre 84,1 % y 87,8 %; los modelos fallan más con los sanos que con los
#   enfermos, lo cual es el lado preferible del error en un tamizaje.
#
# Que una regresión logística iguale a los ensambles indica que la relación entre las
# variables y el riesgo es esencialmente aditiva: la información está en los datos, no en la
# complejidad del modelo.

# %% [markdown]
# ## Figura 1 · Curvas ROC

# %%
punto = md.metricas_con_umbral(y_test, RES[MEJOR]["proba_test"], 0.5)  # Punto de operación del modelo elegido con umbral 0,5
mostrar(fg.fig_roc(RES, MEJOR, punto))  # Curvas ROC de los cuatro modelos, con el elegido resaltado

# %% [markdown]
# **Interpretación.** La curva ROC muestra, para cada umbral posible, qué proporción de
# enfermos se detecta (eje vertical) frente a qué proporción de sanos se alarma por error (eje
# horizontal). Las cuatro curvas se superponen casi por completo y quedan muy cerca de la
# esquina superior izquierda: con AUC ≈ 0,93, si se toma al azar un paciente enfermo y uno sano,
# el modelo asigna mayor probabilidad al enfermo en el 93 % de los casos.
#
# El punto blanco es el umbral elegido en el dashboard. Con 0,5, RandomForest detecta el
# 94,1 % de los enfermos con un 15,9 % de falsas alarmas. Mover el deslizador desplaza ese
# punto a lo largo de la curva.

# %% [markdown]
# ## Figura 2 · Matriz de confusión y efecto del umbral

# %%
mostrar(fg.fig_confusion(punto))  # Matriz de confusión del modelo elegido con umbral 0,5

# %%
filas = []  # Acumulador de filas
for umbral in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:  # Umbrales que recorre el deslizador del dashboard
    m = md.metricas_con_umbral(y_test, RES[MEJOR]["proba_test"], umbral)  # Métricas del modelo elegido con ese umbral
    filas.append({"umbral": umbral, "verdaderos positivos": m["tp"], "falsos positivos": m["fp"], "falsos negativos": m["fn"],  # Celdas de la matriz
                  "verdaderos negativos": m["tn"], "sensibilidad": round(m["sensibilidad"], 3),  # Sanos bien clasificados y sensibilidad
                  "especificidad": round(m["especificidad"], 3), "precisión": round(m["precision"], 3)})  # Especificidad y precisión
pd.DataFrame(filas).set_index("umbral")  # Tabla del efecto del umbral (en el dashboard se ve un umbral cada vez)

# %% [markdown]
# **Interpretación.** Con el umbral 0,5 y 184 pacientes de prueba, RandomForest acierta 165:
# 96 enfermos detectados y 69 sanos descartados. Se equivoca en 19: **13 falsos positivos**
# (sanos enviados a estudio) y **6 falsos negativos** (enfermos no detectados).
#
# La tabla muestra el compromiso que controla el deslizador del dashboard:
#
# * Bajar el umbral a **0,3** reduce los enfermos no detectados de 6 a 4, pero duplica las
#   falsas alarmas (de 13 a 27).
# * Bajarlo a **0,2** deja solo 3 enfermos sin detectar (sensibilidad 97,1 %), a costa de 35
#   falsas alarmas.
# * Subirlo a **0,7** casi elimina las falsas alarmas (6), pero deja pasar a 29 enfermos.
# * Entre **0,4 y 0,5** la sensibilidad no cambia (96 detectados) y las falsas alarmas bajan
#   de 20 a 13, así que 0,5 es mejor que 0,4 en esta muestra.
#
# No existe un umbral "correcto": depende del costo relativo de cada error. En un tamizaje,
# donde un falso negativo puede costar una vida y un falso positivo cuesta un examen adicional,
# se justifica un umbral por debajo de 0,5.

# %% [markdown]
# ## Figura 3 · ¿Hay sobreajuste? AUC en train, validación y test

# %%
mostrar(fg.fig_sobreajuste(RES, MEJOR))  # Gráfico de puntos con los tres AUC de cada modelo

# %% [markdown]
# **Interpretación.** Cada fila es un modelo. El rombo amarillo es el AUC sobre los mismos
# datos de entrenamiento, el círculo azul el de validación cruzada (con su variación entre
# pliegues) y el cuadrado verde el de test. La distancia entre el rombo y los otros dos es la
# **brecha de sobreajuste**.
#
# | Modelo | AUC train | AUC CV | Brecha | Lectura |
# |---|---|---|---|---|
# | LogisticRegression | 0,935 | 0,929 | 0,007 | Sin sobreajuste: modelo lineal con pocos parámetros |
# | GradientBoosting | 0,955 | 0,930 | 0,025 | Brecha pequeña gracias a los árboles de profundidad 2 |
# | RandomForest | 0,974 | 0,934 | 0,040 | Brecha moderada, habitual en bosques; no afecta al test |
# | KNN | 1,000 | 0,924 | 0,076 | Artefacto: ver explicación |
#
# * En los cuatro modelos el AUC de **test cae dentro del intervalo de validación cruzada**:
#   la validación estimó bien el desempeño real y no hay sobreajuste oculto.
# * El AUC de train de **KNN es 1 por construcción**: la validación eligió ponderar por
#   distancia, y al predecir un paciente de train su vecino más cercano es él mismo (distancia
#   cero, peso infinito). Ese 1,000 no mide nada; lo relevante es que validación (0,924) y test
#   (0,940) coinciden.
# * El AUC de train nunca se usa para decidir: solo se muestra para medir la brecha.

# %% [markdown]
# ## Figura 4 · Curvas de aprendizaje
#
# El dashboard muestra la del modelo seleccionado; aquí están las cuatro.

# %%
for nombre in md.NOMBRES_MODELOS:  # Recorre los cuatro modelos
    mostrar(fg.fig_aprendizaje(RES[nombre], md.ETIQUETAS_MODELOS[nombre]))  # Curva de aprendizaje de cada uno

# %%
filas = []  # Acumulador de filas
for nombre in md.NOMBRES_MODELOS:  # Recorre los cuatro modelos
    a = RES[nombre]["aprendizaje"]  # Datos de la curva de aprendizaje
    filas.append({"modelo": md.ETIQUETAS_MODELOS[nombre],  # Nombre del modelo
                  f"AUC CV con {a['n'][0]} pacientes": round(a["cv_media"][0], 3), f"AUC CV con {a['n'][-1]} pacientes": round(a["cv_media"][-1], 3),  # Validación al inicio y al final
                  "ganancia": round(a["cv_media"][-1] - a["cv_media"][0], 3),  # Mejora al usar todos los datos
                  "brecha final (train − CV)": round(a["train_media"][-1] - a["cv_media"][-1], 3)})  # Distancia entre curvas al final
pd.DataFrame(filas).set_index("modelo")  # Tabla resumen de las curvas

# %% [markdown]
# **Interpretación.** La curva de aprendizaje responde dos preguntas: si el modelo
# sobreajusta y si serviría conseguir más datos.
#
# * **Las curvas convergen:** al aumentar los pacientes, el AUC de entrenamiento baja y el de
#   validación sube. Es el comportamiento de un modelo que generaliza.
# * **LogisticRegression** es el caso más claro: con 587 pacientes las dos curvas casi se
#   tocan (0,936 y 0,929). No queda sobreajuste ni margen de mejora con más datos de este tipo.
# * **RandomForest y GradientBoosting** conservan una brecha de 0,03–0,04, pero su curva de
#   validación ya está casi plana: con solo 88 pacientes alcanzaban 0,91 y con 587 llegan a
#   0,93.
# * **KNN** mantiene el AUC de train en 1 por el artefacto explicado antes; su curva de
#   validación crece poco (de 0,913 a 0,924).
#
# **Conclusión práctica:** las curvas de validación se aplanan cerca de 0,93. Más pacientes
# del mismo tipo aportarían poco; para mejorar habría que incorporar **variables nuevas**
# (antecedentes, medicación, biomarcadores), no más filas.

# %% [markdown]
# ## Figura 5 · Variables más importantes
#
# Importancia por permutación en test: cuánto cae el AUC al desordenar una variable. El
# dashboard la muestra para el modelo seleccionado.

# %%
mostrar(fg.fig_importancia(RES[MEJOR], MEJOR))  # Importancia de las variables en el modelo elegido

# %%
importancias = pd.DataFrame({md.ETIQUETAS_MODELOS[n]: RES[n]["importancia"] for n in md.NOMBRES_MODELOS})  # Importancia de cada variable en los cuatro modelos
importancias["media"] = importancias.mean(axis=1)  # Promedio entre modelos
importancias.sort_values("media", ascending=False).round(4)  # Tabla ordenada por importancia media

# %% [markdown]
# **Interpretación.**
#
# * **`ST_Slope` domina en los cuatro modelos:** desordenarla reduce el AUC entre 0,08 y 0,11,
#   varias veces más que cualquier otra variable. Coincide con el análisis exploratorio.
# * Le siguen `ChestPainType`, `Sex`, `Cholesterol` y `ExerciseAngina`, con caídas de 0,01 a
#   0,03.
# * **`Age`, `MaxHR`, `RestingBP` y `RestingECG` casi no aportan** (caída media inferior a
#   0,003) una vez conocidas las demás variables. Llama la atención en `Age` y `MaxHR`, que por
#   separado sí se asocian con la enfermedad: su información ya queda recogida por `ST_Slope`
#   y el resto de resultados de la prueba de esfuerzo.
# * Que cuatro algoritmos distintos coincidan en el orden de importancia da confianza en que
#   la señal es real y no una particularidad de un modelo.
#
# **Dos precauciones.** La importancia por permutación reparte mal el mérito entre variables
# correlacionadas, y parte de la importancia de `Cholesterol` proviene del indicador de dato
# no medido, que refleja el origen de los datos más que la fisiología.

# %% [markdown]
# ## Ranking de modelos
#
# El dashboard presenta la tabla ordenada por AUC de validación cruzada y, debajo, una lectura
# automática del modelo seleccionado. La tabla completa es la de la sección de indicadores.
#
# **Lectura del ranking.** El orden es RandomForest, GradientBoosting, LogisticRegression y
# KNN, pero el margen total (0,010 de AUC) es inferior a la incertidumbre de la estimación.
# La decisión de quedarse con RandomForest es defendible, y también lo sería elegir la
# regresión logística por ser más simple, más rápida y directamente interpretable. Lo que
# no sería defendible es elegir por el resultado en test.

# %% [markdown]
# ## Predicción para un paciente
#
# El formulario del dashboard envía los 11 datos de un paciente al `Pipeline` del modelo
# seleccionado y muestra la probabilidad en un medidor, con una marca en el umbral.

# %%
pacientes = pd.DataFrame([  # Dos perfiles de ejemplo con las 11 variables de entrada
    {"Age": 54, "Sex": "M", "ChestPainType": "ASY", "RestingBP": 140, "Cholesterol": 239, "FastingBS": 0,  # Perfil de alto riesgo (valores por defecto del dashboard)
     "RestingECG": "Normal", "MaxHR": 120, "ExerciseAngina": "Y", "Oldpeak": 1.5, "ST_Slope": "Flat"},  # Angina de esfuerzo y pendiente ST plana
    {"Age": 35, "Sex": "F", "ChestPainType": "ATA", "RestingBP": 118, "Cholesterol": 190, "FastingBS": 0,  # Perfil de bajo riesgo
     "RestingECG": "Normal", "MaxHR": 175, "ExerciseAngina": "N", "Oldpeak": 0.0, "ST_Slope": "Up"},  # Sin angina y pendiente ST ascendente
], index=["Perfil de alto riesgo", "Perfil de bajo riesgo"])  # Nombres de los perfiles
probabilidades = pd.DataFrame({md.ETIQUETAS_MODELOS[n]: RES[n]["modelo"].predict_proba(pacientes)[:, 1] for n in md.NOMBRES_MODELOS}, index=pacientes.index)  # Probabilidad según cada modelo
probabilidades.round(3)  # Muestra las probabilidades

# %%
mostrar(fg.fig_medidor(float(probabilidades.loc["Perfil de alto riesgo", md.ETIQUETAS_MODELOS[MEJOR]]), 0.5))  # Medidor del dashboard para el perfil de alto riesgo

# %% [markdown]
# **Interpretación.** Los cuatro modelos coinciden en ambos casos: probabilidad muy alta
# para el hombre de 54 años con angina de esfuerzo y pendiente ST plana, y muy baja para la
# mujer de 35 años sin signos en la prueba de esfuerzo. Las probabilidades difieren algo
# entre modelos porque cada uno calibra distinto, pero la decisión es la misma.
#
# El medidor colorea la barra según la clase predicha (naranja = enfermedad, azul = sano) y
# marca el umbral con una línea blanca. Es una herramienta académica: muestra el
# comportamiento del modelo, no sustituye una valoración médica.
