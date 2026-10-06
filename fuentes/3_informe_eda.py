# %% [markdown]
# # Sección EDA del dashboard: figuras e interpretación
#
# Este capítulo reproduce, una por una, las figuras de la pestaña **EDA** del dashboard y
# explica qué muestra cada una. Las figuras se generan con las mismas funciones que usa el
# dashboard (`dashboard/figuras.py`), así que son interactivas: al pasar el cursor aparecen
# los valores.
#
# En el dashboard varias figuras dependen de un desplegable. Aquí se muestran las opciones más
# informativas y se resumen las demás en tablas.

# %%
import sys  # Acceso a la configuración del intérprete de Python
sys.path.insert(0, "..")  # Agrega la carpeta raíz del proyecto para importar los paquetes dashboard y src
import pandas as pd  # Manejo de tablas de datos
from IPython.display import HTML, display  # Permite incrustar HTML (las figuras interactivas) en el cuaderno
from dashboard import figuras as fg  # Mismas funciones de figuras que usa el dashboard
from src import modelado as mod  # Rutas y nombres de columnas del proyecto

pd.set_option("display.width", 160)  # Ancho de impresión de las tablas
df = pd.read_csv(mod.RUTA_DATOS)  # Carga el dataset completo (918 pacientes)


def mostrar(fig):  # Muestra una figura del dashboard dentro del informe
    """Aplica el fondo oscuro de las tarjetas del dashboard y la incrusta como HTML interactivo."""
    fig.update_layout(paper_bgcolor=fg.SUPERFICIE, plot_bgcolor=fg.SUPERFICIE)  # Fondo oscuro (en el dashboard lo aporta la tarjeta)
    display(HTML(fig.to_html(include_plotlyjs="cdn", full_html=False)))  # Inserta la figura con la librería Plotly cargada desde internet


print(f"{df.shape[0]} pacientes × {df.shape[1]} columnas")  # Dimensión del dataset

# %% [markdown]
# ## Indicadores de cabecera
#
# La pestaña abre con cinco indicadores que resumen la muestra.

# %%
indicadores = pd.Series({  # Mismos indicadores que muestra el dashboard
    "Sanos (clase 0)": int((df["HeartDisease"] == 0).sum()),  # Pacientes sin enfermedad
    "Con enfermedad (clase 1)": int(df["HeartDisease"].sum()),  # Pacientes con enfermedad
    "Edad mediana (años)": df["Age"].median(),  # Edad central de la muestra
    "Edad mínima – máxima": f"{df['Age'].min()} – {df['Age'].max()}",  # Rango de edades
    "Hombres (%)": round(100 * (df["Sex"] == "M").mean(), 1),  # Proporción de hombres
    "Colesterol no medido (%)": round(100 * (df["Cholesterol"] == 0).mean(), 1),  # Proporción de ceros imposibles
}, name="valor")  # Nombre de la columna
indicadores.to_frame()  # Muestra la tabla

# %% [markdown]
# **Interpretación.** La muestra es de adultos de mediana edad (mediana de 54 años) y está
# dominada por hombres (79 %), lo que limita la validez de cualquier conclusión para mujeres:
# solo hay 193. El 18,7 % de los pacientes no tiene medición de colesterol, un problema de
# calidad de datos que condiciona el preprocesamiento.

# %% [markdown]
# ## Figura 1 · Variable objetivo

# %%
mostrar(fg.fig_objetivo(df))  # Barras con el número de pacientes por clase

# %% [markdown]
# **Interpretación.** Hay 508 pacientes con enfermedad (55,3 %) y 410 sanos (44,7 %). Las
# clases están casi balanceadas, así que:
#
# * no hace falta sobremuestreo ni ponderación de clases;
# * un clasificador trivial que siempre predijera "enfermedad" acertaría el 55,3 %, y ese es el
#   mínimo que cualquier modelo debe superar;
# * las particiones se hacen estratificadas para conservar esta proporción en train y test.
#
# La prevalencia es mucho mayor que en la población general porque los datos provienen de
# pacientes remitidos a estudio cardiológico. Las probabilidades del modelo no deben
# trasladarse sin recalibrar a un contexto de tamizaje poblacional.

# %% [markdown]
# ## Figura 2 · Distribución de una variable numérica por clase
#
# En el dashboard se elige la variable, el tipo de gráfico (histograma, violín o caja) y si
# se excluyen los ceros no medidos. Se muestran aquí las dos variables más discriminantes.

# %%
mostrar(fg.fig_numerica(df, "MaxHR", "histograma", True))  # Frecuencia cardíaca máxima: histograma superpuesto por clase

# %%
mostrar(fg.fig_numerica(df, "Oldpeak", "violin", True))  # Oldpeak: diagrama de violín por clase

# %%
medianas = df.groupby("HeartDisease")[["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]].median().T  # Mediana de cada variable por clase
medianas.columns = ["Sano (0)", "Enfermedad (1)"]  # Nombres de las columnas
medianas["Diferencia"] = medianas["Enfermedad (1)"] - medianas["Sano (0)"]  # Diferencia entre clases
medianas.loc["Cholesterol (sin ceros)"] = list(df[df["Cholesterol"] > 0].groupby("HeartDisease")["Cholesterol"].median()) + [None]  # Colesterol excluyendo los no medidos
medianas.loc["Cholesterol (sin ceros)", "Diferencia"] = medianas.loc["Cholesterol (sin ceros)", "Enfermedad (1)"] - medianas.loc["Cholesterol (sin ceros)", "Sano (0)"]  # Su diferencia
medianas  # Muestra la tabla de medianas

# %% [markdown]
# **Interpretación.**
#
# * **`MaxHR` (frecuencia cardíaca máxima):** es la numérica que mejor separa las clases. Los
#   enfermos alcanzan una mediana de 126 latidos por minuto frente a 150 en los sanos: un
#   corazón enfermo tolera peor el esfuerzo. Las distribuciones se solapan entre 110 y 160, de
#   modo que la variable ayuda pero no decide sola.
# * **`Oldpeak`:** la mitad de los sanos tiene exactamente 0 (sin depresión del segmento ST),
#   mientras que la mediana de los enfermos es 1,2. El violín muestra la masa de sanos
#   concentrada en cero y una cola larga hacia arriba en los enfermos.
# * **`Age`:** los enfermos son mayores (57 frente a 51 años de mediana).
# * **`RestingBP`:** casi no distingue (132 frente a 130 mm Hg).
# * **`Cholesterol`:** con todos los datos parece más bajo en los enfermos (217 frente a 227),
#   pero es un efecto de los ceros. Al excluirlos, los enfermos tienen colesterol algo mayor
#   (246 frente a 231,5). Por eso el dashboard excluye los ceros por defecto en este gráfico.

# %% [markdown]
# ## Figura 3 · Variables categóricas frente al objetivo
#
# El dashboard ofrece dos medidas: la **tasa de enfermedad** por categoría (con la tasa global
# como referencia) y el **conteo** por clase.

# %%
mostrar(fg.fig_categorica(df, "ST_Slope", "tasa"))  # Pendiente del segmento ST: proporción de enfermos por categoría

# %%
mostrar(fg.fig_categorica(df, "ChestPainType", "conteo"))  # Tipo de dolor torácico: pacientes por categoría y clase

# %%
filas = []  # Acumulador de filas de la tabla resumen
for variable in ["Sex", "ChestPainType", "FastingBS", "RestingECG", "ExerciseAngina", "ST_Slope"]:  # Recorre las seis variables del desplegable
    resumen = df.groupby(variable)["HeartDisease"].agg(["size", "mean"])  # Tamaño y tasa de enfermedad de cada categoría
    for categoria, fila in resumen.iterrows():  # Recorre las categorías
        filas.append({"variable": variable, "categoría": categoria, "pacientes": int(fila["size"]), "tasa de enfermedad (%)": round(100 * fila["mean"], 1)})  # Agrega la fila
pd.DataFrame(filas).set_index(["variable", "categoría"])  # Tabla con todas las categorías

# %% [markdown]
# **Interpretación.**
#
# * **`ST_Slope` (pendiente del segmento ST):** es la variable más informativa del dataset. Con
#   pendiente plana (`Flat`) el 82,8 % tiene enfermedad y con pendiente descendente (`Down`)
#   el 77,8 %; con pendiente ascendente (`Up`) solo el 19,7 %. Una pendiente plana o
#   descendente durante el esfuerzo es un signo clásico de isquemia.
# * **`ChestPainType`:** el resultado es contraintuitivo. Los pacientes **asintomáticos**
#   (`ASY`) son los de mayor riesgo (79,0 %) y además el grupo más numeroso (496), mientras que
#   la angina atípica (`ATA`) tiene la tasa más baja (13,9 %). En esta muestra, llegar a
#   estudio sin dolor típico se asocia con enfermedad ya establecida.
# * **`ExerciseAngina`:** con angina inducida por ejercicio la tasa sube a 85,2 %, frente a
#   35,1 % sin ella.
# * **`FastingBS`:** con glucosa en ayunas elevada la tasa es 79,4 % frente a 48,0 %.
# * **`Sex`:** 63,2 % en hombres frente a 25,9 % en mujeres.
# * **`RestingECG`:** es la categórica menos discriminante (entre 51,6 % y 65,7 %).
#
# La línea punteada de la tasa global (55 %) permite ver de inmediato qué categorías están
# por encima o por debajo del riesgo promedio.

# %% [markdown]
# ## Figura 4 · Relación entre dos variables numéricas
#
# Los dos ejes se eligen con desplegables. La combinación por defecto es edad frente a
# frecuencia cardíaca máxima.

# %%
mostrar(fg.fig_dispersion(df, "Age", "MaxHR"))  # Diagrama de dispersión coloreado por clase

# %% [markdown]
# **Interpretación.** La frecuencia máxima disminuye con la edad (r = −0,38), como es
# fisiológicamente esperable. Para una misma edad, los pacientes con enfermedad (naranja)
# tienden a quedar por debajo de los sanos (azul): la zona de arriba a la izquierda (jóvenes
# con frecuencia alta) es casi toda azul y la de abajo a la derecha, casi toda naranja. Aun
# así, no existe una frontera nítida, lo que anticipa que ningún modelo alcanzará una
# separación perfecta con estas variables.

# %% [markdown]
# ## Figura 5 · Correlación entre variables numéricas

# %%
mostrar(fg.fig_correlacion(df))  # Mapa de calor de la correlación de Pearson

# %% [markdown]
# **Interpretación.**
#
# * **Con el objetivo:** las correlaciones más fuertes son `Oldpeak` (r = 0,40) y `MaxHR`
#   (r = −0,40), seguidas de `Age` (0,28) y `FastingBS` (0,27).
# * **`Cholesterol`** aparece con correlación negativa (−0,23) por los ceros no medidos; al
#   excluirlos pasa a ser débil y positiva (0,10).
# * **Entre predictoras:** la mayor es `Age`–`MaxHR` (−0,38). Ninguna supera 0,4 en valor
#   absoluto, así que no hay multicolinealidad que justifique eliminar variables.
#
# La matriz solo incluye variables numéricas, y la correlación de Pearson solo mide relaciones
# lineales. Las variables más predictivas del problema son categóricas (`ST_Slope`,
# `ChestPainType`) y no aparecen aquí; por eso el dashboard complementa esta figura con las
# tasas por categoría.

# %% [markdown]
# ## Detalle que no aparece en el dashboard: los ceros de colesterol
#
# El dashboard menciona el problema, pero no lo cuantifica por completo.

# %%
sin_medir = df["Cholesterol"] == 0  # Máscara de pacientes sin medición de colesterol
detalle = pd.DataFrame({  # Compara ambos grupos
    "pacientes": [int(sin_medir.sum()), int((~sin_medir).sum())],  # Tamaño de cada grupo
    "tasa de enfermedad (%)": [round(100 * df.loc[sin_medir, "HeartDisease"].mean(), 1), round(100 * df.loc[~sin_medir, "HeartDisease"].mean(), 1)],  # Tasa de enfermedad
    "hombres (%)": [round(100 * (df.loc[sin_medir, "Sex"] == "M").mean(), 1), round(100 * (df.loc[~sin_medir, "Sex"] == "M").mean(), 1)],  # Proporción de hombres
    "edad mediana": [df.loc[sin_medir, "Age"].median(), df.loc[~sin_medir, "Age"].median()],  # Edad central
}, index=["Cholesterol = 0 (no medido)", "Cholesterol > 0"])  # Nombres de las filas
detalle  # Muestra la tabla

# %% [markdown]
# **Interpretación.** Los pacientes sin medición de colesterol no son una muestra al azar:
# tienen una tasa de enfermedad muy superior (88,4 % frente a 47,7 %), son casi todos hombres
# (93,6 % frente a 75,6 %) y algo mayores (57,5 frente a 54 años de mediana). El dataset combina
# registros de cinco hospitales y es probable que alguno no midiera esa variable y atendiera
# casos más graves. Tres consecuencias para el modelado:
#
# 1. Eliminar esas filas descartaría casi una quinta parte de los datos y sesgaría la muestra.
# 2. Dejar el cero como valor real engañaría a los modelos que usan distancias o escalas.
# 3. La **ausencia del dato es informativa**: por eso el `Pipeline` imputa la mediana (aprendida
#    en train) y agrega una columna indicadora de "dato faltante".
#
# Esa señal refleja cómo se recogieron los datos, no la fisiología. Un modelo desplegado en
# un hospital que siempre mide el colesterol no dispondrá de ella, y conviene vigilarlo con el
# monitoreo de deriva.

# %% [markdown]
# ## Síntesis del análisis exploratorio
#
# | Hallazgo | Evidencia | Consecuencia para el modelado |
# |---|---|---|
# | Clases casi balanceadas | 55,3 % / 44,7 % | Sin remuestreo; particiones estratificadas |
# | La prueba de esfuerzo concentra la señal | `ST_Slope`, `ExerciseAngina`, `Oldpeak`, `MaxHR` | Se esperan AUC altos incluso con modelos simples |
# | Variables categóricas muy informativas | Tasas entre 14 % y 85 % según la categoría | Codificación *one-hot* dentro del `Pipeline` |
# | Ceros imposibles e informativos | 172 en `Cholesterol`, 1 en `RestingBP` | Imputación con mediana de train + indicador |
# | Sin multicolinealidad | Correlación máxima entre predictoras: 0,38 | Se conservan las 11 variables |
# | Muestra mayoritariamente masculina | 79 % hombres | Cautela al generalizar a mujeres |
