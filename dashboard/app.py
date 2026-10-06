"""Dashboard interactivo del proyecto de predicción de enfermedad cardíaca (Dash + Plotly).

Secciones: Contexto · EDA · ML Models.
Ejecución (desde la raíz del proyecto):  python -m dashboard.app   →   http://localhost:8050
"""

import pandas as pd  # Manejo de tablas de datos
from dash import Dash, Input, Output, dcc, html  # Componentes y decoradores de Dash

from dashboard import figuras as fg  # Funciones que construyen las figuras de Plotly
from dashboard import modelos as md  # Entrenamiento, artefactos y métricas de los modelos
from src import modelado as mod  # Rutas y nombres de columnas del proyecto

# ---------------------------------------------------------------------------
# Datos y modelos (se cargan una sola vez al iniciar la aplicación)
# ---------------------------------------------------------------------------
DF = pd.read_csv(mod.RUTA_DATOS)  # Dataset completo (918 pacientes) para el análisis exploratorio
ART = md.cargar_artefactos()  # Modelos ya entrenados sin fuga de datos, con sus métricas
RES = ART["resultados"]  # Resultados por modelo
Y_TEST = ART["y_test"]  # Etiquetas reales del conjunto de prueba
MEJOR = max(RES, key=lambda n: RES[n]["auc_cv"])  # Modelo con mayor AUC de validación cruzada (elegido sin mirar el test)
NUMERICAS = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]  # Variables numéricas continuas
CATEGORICAS = ["Sex", "ChestPainType", "FastingBS", "RestingECG", "ExerciseAngina", "ST_Slope"]  # Variables categóricas (incluye la binaria FastingBS)
CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]}  # Barra de herramientas de los gráficos simplificada

app = Dash(__name__, title="Heart Disease · ML Dashboard", suppress_callback_exceptions=True)  # Crea la aplicación (carga sola los archivos de assets/)
server = app.server  # Servidor Flask interno, necesario para desplegar con gunicorn


# ---------------------------------------------------------------------------
# Piezas reutilizables de la interfaz
# ---------------------------------------------------------------------------
def numero(valor, decimales=3):  # Da formato a un número con coma decimal
    return f"{valor:.{decimales}f}".replace(".", ",")  # 0.9345 → "0,934"


def tarjeta(*hijos, clase=""):  # Contenedor con fondo, borde y sombra
    return html.Div(list(hijos), className=f"tarjeta {clase}".strip())  # Devuelve un div con la clase CSS "tarjeta"


def kpi(etiqueta, valor, detalle="", id_valor=None):  # Indicador numérico
    cifra = html.Div(valor, className="valor", **({"id": id_valor} if id_valor else {}))  # Cifra grande (con id si se actualiza por callback)
    return tarjeta(html.Div(etiqueta, className="etiqueta"), cifra, html.Div(detalle, className="detalle"), clase="kpi")  # Etiqueta + cifra + detalle


def desplegable(id_, opciones, valor):  # Lista desplegable con estilo propio
    return dcc.Dropdown(id=id_, options=opciones, value=valor, clearable=False, searchable=False)  # Siempre con una opción elegida


def grafico(id_, figura=None):  # Gráfico de Plotly con la configuración común
    return dcc.Graph(id=id_, figure=figura or {}, config=CONFIG)  # Admite figura fija o actualizada por callback


def opciones_variables(variables):  # Opciones de desplegable con el nombre legible de cada variable
    return [{"label": fg.ETIQUETAS[v], "value": v} for v in variables]  # Texto visible → nombre de columna


# ---------------------------------------------------------------------------
# Sección 1: Contexto
# ---------------------------------------------------------------------------
def seccion_contexto():  # Descripción e importancia del problema
    sin_col = DF["Cholesterol"] == 0  # Pacientes sin medición de colesterol
    diccionario = [  # Diccionario de datos: variable, tipo y significado
        ("Age", "Numérica", "Edad del paciente en años"),
        ("Sex", "Categórica", "Sexo: M (hombre) o F (mujer)"),
        ("ChestPainType", "Categórica", "Dolor torácico: TA típica, ATA atípica, NAP no anginoso, ASY asintomático"),
        ("RestingBP", "Numérica", "Presión arterial en reposo (mm Hg)"),
        ("Cholesterol", "Numérica", "Colesterol sérico (mg/dl); 0 = no medido"),
        ("FastingBS", "Binaria", "Glucosa en ayunas mayor a 120 mg/dl (1 = sí)"),
        ("RestingECG", "Categórica", "Electrocardiograma en reposo: Normal, ST o LVH"),
        ("MaxHR", "Numérica", "Frecuencia cardíaca máxima alcanzada"),
        ("ExerciseAngina", "Categórica", "Angina inducida por ejercicio (Y/N)"),
        ("Oldpeak", "Numérica", "Depresión del segmento ST inducida por ejercicio"),
        ("ST_Slope", "Categórica", "Pendiente del segmento ST: Up, Flat o Down"),
        ("HeartDisease", "Objetivo", "1 = enfermedad cardíaca, 0 = sano"),
    ]
    etapas = [  # Etapas del proyecto MLOps
        "Análisis exploratorio y preprocesamiento sin fuga de datos.",
        "Entrenamiento con Pipeline, GridSearchCV y validación cruzada.",
        "API de predicción con FastAPI, empaquetada en Docker.",
        "Despliegue local en Kubernetes (Deployment y Service).",
        "Integración continua con GitHub Actions (estilo y pruebas).",
        "Monitoreo de deriva de datos con Evidently.",
    ]
    return html.Div([  # Contenido de la pestaña
        html.Div([  # Fila de portada: texto + imagen
            tarjeta(  # Texto principal
                html.H2(["Detectar a tiempo la ", html.Span("enfermedad cardíaca", className="resaltado"), " con aprendizaje automático"]),  # Titular
                html.P("Las enfermedades cardiovasculares son la principal causa de muerte en el mundo. Según la Organización "
                       "Mundial de la Salud, en 2019 provocaron 17,9 millones de muertes, el 32 % del total; el 85 % de ellas por "
                       "infarto o accidente cerebrovascular, y más de tres cuartas partes en países de ingresos bajos y medios."),  # Importancia del problema
                html.P("Muchas de esas muertes son prevenibles si el riesgo se identifica pronto. Con datos clínicos de rutina "
                       "(edad, presión arterial, colesterol, electrocardiograma y prueba de esfuerzo) un modelo de clasificación "
                       "puede estimar la probabilidad de enfermedad y ayudar a priorizar a quién estudiar primero."),  # Motivación
                html.P(["Este dashboard resume el proyecto: explora el dataset ", html.B("Heart Failure Prediction"),
                        " (918 pacientes de cinco hospitales) y compara cuatro modelos entrenados con un flujo que evita la ",
                        html.B("fuga de datos"), " y controla el ", html.B("sobreajuste"), "."]),  # Alcance del dashboard
                html.Div([html.Span(t, className="chip") for t in ["Clasificación binaria", "Datos clínicos tabulares", "Pipeline + GridSearchCV", "MLOps local"]]),  # Etiquetas
                clase="heroe",
            ),
            tarjeta(html.Img(src=app.get_asset_url("corazon.svg"), className="imagen-heroe", alt="Corazón con electrocardiograma"), clase="sin-relleno"),  # Imagen
        ], className="rejilla cols-heroe"),
        html.Div([  # Indicadores generales
            kpi("Pacientes", f"{len(DF)}", "registros clínicos"),
            kpi("Variables", "11", "5 numéricas, 5 categóricas y 1 binaria"),
            kpi("Con enfermedad", f"{fg.pct(DF['HeartDisease'].mean())}", f"{int(DF['HeartDisease'].sum())} pacientes"),
            kpi("Mejor AUC en validación", numero(RES[MEJOR]["auc_cv"]), md.ETIQUETAS_MODELOS[MEJOR]),
        ], className="rejilla cols-4"),
        html.Div([  # Diccionario de datos + etapas
            tarjeta(
                html.H3("Diccionario de datos"),
                html.P("Once variables predictoras y la variable objetivo.", className="nota"),
                html.Table([html.Thead(html.Tr([html.Th("Variable"), html.Th("Tipo"), html.Th("Significado")])),  # Encabezado de la tabla
                            html.Tbody([html.Tr([html.Td(html.Code(v)), html.Td(t), html.Td(s)]) for v, t, s in diccionario])], className="tabla"),  # Una fila por variable
            ),
            tarjeta(
                html.H3("¿Por qué importa hacerlo bien?"),
                html.P("En un tamizaje, el error más costoso es el falso negativo: un paciente enfermo que el modelo deja pasar. "
                       "Por eso se reportan la sensibilidad y la especificidad, no solo la exactitud."),
                html.P(f"Calidad de los datos: {int(sin_col.sum())} pacientes tienen colesterol igual a 0, un valor imposible que indica "
                       f"dato no medido. Entre ellos el {fg.pct(DF.loc[sin_col, 'HeartDisease'].mean(), 0)} tiene enfermedad, frente al "
                       f"{fg.pct(DF.loc[~sin_col, 'HeartDisease'].mean(), 0)} del resto, así que el modelo los trata como faltantes informativos."),
                html.H3("Etapas del proyecto", style={"marginTop": "18px"}),
                html.Div([html.Div([html.Div(str(i), className="paso-numero"), html.Div(t, className="paso-texto")], className="paso") for i, t in enumerate(etapas, 1)]),  # Pasos numerados
            ),
        ], className="rejilla cols-2"),
    ])


# ---------------------------------------------------------------------------
# Sección 2: EDA
# ---------------------------------------------------------------------------
def seccion_eda():  # Análisis exploratorio interactivo
    return html.Div([
        html.Div([  # Indicadores del análisis
            kpi("Sanos", f"{int((DF['HeartDisease'] == 0).sum())}", "clase 0"),
            kpi("Con enfermedad", f"{int(DF['HeartDisease'].sum())}", "clase 1"),
            kpi("Edad mediana", f"{DF['Age'].median():.0f} años", f"de {DF['Age'].min()} a {DF['Age'].max()}"),
            kpi("Hombres", f"{fg.pct((DF['Sex'] == 'M').mean(), 0)}", "de la muestra"),
            kpi("Colesterol no medido", f"{fg.pct((DF['Cholesterol'] == 0).mean())}", "ceros imposibles"),
        ], className="rejilla cols-5"),
        html.Div([  # Variable objetivo + variable numérica
            tarjeta(
                html.H3("Variable objetivo"),
                html.P("Clases razonablemente balanceadas: no hace falta remuestreo, pero sí particiones estratificadas.", className="nota"),
                grafico("eda-objetivo", fg.fig_objetivo(DF)),
            ),
            tarjeta(
                html.H3("Distribución de una variable numérica por clase"),
                html.Div([  # Controles en fila
                    html.Div([html.Label("Variable", className="control-etiqueta"), desplegable("eda-num-var", opciones_variables(NUMERICAS), "MaxHR")]),
                    html.Div([html.Label("Tipo de gráfico", className="control-etiqueta"),
                              dcc.RadioItems(id="eda-num-tipo", className="opciones", inline=True, value="histograma",
                                             options=[{"label": "Histograma", "value": "histograma"}, {"label": "Violín", "value": "violin"}, {"label": "Caja", "value": "caja"}])]),
                    html.Div([html.Label("Datos", className="control-etiqueta"),
                              dcc.Checklist(id="eda-num-ceros", className="opciones", inline=True, value=["sin_ceros"],
                                            options=[{"label": "Excluir ceros no medidos", "value": "sin_ceros"}])]),
                ], className="rejilla cols-3", style={"marginBottom": "0"}),
                grafico("eda-numerica"),
            ),
        ], className="rejilla cols-control"),
        html.Div([  # Variable categórica + dispersión
            tarjeta(
                html.H3("Variables categóricas frente al objetivo"),
                html.Div([
                    html.Div([html.Label("Variable", className="control-etiqueta"), desplegable("eda-cat-var", opciones_variables(CATEGORICAS), "ST_Slope")]),
                    html.Div([html.Label("Medida", className="control-etiqueta"),
                              dcc.RadioItems(id="eda-cat-modo", className="opciones", inline=True, value="tasa",
                                             options=[{"label": "Tasa de enfermedad", "value": "tasa"}, {"label": "Conteo por clase", "value": "conteo"}])]),
                ], className="rejilla cols-2", style={"marginBottom": "0"}),
                grafico("eda-categorica"),
            ),
            tarjeta(
                html.H3("Relación entre dos variables numéricas"),
                html.Div([
                    html.Div([html.Label("Eje X", className="control-etiqueta"), desplegable("eda-x", opciones_variables(NUMERICAS), "Age")]),
                    html.Div([html.Label("Eje Y", className="control-etiqueta"), desplegable("eda-y", opciones_variables(NUMERICAS), "MaxHR")]),
                ], className="rejilla cols-2", style={"marginBottom": "0"}),
                grafico("eda-dispersion"),
            ),
        ], className="rejilla cols-2"),
        html.Div([  # Correlación + hallazgos
            tarjeta(
                html.H3("Correlación entre variables numéricas"),
                html.P("Azul: relación inversa. Rojo: relación directa. Gris: sin relación lineal.", className="nota"),
                grafico("eda-correlacion", fg.fig_correlacion(DF)),
            ),
            tarjeta(
                html.H3("Hallazgos principales"),
                html.Ul([
                    html.Li([html.B("Pendiente del segmento ST: "), "es la variable más discriminante. Con pendiente plana o descendente, cerca de 8 de cada 10 pacientes tienen enfermedad; con pendiente ascendente, 2 de cada 10."]),
                    html.Li([html.B("Dolor torácico asintomático (ASY): "), "concentra la mayor tasa de enfermedad, mientras que la angina atípica (ATA) tiene la menor."]),
                    html.Li([html.B("Prueba de esfuerzo: "), "los enfermos alcanzan menor frecuencia cardíaca máxima, tienen mayor Oldpeak y más angina inducida por ejercicio."]),
                    html.Li([html.B("Edad y sexo: "), "la enfermedad es más frecuente en pacientes mayores y en hombres, que son además la mayoría de la muestra."]),
                    html.Li([html.B("Colesterol: "), "su correlación negativa con el objetivo (r = −0,23) es un artefacto de los ceros no medidos; al excluirlos pasa a ser débil y positiva (r = 0,10)."]),
                    html.Li([html.B("Multicolinealidad: "), "ninguna correlación entre predictoras supera 0,4 en valor absoluto, así que se conservan todas."]),
                ]),
            ),
        ], className="rejilla cols-2"),
    ])


# ---------------------------------------------------------------------------
# Sección 3: ML Models
# ---------------------------------------------------------------------------
def campo(etiqueta, componente):  # Campo del formulario de predicción: etiqueta + control
    return html.Div([html.Label(etiqueta, className="control-etiqueta"), componente])  # Agrupa ambos elementos


def seccion_modelos():  # Comparación de modelos y predicción individual
    pasos = [  # Cómo se evita la fuga de datos y el sobreajuste
        ("Partición antes de transformar", "El 20 % de los pacientes (184) se aparta como conjunto de prueba, estratificado, antes de imputar o escalar."),
        ("Preprocesamiento dentro del Pipeline", "La mediana para imputar, el rango del escalado MinMax y las categorías del one-hot se aprenden solo con los datos de entrenamiento."),
        ("GridSearchCV con validación cruzada", "Cada combinación de hiperparámetros se evalúa en 5 pliegues estratificados; en cada pliegue el Pipeline se reajusta desde cero."),
        ("El test se usa una sola vez", "El modelo se elige por su AUC de validación cruzada. El conjunto de prueba solo sirve para reportar el desempeño final."),
        ("Control del sobreajuste", "Las mallas incluyen parámetros que limitan la complejidad (regularización, profundidad, tamaño de hoja, número de vecinos) y se compara el AUC de train, validación y test."),
    ]
    return html.Div([
        html.Div([  # Controles + método
            tarjeta(
                html.H3("Modelo y umbral de decisión"),
                html.Label("Modelo", className="control-etiqueta"),
                desplegable("ml-modelo", [{"label": md.ETIQUETAS_MODELOS[n], "value": n} for n in md.NOMBRES_MODELOS], MEJOR),
                html.Label("Umbral de probabilidad para predecir «enfermedad»", className="control-etiqueta"),
                dcc.Slider(id="ml-umbral", min=0.1, max=0.9, step=0.05, value=0.5, marks={0.1: "0,1", 0.3: "0,3", 0.5: "0,5", 0.7: "0,7", 0.9: "0,9"},
                           tooltip={"placement": "bottom", "always_visible": False}),
                html.P("Bajar el umbral detecta más enfermos (más sensibilidad) a costa de más falsas alarmas.", className="nota", style={"marginTop": "14px"}),
                html.Div(id="ml-parametros"),
            ),
            tarjeta(
                html.H3("Cómo se evitan la fuga de datos y el sobreajuste"),
                html.Div([html.Div([html.Div(str(i), className="paso-numero"),
                                    html.Div([html.B(t + ". "), d], className="paso-texto")], className="paso") for i, (t, d) in enumerate(pasos, 1)]),
            ),
        ], className="rejilla cols-control"),
        html.Div([  # Indicadores del modelo seleccionado
            kpi("AUC validación cruzada", "", "5 pliegues, solo con train", id_valor="kpi-cv"),
            kpi("AUC en test", "", "184 pacientes nunca vistos", id_valor="kpi-test"),
            kpi("Exactitud", "", "con el umbral elegido", id_valor="kpi-acc"),
            kpi("Sensibilidad", "", "enfermos detectados", id_valor="kpi-sens"),
            kpi("Especificidad", "", "sanos bien clasificados", id_valor="kpi-esp"),
        ], className="rejilla cols-5"),
        html.Div([  # ROC + matriz de confusión
            tarjeta(html.H3("Curvas ROC en el conjunto de prueba"),
                    html.P("El modelo seleccionado se resalta; el punto blanco es el umbral elegido.", className="nota"), grafico("ml-roc")),
            tarjeta(html.H3("Matriz de confusión"),
                    html.P("Aciertos y errores del modelo seleccionado con el umbral elegido.", className="nota"), grafico("ml-confusion")),
        ], className="rejilla cols-2"),
        html.Div([  # Sobreajuste + curva de aprendizaje
            tarjeta(html.H3("¿Hay sobreajuste? AUC en train, validación y test"),
                    html.P("Cuanto más lejos queda el AUC de train del de validación y test, mayor es la brecha de sobreajuste.", className="nota"), grafico("ml-sobreajuste")),
            tarjeta(html.H3("Curva de aprendizaje"),
                    html.P("Si las curvas se acercan al aumentar los pacientes, el modelo generaliza.", className="nota"), grafico("ml-aprendizaje")),
        ], className="rejilla cols-2"),
        html.Div([  # Ranking + importancia
            tarjeta(html.H3("Ranking de modelos"),
                    html.P("Ordenado por AUC de validación cruzada, el criterio de selección.", className="nota"), html.Div(id="ml-ranking"), html.Div(id="ml-lectura")),
            tarjeta(html.H3("Variables más importantes"),
                    html.P("Importancia por permutación en test para el modelo seleccionado.", className="nota"), grafico("ml-importancia")),
        ], className="rejilla cols-2"),
        tarjeta(  # Predicción individual
            html.H3("Predicción para un paciente"),
            html.P("Cambia los datos clínicos y observa la probabilidad estimada por el modelo seleccionado. Herramienta académica: no sustituye un diagnóstico médico.", className="nota"),
            html.Div([
                html.Div([  # Formulario
                    html.Div([
                        campo("Edad", dcc.Input(id="p-Age", type="number", value=54, min=18, max=100, className="entrada")),
                        campo("Sexo", desplegable("p-Sex", [{"label": "Hombre", "value": "M"}, {"label": "Mujer", "value": "F"}], "M")),
                        campo("Dolor torácico", desplegable("p-ChestPainType", [{"label": "Asintomático (ASY)", "value": "ASY"}, {"label": "Angina atípica (ATA)", "value": "ATA"},
                                                                                {"label": "No anginoso (NAP)", "value": "NAP"}, {"label": "Angina típica (TA)", "value": "TA"}], "ASY")),
                        campo("Presión en reposo", dcc.Input(id="p-RestingBP", type="number", value=140, min=0, max=250, className="entrada")),
                        campo("Colesterol (0 = no medido)", dcc.Input(id="p-Cholesterol", type="number", value=239, min=0, max=700, className="entrada")),
                        campo("Glucosa en ayunas > 120", desplegable("p-FastingBS", [{"label": "No", "value": 0}, {"label": "Sí", "value": 1}], 0)),
                        campo("ECG en reposo", desplegable("p-RestingECG", [{"label": "Normal", "value": "Normal"}, {"label": "Anomalía ST", "value": "ST"}, {"label": "Hipertrofia (LVH)", "value": "LVH"}], "Normal")),
                        campo("Frecuencia máxima", dcc.Input(id="p-MaxHR", type="number", value=120, min=50, max=220, className="entrada")),
                        campo("Angina por ejercicio", desplegable("p-ExerciseAngina", [{"label": "No", "value": "N"}, {"label": "Sí", "value": "Y"}], "Y")),
                        campo("Oldpeak", dcc.Input(id="p-Oldpeak", type="number", value=1.5, min=-3, max=7, step=0.1, className="entrada")),
                        campo("Pendiente ST", desplegable("p-ST_Slope", [{"label": "Ascendente (Up)", "value": "Up"}, {"label": "Plana (Flat)", "value": "Flat"}, {"label": "Descendente (Down)", "value": "Down"}], "Flat")),
                    ], className="rejilla cols-4", style={"marginBottom": "0"}),
                ]),
                html.Div([grafico("p-medidor"), html.Div(id="p-texto", className="diagnostico")], className="resultado"),  # Medidor y resultado
            ], className="rejilla", style={"gridTemplateColumns": "minmax(0, 2.2fr) minmax(0, 1fr)", "marginBottom": "0", "alignItems": "center"}),
        ),
    ])


# ---------------------------------------------------------------------------
# Estructura general de la página
# ---------------------------------------------------------------------------
app.layout = html.Div([
    html.Div([  # Encabezado
        html.Div([html.Div("♥", className="marca-icono"),
                  html.Div([html.H1("Heart Disease · ML Dashboard"), html.P("Predicción de enfermedad cardíaca con aprendizaje automático")])], className="marca"),
        html.Div([html.B("Daniel Serrano Romero"), html.Br(), "Maestría en Matemáticas · Machine Learning"], className="autor"),
    ], className="encabezado"),
    dcc.Tabs(id="pestanas", value="contexto", parent_className="pestanas-contenedor", className="pestanas", children=[  # Navegación por pestañas
        dcc.Tab(label="Contexto", value="contexto", className="tab", selected_className="tab--selected", children=seccion_contexto()),
        dcc.Tab(label="EDA", value="eda", className="tab", selected_className="tab--selected", children=seccion_eda()),
        dcc.Tab(label="ML Models", value="modelos", className="tab", selected_className="tab--selected", children=seccion_modelos()),
    ]),
    html.Div("Dataset: Heart Failure Prediction (Kaggle, fedesoriano) · Cifras de contexto: Organización Mundial de la Salud", className="pie"),  # Pie de página
], className="contenedor")


# ---------------------------------------------------------------------------
# Interactividad (callbacks)
# ---------------------------------------------------------------------------
@app.callback(Output("eda-numerica", "figure"), Input("eda-num-var", "value"), Input("eda-num-tipo", "value"), Input("eda-num-ceros", "value"))  # Se ejecuta al cambiar cualquiera de los tres controles
def actualizar_numerica(variable, tipo, ceros):  # Redibuja la distribución numérica
    return fg.fig_numerica(DF, variable, tipo, "sin_ceros" in (ceros or []))  # Figura según la variable, el tipo y el filtro de ceros


@app.callback(Output("eda-categorica", "figure"), Input("eda-cat-var", "value"), Input("eda-cat-modo", "value"))  # Variable categórica y medida
def actualizar_categorica(variable, modo):  # Redibuja el gráfico categórico
    return fg.fig_categorica(DF, variable, modo)  # Tasa de enfermedad o conteo por clase


@app.callback(Output("eda-dispersion", "figure"), Input("eda-x", "value"), Input("eda-y", "value"))  # Ejes del diagrama de dispersión
def actualizar_dispersion(var_x, var_y):  # Redibuja la dispersión
    return fg.fig_dispersion(DF, var_x, var_y)  # Puntos coloreados por clase


@app.callback(  # Todo lo que depende del modelo y del umbral en la sección de modelos
    Output("kpi-cv", "children"), Output("kpi-test", "children"), Output("kpi-acc", "children"), Output("kpi-sens", "children"), Output("kpi-esp", "children"),
    Output("ml-roc", "figure"), Output("ml-confusion", "figure"), Output("ml-sobreajuste", "figure"), Output("ml-aprendizaje", "figure"),
    Output("ml-importancia", "figure"), Output("ml-ranking", "children"), Output("ml-parametros", "children"), Output("ml-lectura", "children"),
    Input("ml-modelo", "value"), Input("ml-umbral", "value"),
)
def actualizar_modelos(nombre, umbral):  # Recalcula indicadores, figuras y tablas
    r = RES[nombre]  # Resultados del modelo seleccionado
    m = md.metricas_con_umbral(Y_TEST, r["proba_test"], umbral)  # Matriz de confusión y métricas con el umbral elegido
    orden = sorted(RES, key=lambda n: RES[n]["auc_cv"], reverse=True)  # Modelos de mayor a menor AUC de validación
    filas = []  # Filas de la tabla de ranking
    for posicion, n in enumerate(orden, 1):  # Recorre el ranking
        mm = md.metricas_con_umbral(Y_TEST, RES[n]["proba_test"], umbral)  # Métricas de ese modelo con el mismo umbral
        filas.append(html.Tr([html.Td(str(posicion)), html.Td(md.ETIQUETAS_MODELOS[n]),  # Posición y nombre
                              html.Td(numero(RES[n]["auc_train"]), className="num"), html.Td(numero(RES[n]["auc_cv"]), className="num"),  # AUC train y validación
                              html.Td(numero(RES[n]["auc_test"]), className="num"), html.Td(f"{fg.pct(mm['accuracy'])}", className="num"),  # AUC test y exactitud
                              html.Td(f"{fg.pct(mm['sensibilidad'])}", className="num")], className="activa" if n == nombre else ""))  # Sensibilidad; resalta el seleccionado
    ranking = html.Table([html.Thead(html.Tr([html.Th("#"), html.Th("Modelo"), html.Th("AUC train", className="num"), html.Th("AUC CV", className="num"),
                                              html.Th("AUC test", className="num"), html.Th("Exactitud", className="num"), html.Th("Sensib.", className="num")])),
                          html.Tbody(filas)], className="tabla")  # Tabla completa
    parametros = html.Div([html.Label("Hiperparámetros elegidos por validación cruzada", className="control-etiqueta"),
                           html.Div([html.Span(f"{k} = {v}", className="chip") for k, v in r["mejores_parametros"].items()])])  # Etiquetas con los hiperparámetros
    brecha = r["auc_train"] - r["auc_cv"]  # Diferencia entre el AUC de train y el de validación
    if nombre == "KNN":  # Caso particular de KNN con pesos por distancia
        nota = "En KNN con pesos por distancia cada paciente de train es su propio vecino más cercano, por eso el AUC de train es 1 por construcción; lo relevante es que validación y test coinciden."
    elif brecha > 0.03:  # Brecha apreciable
        nota = "La brecha entre train y validación es apreciable, pero el AUC de test coincide con el de validación: el modelo generaliza como se estimó."
    else:  # Brecha pequeña
        nota = "La brecha entre train y validación es pequeña y el AUC de test coincide con el de validación: no hay señales de sobreajuste."
    lectura = html.P([html.B(f"{md.ETIQUETAS_MODELOS[nombre]}: "), f"AUC train {numero(r['auc_train'])}, validación {numero(r['auc_cv'])} "
                      f"(± {numero(r['auc_cv_desv'])}), test {numero(r['auc_test'])}. {nota}"], style={"marginTop": "14px"})  # Lectura automática del resultado
    return (numero(r["auc_cv"]), numero(r["auc_test"]), f"{fg.pct(m['accuracy'])}", f"{fg.pct(m['sensibilidad'])}", f"{fg.pct(m['especificidad'])}",  # Indicadores
            fg.fig_roc(RES, nombre, m), fg.fig_confusion(m), fg.fig_sobreajuste(RES, nombre), fg.fig_aprendizaje(r, md.ETIQUETAS_MODELOS[nombre]),  # Figuras
            fg.fig_importancia(r, nombre), ranking, parametros, lectura)  # Importancia, tabla, hiperparámetros y lectura


@app.callback(  # Predicción individual: depende del modelo, del umbral y de los 11 datos del paciente
    Output("p-medidor", "figure"), Output("p-texto", "children"),
    Input("ml-modelo", "value"), Input("ml-umbral", "value"),
    *[Input(f"p-{columna}", "value") for columna in mod.COLUMNAS_ENTRADA],
)
def predecir_paciente(nombre, umbral, *valores):  # Calcula la probabilidad para el paciente del formulario
    if any(v is None for v in valores):  # Si algún campo numérico está vacío o fuera de rango...
        return fg.fig_medidor(0.0, umbral), "Completa todos los campos con valores válidos."  # ...se pide completarlo
    paciente = pd.DataFrame([dict(zip(mod.COLUMNAS_ENTRADA, valores))])  # DataFrame de una fila con los nombres de columna del entrenamiento
    probabilidad = float(RES[nombre]["modelo"].predict_proba(paciente)[0, 1])  # El Pipeline preprocesa y devuelve P(enfermedad)
    etiqueta = "Riesgo alto: predice enfermedad" if probabilidad >= umbral else "Riesgo bajo: predice sano"  # Decisión según el umbral
    return fg.fig_medidor(probabilidad, umbral), etiqueta  # Medidor actualizado y texto del resultado


if __name__ == "__main__":  # Punto de entrada al ejecutar  python -m dashboard.app
    app.run(debug=False, host="127.0.0.1", port=8050)  # Inicia el servidor local en el puerto 8050
