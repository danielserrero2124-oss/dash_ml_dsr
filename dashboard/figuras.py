"""Funciones que construyen las figuras de Plotly del dashboard.

Todas comparten el mismo estilo (función ``_base``) y la misma asignación de colores:
el color identifica siempre a la misma entidad (clase o modelo) en todas las figuras.
"""

import numpy as np  # Cálculo numérico
import plotly.graph_objects as go  # Construcción de figuras de Plotly

# ---------------------------------------------------------------------------
# Paleta (modo oscuro, validada para daltonismo) y textos comunes
# ---------------------------------------------------------------------------
SUPERFICIE = "#1a1a19"  # Color de fondo de las tarjetas (se usa para separar marcas superpuestas)
TEXTO = "#ffffff"  # Texto principal
TEXTO_2 = "#c3c2b7"  # Texto secundario (ejes, leyendas)
REJILLA = "#2e2e33"  # Líneas de la cuadrícula
AZUL, NARANJA, AGUA, AMARILLO = "#3987e5", "#d95926", "#199e70", "#c98500"  # Cuatro primeros colores categóricos de la paleta
COLOR_CLASE = {0: AZUL, 1: NARANJA}  # Color fijo por clase: sano = azul, enfermedad = naranja
NOMBRE_CLASE = {0: "Sano (0)", 1: "Enfermedad (1)"}  # Texto de cada clase
COLOR_MODELO = {"LogisticRegression": AZUL, "RandomForest": NARANJA, "KNN": AGUA, "GradientBoosting": AMARILLO}  # Color fijo por modelo
ESCALA_SECUENCIAL = [[0, "#0d366b"], [0.5, "#256abf"], [1, "#9ec5f4"]]  # Un solo tono (azul) de oscuro a claro para magnitudes
ESCALA_DIVERGENTE = [[0, "#3987e5"], [0.5, "#383835"], [1, "#e66767"]]  # Azul ↔ rojo con gris neutro en el centro para correlaciones
ETIQUETAS = {  # Nombre legible de cada variable del dataset
    "Age": "Edad (años)", "RestingBP": "Presión arterial en reposo (mm Hg)", "Cholesterol": "Colesterol (mg/dl)",  # Numéricas (1)
    "MaxHR": "Frecuencia cardíaca máxima", "Oldpeak": "Oldpeak (depresión ST)", "FastingBS": "Glucosa en ayunas > 120",  # Numéricas (2)
    "Sex": "Sexo", "ChestPainType": "Tipo de dolor torácico", "RestingECG": "ECG en reposo",  # Categóricas (1)
    "ExerciseAngina": "Angina por ejercicio", "ST_Slope": "Pendiente del segmento ST",  # Categóricas (2)
}


def pct(valor, decimales=1):  # Da formato de porcentaje con coma decimal
    """0.553 → '55,3 %'."""
    return f"{valor * 100:.{decimales}f}".replace(".", ",") + " %"  # Multiplica por 100, fija decimales y cambia el punto por coma


def _base(fig, alto=360, leyenda=True):  # Aplica el estilo común a cualquier figura
    """Fondo transparente, tipografía, cuadrícula tenue, leyenda horizontal y tooltip oscuro."""
    fig.update_layout(  # Ajustes generales de la figura
        height=alto,  # Alto en píxeles
        paper_bgcolor="rgba(0,0,0,0)",  # Fondo exterior transparente (se ve la tarjeta)
        plot_bgcolor="rgba(0,0,0,0)",  # Fondo del área de datos transparente
        font=dict(family="Segoe UI, system-ui, sans-serif", color=TEXTO_2, size=13),  # Tipografía y color del texto
        margin=dict(l=56, r=18, t=18, b=48),  # Márgenes ajustados
        showlegend=leyenda,  # Muestra u oculta la leyenda
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, bgcolor="rgba(0,0,0,0)", font=dict(color=TEXTO_2)),  # Leyenda horizontal sobre el gráfico
        hoverlabel=dict(bgcolor="#222226", bordercolor=REJILLA, font=dict(color=TEXTO, size=13)),  # Tooltip oscuro
        bargap=0.25,  # Separación entre grupos de barras
    )
    fig.update_xaxes(gridcolor=REJILLA, zerolinecolor=REJILLA, linecolor=REJILLA, title_font=dict(color=TEXTO_2), tickfont=dict(color=TEXTO_2))  # Estilo del eje X
    fig.update_yaxes(gridcolor=REJILLA, zerolinecolor=REJILLA, linecolor=REJILLA, title_font=dict(color=TEXTO_2), tickfont=dict(color=TEXTO_2))  # Estilo del eje Y
    return fig  # Devuelve la figura ya estilizada


# ---------------------------------------------------------------------------
# Figuras del análisis exploratorio (EDA)
# ---------------------------------------------------------------------------
def fig_objetivo(df):  # Distribución de la variable objetivo
    """Barras con el número de pacientes sanos y con enfermedad."""
    conteo = df["HeartDisease"].value_counts().sort_index()  # Pacientes por clase (0 y 1)
    fig = go.Figure(go.Bar(  # Una barra por clase
        x=[NOMBRE_CLASE[c] for c in conteo.index],  # Nombre de cada clase en el eje X
        y=conteo.values,  # Número de pacientes
        marker=dict(color=[COLOR_CLASE[c] for c in conteo.index], cornerradius=4),  # Color fijo por clase y extremos redondeados
        text=[f"{n} · {pct(n / len(df))}" for n in conteo.values],  # Etiqueta directa: conteo y porcentaje
        textposition="outside", textfont=dict(color=TEXTO),  # Etiqueta sobre la barra, en color de texto
        hovertemplate="%{x}<br>%{y} pacientes<extra></extra>",  # Texto del tooltip
        width=0.5,  # Barras delgadas
    ))
    fig.update_yaxes(title="Pacientes", range=[0, conteo.max() * 1.18])  # Eje Y desde cero con espacio para las etiquetas
    return _base(fig, alto=320, leyenda=False)  # Sin leyenda: las categorías ya están en el eje


def fig_numerica(df, variable, tipo, excluir_ceros):  # Distribución de una variable numérica por clase
    """Histograma, violín o caja de ``variable`` separando sanos y enfermos."""
    datos = df[df[variable] > 0] if excluir_ceros and variable in ("Cholesterol", "RestingBP") else df  # Opcionalmente quita los ceros "no medidos"
    fig = go.Figure()  # Figura vacía
    for clase in (0, 1):  # Una serie por clase
        valores = datos.loc[datos["HeartDisease"] == clase, variable]  # Valores de la variable en esa clase
        comun = dict(name=NOMBRE_CLASE[clase], marker_color=COLOR_CLASE[clase])  # Nombre y color comunes a los tres tipos
        if tipo == "histograma":  # Histograma superpuesto
            fig.add_trace(go.Histogram(x=valores, nbinsx=32, opacity=0.72, marker_line=dict(color=SUPERFICIE, width=1),  # Barras semitransparentes con borde separador
                                       hovertemplate="%{x}<br>%{y} pacientes<extra>" + NOMBRE_CLASE[clase] + "</extra>", **comun))  # Tooltip con rango y conteo
        elif tipo == "violin":  # Diagrama de violín con caja interior
            fig.add_trace(go.Violin(y=valores, box_visible=True, meanline_visible=True, opacity=0.85, line_width=2, **comun))  # Densidad + caja + media
        else:  # Diagrama de caja
            fig.add_trace(go.Box(y=valores, boxmean=True, line_width=2, **comun))  # Caja con la media marcada
    if tipo == "histograma":  # Ajustes propios del histograma
        fig.update_layout(barmode="overlay")  # Superpone las dos distribuciones
        fig.update_xaxes(title=ETIQUETAS[variable])  # Eje X: la variable
        fig.update_yaxes(title="Pacientes")  # Eje Y: frecuencia
    else:  # Ajustes de violín y caja
        fig.update_yaxes(title=ETIQUETAS[variable])  # Eje Y: la variable
    return _base(fig, alto=380)  # Aplica el estilo común


def fig_categorica(df, variable, modo):  # Relación de una variable categórica con el objetivo
    """``modo='conteo'``: barras agrupadas por clase. ``modo='tasa'``: proporción de enfermos por categoría."""
    if modo == "tasa":  # Tasa de enfermedad por categoría
        tasa = df.groupby(variable)["HeartDisease"].agg(["mean", "size"]).sort_values("mean")  # Proporción de enfermos y tamaño de cada categoría
        fig = go.Figure(go.Bar(  # Una sola serie: barras horizontales
            x=tasa["mean"], y=tasa.index.astype(str), orientation="h",  # Valor en X, categoría en Y
            marker=dict(color=NARANJA, cornerradius=4),  # Color de la clase "enfermedad"
            text=[pct(v, 0) for v in tasa["mean"]], textposition="outside", textfont=dict(color=TEXTO),  # Etiqueta directa con el porcentaje
            customdata=tasa["size"], hovertemplate="%{y}<br>%{x:.1%} con enfermedad<br>%{customdata} pacientes<extra></extra>",  # Tooltip con tasa y tamaño
            width=0.55,  # Barras delgadas
        ))
        global_ = df["HeartDisease"].mean()  # Tasa global de enfermedad
        fig.add_vline(x=global_, line=dict(color=TEXTO_2, dash="dash", width=1.5),  # Línea de referencia con la tasa global
                      annotation_text=f"Tasa global {pct(global_, 0)}", annotation_font_color=TEXTO_2, annotation_position="top")  # Rótulo de la línea
        fig.update_xaxes(title="Proporción de pacientes con enfermedad", tickformat=".0%", range=[0, 1.08])  # Eje X de 0 a 100 %
        fig.update_yaxes(title=ETIQUETAS[variable])  # Eje Y: la variable
        return _base(fig, alto=380, leyenda=False)  # Una sola serie: sin leyenda
    tabla = df.groupby([variable, "HeartDisease"]).size().unstack(fill_value=0)  # Conteo de pacientes por categoría y clase
    fig = go.Figure()  # Figura vacía
    for clase in (0, 1):  # Una serie por clase
        fig.add_trace(go.Bar(  # Barras agrupadas
            x=tabla.index.astype(str), y=tabla[clase], name=NOMBRE_CLASE[clase],  # Categorías en X, pacientes en Y
            marker=dict(color=COLOR_CLASE[clase], cornerradius=4, line=dict(color=SUPERFICIE, width=2)),  # Color por clase y borde separador
            hovertemplate="%{x}<br>%{y} pacientes<extra>" + NOMBRE_CLASE[clase] + "</extra>",  # Tooltip
        ))
    fig.update_layout(barmode="group")  # Barras lado a lado
    fig.update_xaxes(title=ETIQUETAS[variable])  # Eje X: la variable
    fig.update_yaxes(title="Pacientes")  # Eje Y: frecuencia
    return _base(fig, alto=380)  # Aplica el estilo común


def fig_dispersion(df, var_x, var_y):  # Relación entre dos variables numéricas
    """Diagrama de dispersión coloreado por clase."""
    fig = go.Figure()  # Figura vacía
    for clase in (0, 1):  # Una serie por clase
        d = df[df["HeartDisease"] == clase]  # Pacientes de esa clase
        fig.add_trace(go.Scatter(  # Puntos
            x=d[var_x], y=d[var_y], mode="markers", name=NOMBRE_CLASE[clase],  # Coordenadas y nombre de la serie
            marker=dict(color=COLOR_CLASE[clase], size=9, opacity=0.8, line=dict(color=SUPERFICIE, width=1.5)),  # Marcadores con anillo del color de fondo
            hovertemplate=f"{var_x}: %{{x}}<br>{var_y}: %{{y}}<extra>{NOMBRE_CLASE[clase]}</extra>",  # Tooltip con ambos valores
        ))
    fig.update_xaxes(title=ETIQUETAS[var_x])  # Eje X
    fig.update_yaxes(title=ETIQUETAS[var_y])  # Eje Y
    return _base(fig, alto=400)  # Aplica el estilo común


def fig_correlacion(df):  # Matriz de correlación
    """Mapa de calor de la correlación de Pearson entre variables numéricas y el objetivo."""
    columnas = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak", "HeartDisease"]  # Variables numéricas
    corr = df[columnas].corr().round(2)  # Coeficientes de correlación
    fig = go.Figure(go.Heatmap(  # Mapa de calor
        z=corr.values, x=columnas, y=columnas, zmin=-1, zmax=1, colorscale=ESCALA_DIVERGENTE,  # Escala divergente centrada en cero
        text=corr.values, texttemplate="%{text:.2f}", textfont=dict(color=TEXTO, size=12),  # Coeficiente escrito en cada celda
        xgap=2, ygap=2,  # Separación entre celdas
        colorbar=dict(title="r", tickfont=dict(color=TEXTO_2), outlinewidth=0, thickness=12),  # Barra de color
        hovertemplate="%{y} vs %{x}<br>r = %{z:.2f}<extra></extra>",  # Tooltip
    ))
    fig.update_yaxes(autorange="reversed")  # Diagonal de arriba-izquierda a abajo-derecha
    fig.update_xaxes(tickangle=-35)  # Inclina los nombres del eje X para que no se monten
    return _base(fig, alto=420, leyenda=False)  # Aplica el estilo común


# ---------------------------------------------------------------------------
# Figuras de los modelos
# ---------------------------------------------------------------------------
def fig_roc(resultados, seleccionado, punto):  # Curvas ROC de los cuatro modelos
    """Resalta el modelo seleccionado y marca su punto de operación para el umbral elegido."""
    fig = go.Figure()  # Figura vacía
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=TEXTO_2, dash="dot", width=1), name="Azar (AUC 0,50)", hoverinfo="skip"))  # Diagonal de referencia
    for nombre, r in resultados.items():  # Una curva por modelo
        activo = nombre == seleccionado  # ¿Es el modelo seleccionado?
        fig.add_trace(go.Scatter(  # Curva ROC
            x=r["roc"]["fpr"], y=r["roc"]["tpr"], mode="lines", name=f"{nombre} (AUC {r['auc_test']:.3f})",  # Puntos de la curva y leyenda con el AUC
            line=dict(color=COLOR_MODELO[nombre], width=3.5 if activo else 2), opacity=1 if activo else 0.55,  # El seleccionado va más grueso y opaco
            hovertemplate="FPR %{x:.2f} · TPR %{y:.2f}<extra>" + nombre + "</extra>",  # Tooltip
        ))
    fig.add_trace(go.Scatter(  # Punto de operación del modelo seleccionado
        x=[1 - punto["especificidad"]], y=[punto["sensibilidad"]], mode="markers", name="Umbral elegido",  # Coordenadas según el umbral
        marker=dict(color=TEXTO, size=13, line=dict(color=COLOR_MODELO[seleccionado], width=3)),  # Marcador blanco con borde del color del modelo
        hovertemplate="Sensibilidad %{y:.1%}<br>1 − especificidad %{x:.1%}<extra>Umbral elegido</extra>",  # Tooltip
    ))
    fig.update_xaxes(title="Tasa de falsos positivos (1 − especificidad)", range=[-0.01, 1.01])  # Eje X
    fig.update_yaxes(title="Sensibilidad (tasa de verdaderos positivos)", range=[0, 1.02])  # Eje Y
    return _base(fig, alto=430)  # Aplica el estilo común


def fig_confusion(m):  # Matriz de confusión
    """Mapa de calor 2×2 con los aciertos y errores para el umbral elegido."""
    z = [[m["tn"], m["fp"]], [m["fn"], m["tp"]]]  # Filas: clase real; columnas: clase predicha
    nombres = [["Verdaderos negativos", "Falsos positivos"], ["Falsos negativos", "Verdaderos positivos"]]  # Nombre de cada celda
    texto = [[f"<b>{z[i][j]}</b><br>{nombres[i][j]}" for j in range(2)] for i in range(2)]  # Texto de cada celda: conteo y nombre
    fig = go.Figure(go.Heatmap(  # Mapa de calor
        z=z, x=["Predice: sano", "Predice: enfermedad"], y=["Real: sano", "Real: enfermedad"],  # Ejes
        colorscale=ESCALA_SECUENCIAL, showscale=False, xgap=3, ygap=3,  # Un solo tono; más claro = más pacientes
        text=texto, texttemplate="%{text}", textfont=dict(color=TEXTO, size=15),  # Texto dentro de las celdas
        hovertemplate="%{y} · %{x}<br>%{z} pacientes<extra></extra>",  # Tooltip
    ))
    fig.update_yaxes(autorange="reversed")  # "Real: sano" arriba
    return _base(fig, alto=330, leyenda=False)  # Aplica el estilo común


def fig_sobreajuste(resultados, seleccionado):  # Comparación del AUC en train, validación cruzada y test
    """Gráfico de puntos: la distancia entre train y validación/test es la brecha de sobreajuste."""
    nombres = list(resultados)  # Nombres de los modelos
    fig = go.Figure()  # Figura vacía
    for nombre in nombres:  # Línea que une los tres valores de cada modelo
        r = resultados[nombre]  # Resultados del modelo
        valores = [r["auc_train"], r["auc_cv"], r["auc_test"]]  # Los tres AUC
        fig.add_trace(go.Scatter(x=[min(valores), max(valores)], y=[nombre, nombre], mode="lines",  # Segmento del mínimo al máximo
                                 line=dict(color=TEXTO if nombre == seleccionado else REJILLA, width=3 if nombre == seleccionado else 2),  # Resalta el seleccionado
                                 showlegend=False, hoverinfo="skip"))  # Sin leyenda ni tooltip
    series = [("auc_train", "AUC en train", AMARILLO, "diamond"), ("auc_cv", "AUC validación cruzada", AZUL, "circle"), ("auc_test", "AUC en test", AGUA, "square")]  # Tres medidas con color y símbolo propios
    for clave, etiqueta, color, simbolo in series:  # Una serie de puntos por medida
        fig.add_trace(go.Scatter(  # Puntos
            x=[resultados[n][clave] for n in nombres], y=nombres, mode="markers", name=etiqueta,  # Valor por modelo
            marker=dict(color=color, size=14, symbol=simbolo, line=dict(color=SUPERFICIE, width=2)),  # Color + símbolo (doble codificación)
            error_x=dict(type="data", array=[resultados[n]["auc_cv_desv"] for n in nombres], color=color, thickness=1.5, width=5) if clave == "auc_cv" else None,  # Barra de error solo en validación cruzada
            hovertemplate="%{y}<br>" + etiqueta + ": %{x:.4f}<extra></extra>",  # Tooltip
        ))
    fig.update_xaxes(title="AUC", range=[0.88, 1.005])  # Eje acotado (válido en un gráfico de puntos)
    fig.update_yaxes(autorange="reversed")  # Primer modelo arriba
    return _base(fig, alto=330)  # Aplica el estilo común


def fig_aprendizaje(r, nombre):  # Curva de aprendizaje del modelo seleccionado
    """AUC de entrenamiento y de validación según el número de pacientes usados para entrenar."""
    a = r["aprendizaje"]  # Datos de la curva de aprendizaje
    fig = go.Figure()  # Figura vacía
    for media, desv, etiqueta, color in [("train_media", "train_desv", "Entrenamiento", AMARILLO), ("cv_media", "cv_desv", "Validación cruzada", AZUL)]:  # Dos curvas
        sup, inf = a[media] + a[desv], a[media] - a[desv]  # Límites de la banda de ±1 desviación
        fig.add_trace(go.Scatter(x=np.concatenate([a["n"], a["n"][::-1]]), y=np.concatenate([sup, inf[::-1]]), fill="toself", mode="lines",  # Banda sombreada (solo relleno, sin marcadores)
                                 fillcolor=f"rgba({int(color[1:3], 16)},{int(color[3:5], 16)},{int(color[5:7], 16)},0.16)",  # Mismo color con transparencia
                                 line=dict(width=0), showlegend=False, hoverinfo="skip"))  # Sin borde, leyenda ni tooltip
        fig.add_trace(go.Scatter(x=a["n"], y=a[media], mode="lines+markers", name=etiqueta,  # Curva media
                                 line=dict(color=color, width=2.5), marker=dict(size=8, line=dict(color=SUPERFICIE, width=1.5)),  # Línea y marcadores
                                 hovertemplate="%{x} pacientes<br>AUC %{y:.3f}<extra>" + etiqueta + "</extra>"))  # Tooltip
    fig.update_xaxes(title=f"Pacientes de entrenamiento ({nombre})")  # Eje X
    fig.update_yaxes(title="AUC", range=[0.84, 1.005])  # Eje Y acotado
    fig.update_layout(hovermode="x unified")  # Un solo tooltip con ambas curvas
    return _base(fig, alto=330)  # Aplica el estilo común


def fig_importancia(r, nombre):  # Importancia de las variables
    """Barras horizontales: caída del AUC en test al permutar cada variable."""
    orden = sorted(r["importancia"].items(), key=lambda par: par[1])  # Variables ordenadas de menor a mayor importancia
    fig = go.Figure(go.Bar(  # Una sola serie
        x=[v for _, v in orden], y=[k for k, _ in orden], orientation="h",  # Importancia en X, variable en Y
        marker=dict(color=COLOR_MODELO[nombre], cornerradius=4), width=0.6,  # Color del modelo seleccionado
        hovertemplate="%{y}<br>Caída de AUC: %{x:.4f}<extra></extra>",  # Tooltip
    ))
    fig.update_xaxes(title="Caída media del AUC al permutar la variable")  # Eje X
    return _base(fig, alto=380, leyenda=False)  # Aplica el estilo común


def fig_medidor(probabilidad, umbral):  # Medidor de la predicción individual
    """Indicador tipo velocímetro con la probabilidad estimada y el umbral de decisión."""
    fig = go.Figure(go.Indicator(  # Indicador
        mode="gauge+number", value=probabilidad * 100,  # Aguja y número en porcentaje
        number=dict(suffix=" %", font=dict(color=TEXTO, size=44), valueformat=".1f"),  # Formato del número
        gauge=dict(  # Configuración del medidor
            axis=dict(range=[0, 100], tickcolor=TEXTO_2, tickfont=dict(color=TEXTO_2)),  # Escala de 0 a 100
            bar=dict(color=NARANJA if probabilidad >= umbral else AZUL, thickness=0.7),  # Color de la clase predicha
            bgcolor="#222226", borderwidth=0,  # Fondo del arco
            threshold=dict(line=dict(color=TEXTO, width=3), thickness=0.9, value=umbral * 100),  # Marca del umbral
        ),
    ))
    fig.update_layout(height=220, margin=dict(l=40, r=40, t=22, b=6), paper_bgcolor="rgba(0,0,0,0)", font=dict(color=TEXTO_2))  # Tamaño y fondo
    return fig  # Devuelve el medidor
