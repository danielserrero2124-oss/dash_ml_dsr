# El dashboard: arquitectura y diseño

Este capítulo documenta cómo está construido el dashboard, algo que el propio dashboard no
muestra.

## Cómo ejecutarlo

Desde la carpeta raíz del proyecto:

```bash
.venv\Scripts\python -m dashboard.app
```

Se abre en <http://localhost:8050>. Los modelos ya están entrenados y guardados en
`dashboard/artefactos.joblib`, por lo que arranca al instante. Para volver a entrenarlos:

```bash
.venv\Scripts\python -m dashboard.modelos
```

## Arquitectura

| Archivo | Responsabilidad |
|---|---|
| `dashboard/modelos.py` | Entrena los cuatro modelos sin fuga de datos, calcula métricas, curvas ROC, curvas de aprendizaje e importancias, y guarda todo en `artefactos.joblib` |
| `dashboard/figuras.py` | Construye cada figura de Plotly con un estilo común |
| `dashboard/app.py` | Define la página (secciones y controles) y los *callbacks* que la hacen interactiva |
| `dashboard/assets/estilos.css` | Tema visual oscuro: tarjetas, pestañas, desplegables, deslizador y tablas |
| `dashboard/assets/corazon.svg` | Ilustración de portada, dibujada para el proyecto |
| `src/modelado.py` | Funciones compartidas con los cuadernos: carga, partición, preprocesador y `train_pipeline` |

El entrenamiento está separado de la aplicación. `modelos.py` hace el trabajo costoso una
sola vez y `app.py` solo lee resultados. Así el dashboard responde de inmediato, y los
números que muestra son exactamente los mismos en cada ejecución.

Los modelos del dashboard y los de los cuadernos comparten el mismo código
(`src/modelado.py`), de modo que no puede haber diferencias de preprocesamiento entre lo que
se analiza y lo que se muestra.

## Secciones y controles

| Sección | Contenido | Controles |
|---|---|---|
| **Contexto** | Descripción e importancia, imagen, cuatro indicadores, diccionario de datos, etapas | Ninguno (lectura) |
| **EDA** | Cinco indicadores, variable objetivo, distribución numérica, variable categórica, dispersión, correlaciones, hallazgos | Variable numérica, tipo de gráfico, excluir ceros, variable categórica, medida, eje X, eje Y |
| **ML Models** | Método, cinco indicadores, ROC, matriz de confusión, sobreajuste, curva de aprendizaje, ranking, importancia, predicción | Modelo, umbral de decisión y los 11 datos del paciente |

## Interactividad

Cada control está conectado a una función de Python (*callback*) que recalcula solo lo
necesario:

| *Callback* | Entradas | Qué actualiza |
|---|---|---|
| `actualizar_numerica` | Variable, tipo de gráfico, filtro de ceros | Distribución numérica por clase |
| `actualizar_categorica` | Variable, medida | Tasa de enfermedad o conteo por categoría |
| `actualizar_dispersion` | Eje X, eje Y | Diagrama de dispersión |
| `actualizar_modelos` | Modelo, umbral | Cinco indicadores, cinco figuras, ranking, hiperparámetros y lectura automática |
| `predecir_paciente` | Modelo, umbral y 11 datos del paciente | Medidor de probabilidad y resultado |

El umbral no reentrena nada: las probabilidades de test están guardadas, y al mover el
deslizador solo se vuelve a contar cuántas superan el umbral.

## Decisiones de diseño visual

* **Tema oscuro unificado.** Fondo, tarjetas, pestañas, desplegables, deslizador y tablas
  comparten una misma paleta definida como variables CSS.
* **El color identifica siempre lo mismo.** Azul es "sano" y naranja es "enfermedad" en todas
  las figuras del EDA; cada modelo conserva su color en todas las figuras de ML Models.
* **Paleta apta para daltonismo.** Los colores provienen de una paleta de referencia validada
  para las formas más comunes de daltonismo. Donde hay tres series de puntos (AUC en train,
  validación y test) se usa además un símbolo distinto para cada una.
* **Un solo eje por figura.** No hay gráficos de doble eje. Para ver conteos y tasas de una
  variable categórica se cambia de medida con un botón.
* **Escalas con significado.** Las correlaciones usan una escala divergente (azul, gris,
  rojo) con el gris en cero; la matriz de confusión, un solo tono que se aclara con el conteo.
* **Barras desde cero; puntos con eje acotado.** Las barras siempre parten de cero. El AUC,
  que varía entre 0,90 y 1,00, se muestra con puntos para poder acercar el eje sin exagerar
  las diferencias.
* **Etiquetas directas y *tooltips*.** Las barras llevan su valor escrito y todas las figuras
  muestran el detalle al pasar el cursor.
* **Lectura automática.** Bajo el ranking, un texto generado explica el resultado del modelo
  seleccionado (incluido el caso particular de KNN).

## Pruebas del dashboard

`tests/test_dashboard.py` comprueba que los cuatro modelos están entrenados, que validación
y test coinciden, que cada *callback* devuelve todos sus elementos, que el umbral mueve la
sensibilidad y la especificidad en la dirección esperada y que el formulario de predicción
valida los campos. Se ejecutan en cada *push* junto con las pruebas de la API.

## Publicación

Un dashboard de Dash necesita un servidor de Python, por lo que no puede alojarse en
GitHub Pages. El archivo `render.yaml` permite publicarlo en Render: instala las
dependencias y arranca `gunicorn dashboard.app:server`. La configuración se probó en un
contenedor Linux con Python 3.10; el servicio usa unos 300 MB de memoria, dentro del límite
del plan gratuito.
