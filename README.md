# Predicción de falla cardíaca — Dashboard interactivo y MLOps local

Proyecto integrador de Aprendizaje Automático. Cubre el ciclo de vida completo de un modelo
de clasificación binaria que predice si un paciente tiene enfermedad cardíaca
(`HeartDisease` = 1) a partir de 11 variables clínicas del dataset *Heart Failure Prediction*
(Kaggle, 918 pacientes).

**Autor:** Daniel Serrano Romero · Maestría en Matemáticas · Machine Learning

## Dashboard interactivo (entregable)

Dashboard en Dash + Plotly con tres secciones:

* **Contexto:** descripción e importancia del problema, indicadores del dataset, diccionario de
  datos y etapas del proyecto.
* **EDA:** distribución del objetivo, variables numéricas (histograma, violín o caja), variables
  categóricas (tasa de enfermedad o conteo), dispersión entre dos variables, correlaciones y hallazgos.
* **ML Models:** `LogisticRegression`, `RandomForestClassifier`, `KNeighborsClassifier` y
  `GradientBoostingClassifier`, entrenados sin fuga de datos; curvas ROC, matriz de confusión
  con umbral ajustable, comparación del AUC en train, validación y test, curva de aprendizaje,
  importancia de variables, ranking y predicción para un paciente.

```bash
.venv\Scripts\python -m dashboard.app
```

Se abre en <http://localhost:8050>. Los modelos ya están entrenados en
`dashboard/artefactos.joblib`; para regenerarlos:

```bash
.venv\Scripts\python -m dashboard.modelos
```

## Resultados

| Elemento | Resultado |
|---|---|
| Modelo elegido | `RandomForestClassifier` dentro de un `Pipeline` (imputación + MinMax + one-hot) |
| AUC validación cruzada (5 pliegues) | 0,9345 |
| AUC en test | 0,930 |
| Sensibilidad / especificidad en test | 94,1 % / 84,1 % |
| Pruebas automáticas | 14 de 14 (API y dashboard) |
| Deriva train → test | 1 de 11 variables; sin deriva del dataset |

## Estructura

```text
proyecto_heart_disease/
├── app/
│   ├── api.py                  # API REST (FastAPI): /health y /predict
│   └── model.joblib            # Pipeline entrenado que carga la API
├── dashboard/
│   ├── app.py                  # Aplicación Dash: secciones, controles y callbacks
│   ├── figuras.py              # Figuras de Plotly con estilo común
│   ├── modelos.py              # Entrenamiento sin fuga de los cuatro modelos y métricas
│   ├── artefactos.joblib       # Modelos y métricas ya calculados
│   ├── requirements.txt        # Dependencias del dashboard
│   └── assets/                 # Estilos (CSS) e imagen de portada
├── docker/
│   ├── Dockerfile              # Imagen de la API
│   └── requirements.txt        # Dependencias de producción (versiones fijadas)
├── k8s/
│   ├── deployment.yaml         # Deployment de Kubernetes
│   └── service.yaml            # Service (LoadBalancer)
├── notebooks/
│   ├── 1_model_leakage_demo.ipynb   # EDA, preprocesamiento, data leakage y ranking de modelos
│   └── 2_model_pipeline_cv.ipynb    # Pipeline + GridSearchCV, evaluación, exportación y deriva
├── fuentes/                    # Código fuente de los notebooks (formato "percent")
├── src/modelado.py             # Funciones reutilizables de carga, entrenamiento y evaluación
├── tests/                      # Pruebas de la API y del dashboard (pytest)
├── herramientas/construir_notebooks.py   # fuentes/*.py → notebooks/*.ipynb
├── .github/workflows/ci.yml    # Integración continua (flake8 + pytest)
├── data/heart.csv              # Dataset
├── drift_report.html           # Reporte de deriva (train vs test)
├── drift_report_simulado.html  # Reporte de deriva con una población simulada
├── model.joblib                # Copia del modelo (estructura de la Etapa 0)
├── requirements-dev.txt        # Entorno de desarrollo
└── README.md
```

## Preparar el entorno

```bash
python -m venv .venv
```

```bash
.venv\Scripts\python -m pip install -r requirements-dev.txt
```

## Etapas 1 y 2 — Notebooks

Los notebooks ya están ejecutados. Para regenerarlos desde `fuentes/` y volver a ejecutarlos:

```bash
.venv\Scripts\python herramientas\construir_notebooks.py
```

```bash
.venv\Scripts\python -m jupyter nbconvert --to notebook --execute --inplace notebooks\1_model_leakage_demo.ipynb notebooks\2_model_pipeline_cv.ipynb
```

El segundo notebook guarda el modelo en `app/model.joblib` y genera `drift_report.html`.

## Etapa 3 — API con FastAPI y Docker

Sin Docker:

```bash
.venv\Scripts\python -m uvicorn app.api:app --port 8000
```

Con Docker (requiere Docker Desktop iniciado):

```bash
docker build -t heart-api -f docker/Dockerfile .
```

```bash
docker run -p 8000:8000 heart-api
```

Documentación interactiva: <http://localhost:8000/docs>. Petición de ejemplo:

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{\"Age\":54,\"Sex\":\"M\",\"ChestPainType\":\"ASY\",\"RestingBP\":140,\"Cholesterol\":239,\"FastingBS\":0,\"RestingECG\":\"Normal\",\"MaxHR\":120,\"ExerciseAngina\":\"Y\",\"Oldpeak\":1.5,\"ST_Slope\":\"Flat\"}"
```

Respuesta: `{"heart_disease_probability": 0.978, "prediction": 1}`

## Etapa 4 — Kubernetes local (Minikube)

```bash
minikube start
```

```bash
minikube image load heart-api:latest
```

```bash
kubectl apply -f k8s/deployment.yaml
```

```bash
kubectl apply -f k8s/service.yaml
```

```bash
kubectl get pods,svc
```

```bash
minikube service heart-service
```

El último comando abre la URL del servicio. Con el Kubernetes de Docker Desktop no hace
falta `minikube image load` y el servicio queda en <http://localhost:80>.

## Etapa 5 — Integración continua

`.github/workflows/ci.yml` instala las dependencias, revisa el estilo y ejecuta las pruebas en
cada *push*. Lo mismo en local:

```bash
.venv\Scripts\python -m flake8 app/ tests/ src/ dashboard/
```

```bash
.venv\Scripts\python -m pytest tests/ -v
```

## Etapa 6 — Monitoreo

`drift_report.html` compara la distribución de cada variable entre train (referencia) y test
(actual) con Evidently. `drift_report_simulado.html` muestra el mismo reporte con una
población alterada a propósito, para comprobar que la deriva se detecta.

## Decisiones respecto al enunciado

* **Nombre del objetivo y variables categóricas.** El dataset de Kaggle usa `HeartDisease`
  (no `target`) y tiene cinco variables de texto, así que el `MinMaxScaler` del ejemplo se
  integra en un `ColumnTransformer` con `OneHotEncoder`.
* **Ceros imposibles.** `Cholesterol` = 0 (172 pacientes) y `RestingBP` = 0 se tratan como
  faltantes: mediana aprendida en train más una columna indicadora, todo dentro del `Pipeline`.
* **Demostración de fuga.** En el código del enunciado `leaky_feature` queda en ambos flujos;
  aquí se separan tres casos (fuga del objetivo, preprocesamiento antes de dividir, selección de
  variables antes de validar) frente al flujo correcto.
* **Entrada de la API.** En vez de `features: list` se usan campos con nombre y validación
  (`Age`, `Sex`, …): el modelo necesita los nombres de columna y así se rechazan datos inválidos.
* **Versiones fijadas.** El modelo se guardó con scikit-learn 1.3.0; `docker/requirements.txt`
  fija la misma versión para que la imagen y la CI lo carguen sin diferencias.
* **Imagen local en Kubernetes.** El `Deployment` usa `heart-api:latest` con
  `imagePullPolicy: IfNotPresent`; para Docker Hub se cambia por `<TU_USUARIO_DOCKER>/heart-api`.
* **Carpeta `tests/`.** La CI del enunciado ejecuta `pytest tests/`, por lo que se agregó.
