# MLOps: API, Docker, Kubernetes e integración continua

Este capítulo documenta las etapas 0, 3, 4 y 5 del proyecto, que no aparecen en el
dashboard. Las salidas que se muestran son las obtenidas en el equipo de desarrollo
(Windows 11, Docker Desktop 29.8 con su Kubernetes integrado, v1.36).

## Etapa 0 · Estructura del proyecto

```text
proyecto_heart_disease/
├── app/                 API REST (api.py) y modelo entrenado (model.joblib)
├── dashboard/           Dashboard interactivo (Dash + Plotly)
├── docker/              Dockerfile y dependencias de producción
├── k8s/                 Manifiestos de Kubernetes (deployment.yaml, service.yaml)
├── notebooks/           Cuadernos ejecutados
├── fuentes/             Código fuente de los cuadernos
├── informe/             Capítulos de texto de este libro
├── src/                 Funciones reutilizables (modelado.py)
├── tests/               Pruebas automáticas de la API y del dashboard
├── .github/workflows/   Integración continua (ci.yml)
├── data/heart.csv       Dataset
├── drift_report.html    Reporte de deriva de Evidently
└── model.joblib         Copia del modelo (estructura pedida en el enunciado)
```

Respecto a la estructura del enunciado se agregaron `src/`, `tests/`, `fuentes/`, `data/`,
`dashboard/` e `informe/`.

## Etapa 3 · API con FastAPI

`app/api.py` carga el `Pipeline` entrenado una sola vez al iniciar y expone dos rutas:

| Ruta | Método | Función |
|---|---|---|
| `/health` | GET | Comprueba que el servicio está vivo e indica qué clasificador está cargado |
| `/predict` | POST | Recibe los 11 datos de un paciente y devuelve la probabilidad y la predicción |

**Diferencia con el enunciado.** El ejemplo recibe `features: list`. Aquí la entrada tiene
campos con nombre y validación, porque el modelo necesita los nombres de columna para aplicar
el preprocesamiento y porque así se rechazan datos inválidos antes de predecir:

```python
class Paciente(BaseModel):
    Age: int = Field(..., ge=1, le=120)
    Sex: Literal["M", "F"]
    ChestPainType: Literal["TA", "ATA", "NAP", "ASY"]
    RestingBP: int = Field(..., ge=0, le=300)
    Cholesterol: int = Field(..., ge=0, le=1000)
    FastingBS: Literal[0, 1]
    RestingECG: Literal["Normal", "ST", "LVH"]
    MaxHR: int = Field(..., ge=30, le=250)
    ExerciseAngina: Literal["Y", "N"]
    Oldpeak: float = Field(..., ge=-10, le=10)
    ST_Slope: Literal["Up", "Flat", "Down"]
```

Petición y respuesta reales:

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"Age":54,"Sex":"M","ChestPainType":"ASY","RestingBP":140,"Cholesterol":239,"FastingBS":0,
       "RestingECG":"Normal","MaxHR":120,"ExerciseAngina":"Y","Oldpeak":1.5,"ST_Slope":"Flat"}'
```

```json
{"heart_disease_probability": 0.978289581291574, "prediction": 1}
```

Una petición incompleta (por ejemplo, solo `{"Age": 54}`) recibe un error **422** con el
detalle de los campos que faltan. FastAPI genera además documentación interactiva en `/docs`.

## Etapa 3 · Contenedor Docker

```bash
docker build -t heart-api -f docker/Dockerfile .
docker run -p 8000:8000 heart-api
```

| Aspecto | Resultado |
|---|---|
| Imagen | `heart-api:latest`, 648 MB (base `python:3.10-slim`) |
| Estado del contenedor | `healthy` (comprobación periódica de `/health`) |
| Usuario de ejecución | `apiuser`, sin privilegios de administrador |
| Predicción dentro del contenedor | 0,978 para el paciente de ejemplo, idéntica a la obtenida fuera de Docker |

Decisiones sobre el `Dockerfile` del enunciado:

* **Versiones fijadas.** El modelo se guardó con scikit-learn 1.3.0. Con `scikit-learn` sin
  versión, la imagen instalaría una más reciente y el archivo podría no cargar o predecir
  distinto. `docker/requirements.txt` fija scikit-learn, numpy, pandas y joblib.
* **`pandas` añadido.** La API construye un DataFrame con los nombres de las columnas.
* **Usuario sin privilegios y `HEALTHCHECK`.** Buenas prácticas que no cambian el
  comportamiento de la API.
* **`.dockerignore`.** Solo entran a la imagen `app/` y la lista de dependencias.

## Etapa 4 · Kubernetes local

Se usó el Kubernetes integrado en Docker Desktop en lugar de Minikube: es equivalente para
un despliegue local y comparte las imágenes de Docker, así que no hace falta publicar la
imagen en Docker Hub.

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl get deployment,pods,svc
```

```text
NAME                          READY   UP-TO-DATE   AVAILABLE
deployment.apps/heart-model   1/1     1            1

NAME                               READY   STATUS    RESTARTS
pod/heart-model-658f4f44bd-kkgpf   1/1     Running   0

NAME            TYPE           CLUSTER-IP      EXTERNAL-IP   PORT(S)
heart-service   LoadBalancer   10.96.143.225   172.18.0.5    80:31955/TCP
```

El servicio responde en el puerto 80 del equipo:

```bash
curl http://localhost/health
```

```json
{"status": "ok", "model": "RandomForestClassifier"}
```

Diferencias con los manifiestos del enunciado:

| Cambio | Motivo |
|---|---|
| `image: heart-api:latest` con `imagePullPolicy: IfNotPresent` | Usa la imagen local; con Docker Hub se sustituye por `<usuario>/heart-api` |
| `readinessProbe` y `livenessProbe` sobre `/health` | El pod recibe tráfico solo cuando el modelo está cargado y se reinicia si deja de responder |
| `resources` (CPU y memoria) | Evita que un pod consuma todos los recursos del nodo |

El `Service` es de tipo `LoadBalancer`, como pide el enunciado: redirige el puerto 80 al
puerto 8000 del contenedor y repartiría el tráfico si se aumentaran las réplicas.

## Etapa 5 · Integración continua

`.github/workflows/ci.yml` se ejecuta en cada *push*: instala las dependencias, revisa el
estilo con flake8 y ejecuta las pruebas con pytest sobre Python 3.10.

| Paso | Qué valida |
|---|---|
| `flake8 app/ tests/ src/ dashboard/` | Sintaxis y estilo del código |
| `pytest tests/` | 14 pruebas: 6 de la API y 8 del dashboard |

Pruebas de la API (`tests/test_api.py`):

| Prueba | Comprueba que… |
|---|---|
| `test_health` | el servicio responde y tiene un modelo cargado |
| `test_predict_devuelve_probabilidad_valida` | la probabilidad está entre 0 y 1 y la predicción es coherente con el umbral |
| `test_predict_ordena_bien_el_riesgo` | un perfil de alto riesgo recibe más probabilidad que uno de bajo riesgo |
| `test_predict_acepta_colesterol_no_medido` | un colesterol igual a 0 se imputa y no rompe la predicción |
| `test_predict_rechaza_categoria_invalida` | un valor no permitido devuelve error 422 |
| `test_predict_rechaza_campos_faltantes` | un campo ausente devuelve error 422 |

La ejecución en GitHub Actions terminó correctamente. Como la CI corre en Linux con
Python 3.10 y los modelos se entrenaron en Windows con Python 3.9, su éxito confirma además
que los modelos guardados se cargan igual en el entorno de producción.

**Ajuste de estilo.** El código lleva un comentario explicativo en cada línea, por lo que el
límite de longitud de línea de flake8 se amplió a 240 caracteres (archivo `.flake8`).

## Etapa 6 · Monitoreo

El reporte de deriva con Evidently se genera y se interpreta en el capítulo 5. En resumen:
entre train y test no hay deriva del dataset (1 de 11 variables, atribuible al azar), y en
una población alterada a propósito se detectan exactamente las tres variables modificadas,
con una caída del AUC de 0,930 a 0,882.

En producción, el paso siguiente sería que la API registrara cada petición y que un proceso
periódico comparase ese registro con los datos de entrenamiento usando el mismo reporte.
