# Predicción de enfermedad cardíaca: informe del desarrollo

**Proyecto integrador de Aprendizaje Automático · Dashboard interactivo y MLOps local**

Este libro es el informe del desarrollo del proyecto. Acompaña al dashboard interactivo:
interpreta cada resultado que aparece en él y documenta lo que el dashboard no muestra
(la demostración de fuga de datos, la comparación de siete modelos, la API, Docker,
Kubernetes, la integración continua y el monitoreo).

* **Código fuente:** <https://github.com/danielserrero2124-oss/dash_ml_dsr>
* **Dashboard:** se ejecuta con `python -m dashboard.app` (ver el capítulo *El dashboard*).

## Resumen

| Elemento | Resultado |
|---|---|
| Problema | Clasificación binaria: `HeartDisease` = 1 (enfermedad) o 0 (sano) |
| Datos | *Heart Failure Prediction* (Kaggle): 918 pacientes, 11 variables clínicas |
| Modelos del dashboard | `LogisticRegression`, `RandomForestClassifier`, `KNeighborsClassifier`, `GradientBoostingClassifier` |
| Modelo elegido | `RandomForestClassifier` (mayor AUC de validación cruzada: 0,934) |
| Desempeño en test | AUC 0,930 · sensibilidad 94,1 % · especificidad 84,1 % |
| Variable más importante | `ST_Slope` (pendiente del segmento ST en la prueba de esfuerzo) |
| Despliegue | API FastAPI en Docker y Kubernetes local; integración continua en GitHub Actions |
| Monitoreo | Reporte de deriva de datos con Evidently |

## Cómo leer este informe

| Capítulo | Contenido | ¿Está en el dashboard? |
|---|---|---|
| 1. Contexto del problema | Importancia, datos y diccionario de variables | Sí (pestaña *Contexto*) |
| 2. EDA: figuras e interpretación | Cada figura de la pestaña *EDA*, interpretada | Sí, con interpretación ampliada |
| 3. ML Models: resultados e interpretación | Cada indicador y figura de la pestaña *ML Models* | Sí, con análisis adicionales |
| 4. Fuga de datos y comparación de siete modelos | Tres casos de *data leakage* y ranking completo | No |
| 5. Pipeline, evaluación y monitoreo | Modelo final, exportación y deriva con Evidently | No |
| 6. El dashboard | Arquitectura, diseño y publicación | No |
| 7. MLOps | API, Docker, Kubernetes e integración continua | No |
| 8. Conclusiones | Hallazgos, limitaciones y trabajo futuro | No |

Las figuras de los capítulos 2 y 3 son las mismas del dashboard y conservan la
interactividad: al pasar el cursor se muestran los valores.

## Reproducibilidad

* Semilla fija (42) en la partición, la validación cruzada y los modelos.
* Versiones fijadas: Python 3.9 (desarrollo) y 3.10 (Docker y CI), scikit-learn 1.3.0.
* Todo el código está comentado línea por línea; los cuadernos se generan desde `fuentes/`.
