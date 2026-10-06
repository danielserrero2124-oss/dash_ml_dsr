# Contexto del problema

Este capítulo corresponde a la pestaña **Contexto** del dashboard y la amplía.

```{image} ../dashboard/assets/corazon.svg
:alt: Corazón con señal de electrocardiograma
:width: 420px
:align: center
```

## Importancia

Las enfermedades cardiovasculares son la principal causa de muerte en el mundo. Según la
Organización Mundial de la Salud, en 2019 provocaron 17,9 millones de muertes, el 32 % del
total mundial. El 85 % de ellas se debió a infartos y accidentes cerebrovasculares, y más de
tres cuartas partes ocurrieron en países de ingresos bajos y medios.

Buena parte de esas muertes es prevenible si el riesgo se identifica a tiempo. Los
historiales clínicos digitales contienen datos de rutina (edad, presión arterial, colesterol,
electrocardiograma, prueba de esfuerzo) con los que un modelo de clasificación puede estimar
la probabilidad de enfermedad y ayudar a priorizar a quién estudiar primero.

## Objetivo

Construir, evaluar y desplegar un modelo de clasificación binaria que prediga si un paciente
tiene enfermedad cardíaca (`HeartDisease` = 1) o no (0), aplicando prácticas de MLOps en un
entorno local, y presentar los resultados en un dashboard interactivo.

## Los datos

El dataset *Heart Failure Prediction* (Kaggle, fedesoriano) reúne 918 pacientes de cinco
conjuntos clásicos de cardiología (Cleveland, Hungría, Suiza, Long Beach VA y Stalog) tras
eliminar duplicados. Tiene 11 variables predictoras y la variable objetivo.

| Variable | Tipo | Significado |
|---|---|---|
| `Age` | Numérica | Edad en años |
| `Sex` | Categórica | Sexo: M o F |
| `ChestPainType` | Categórica | Dolor torácico: TA (angina típica), ATA (atípica), NAP (no anginoso), ASY (asintomático) |
| `RestingBP` | Numérica | Presión arterial en reposo (mm Hg) |
| `Cholesterol` | Numérica | Colesterol sérico (mg/dl); 0 = no medido |
| `FastingBS` | Binaria | Glucosa en ayunas mayor a 120 mg/dl (1 = sí) |
| `RestingECG` | Categórica | Electrocardiograma en reposo: Normal, ST (anomalía de onda ST-T), LVH (hipertrofia ventricular) |
| `MaxHR` | Numérica | Frecuencia cardíaca máxima alcanzada |
| `ExerciseAngina` | Categórica | Angina inducida por ejercicio (Y/N) |
| `Oldpeak` | Numérica | Depresión del segmento ST inducida por ejercicio |
| `ST_Slope` | Categórica | Pendiente del segmento ST en ejercicio: Up, Flat, Down |
| `HeartDisease` | Objetivo | 1 = enfermedad cardíaca, 0 = sano |

## Indicadores de la pestaña Contexto

| Indicador | Valor | Lectura |
|---|---|---|
| Pacientes | 918 | Muestra pequeña: obliga a usar validación cruzada y a desconfiar de diferencias mínimas |
| Variables | 11 | 5 numéricas, 5 categóricas y 1 binaria |
| Con enfermedad | 55,3 % (508) | Clases casi balanceadas; prevalencia muy superior a la de la población general |
| Mejor AUC en validación | 0,934 | `RandomForestClassifier`, estimado sin usar el conjunto de prueba |

## Por qué importa hacerlo bien

En un tamizaje los dos errores no cuestan lo mismo. Un **falso positivo** envía a un paciente
sano a un examen adicional; un **falso negativo** deja sin estudiar a un paciente enfermo. Por
eso el proyecto reporta la sensibilidad y la especificidad, además de la exactitud, y el
dashboard permite mover el umbral de decisión.

Tampoco basta con un número alto: un modelo evaluado con fuga de datos puede aparentar un
desempeño que no tendrá en producción. El capítulo 4 lo demuestra con tres ejemplos, y todo
el flujo del proyecto está diseñado para evitarlo.

## Etapas del proyecto

| Etapa | Qué se hizo | Capítulo |
|---|---|---|
| 0. Estructura | Carpetas para código, cuadernos, API, despliegue y CI | 7 |
| 1. EDA y preprocesamiento | Exploración, ceros imposibles, demostración de fuga | 2 y 4 |
| 2. Entrenamiento seguro | `Pipeline` + `GridSearchCV` con validación cruzada | 3 y 5 |
| 3. Despliegue local | API FastAPI y contenedor Docker | 7 |
| 4. Orquestación | Deployment y Service en Kubernetes | 7 |
| 5. Integración continua | Estilo y pruebas en GitHub Actions | 7 |
| 6. Monitoreo | Deriva de datos con Evidently | 5 |
