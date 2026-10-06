# Conclusiones

## Hallazgos

1. **El problema es predecible con datos clínicos de rutina.** Los cuatro modelos del
   dashboard alcanzan un AUC de 0,93 en pacientes que nunca vieron, con sensibilidad cercana
   al 94 % y especificidad entre 84 % y 88 %.
2. **La señal está en la prueba de esfuerzo.** La pendiente del segmento ST (`ST_Slope`) es,
   con diferencia, la variable más importante en todos los modelos; le siguen el tipo de dolor
   torácico, el sexo, el colesterol y la angina inducida por ejercicio.
3. **La complejidad del modelo importa poco.** Una regresión logística iguala a los ensambles:
   la diferencia entre el mejor y el peor de los cuatro (0,010 de AUC) es menor que la
   variación entre pliegues (± 0,03). Se eligió `RandomForestClassifier` por tener el mayor
   AUC de validación cruzada, pero la regresión logística sería una alternativa igual de
   válida y más simple.
4. **El umbral es una decisión clínica, no técnica.** Con 0,5 quedan 6 enfermos sin detectar
   y 13 falsas alarmas; con 0,3, 4 y 27. El dashboard permite explorar ese compromiso.
5. **La fuga de datos puede fabricar resultados.** Una variable derivada de la etiqueta lleva
   el AUC a 1,000, y seleccionar variables antes de validar aparenta 0,71 con ruido puro. El
   flujo correcto (dividir primero y preprocesar dentro del `Pipeline`) da la cifra honesta.
6. **No hay sobreajuste oculto.** En los cuatro modelos el AUC de test cae dentro del
   intervalo de la validación cruzada, y las curvas de aprendizaje convergen.

## Limitaciones

* **Muestra pequeña.** Con 918 pacientes (184 de prueba), las diferencias de centésimas
  entre modelos no son concluyentes.
* **Población seleccionada.** Son pacientes remitidos a estudio cardiológico, con una
  prevalencia del 55 %. Las probabilidades no son trasladables a la población general sin
  recalibrar.
* **Desbalance por sexo.** El 79 % son hombres; el desempeño en mujeres se apoya en solo 193
  casos y no se evaluó por separado.
* **Datos de varios hospitales.** El colesterol no medido se concentra en un subgrupo con
  mucha más enfermedad. Parte de lo que el modelo aprende de esa señal refleja cómo se
  recogieron los datos y podría no repetirse en otro hospital.
* **Sin validación externa.** El conjunto de prueba proviene de la misma fuente que el de
  entrenamiento.
* **Probabilidades sin calibrar.** Se evaluó la capacidad de ordenar el riesgo (AUC), no si
  una probabilidad de 0,8 corresponde a un 80 % real de enfermos.

## Trabajo futuro

* Evaluar la calibración de las probabilidades y, si hace falta, corregirla.
* Medir el desempeño por subgrupos (sexo, edad, con y sin colesterol medido).
* Elegir el umbral con un criterio de costos acordado con personal clínico.
* Registrar las peticiones de la API y automatizar el reporte de deriva.
* Validar el modelo con datos de otra institución antes de cualquier uso real.

## Entregables

| Entregable | Ubicación |
|---|---|
| Dashboard interactivo | `python -m dashboard.app` (código en `dashboard/`) |
| Informe del desarrollo | Este libro |
| Cuadernos del enunciado | `notebooks/1_model_leakage_demo.ipynb` y `notebooks/2_model_pipeline_cv.ipynb` |
| API, Docker y Kubernetes | `app/`, `docker/` y `k8s/` |
| Integración continua | `.github/workflows/ci.yml` |
| Reporte de deriva | `drift_report.html` |

Este proyecto es un ejercicio académico. El modelo no es un dispositivo médico y no debe
usarse para tomar decisiones sobre pacientes reales.
