# Imagen de la API de predicción de enfermedad cardíaca.
# Construcción (desde la raíz del proyecto):  docker build -t heart-api -f docker/Dockerfile .
# Ejecución:                                  docker run -p 8000:8000 heart-api

# Imagen base oficial de Python 3.10 en su variante ligera (menos tamaño y menos superficie de ataque)
FROM python:3.10-slim

# Evita que Python escriba archivos .pyc y hace que los mensajes salgan de inmediato en los registros del contenedor
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# Carpeta de trabajo dentro del contenedor; los comandos siguientes se ejecutan desde aquí
WORKDIR /app

# Copia primero solo la lista de dependencias para aprovechar la caché de capas de Docker
COPY docker/requirements.txt .

# Instala las dependencias sin guardar la caché de pip (reduce el tamaño de la imagen)
RUN pip install --no-cache-dir -r requirements.txt

# Copia el código de la API y el modelo entrenado (app/api.py y app/model.joblib)
COPY app/ ./app/

# Crea un usuario sin privilegios y lo usa para ejecutar la API (buena práctica de seguridad)
RUN useradd --create-home apiuser
USER apiuser

# Documenta el puerto en el que escucha la API
EXPOSE 8000

# Comprobación periódica de salud: Docker marca el contenedor como "unhealthy" si /health deja de responder
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Comando de arranque: servidor uvicorn escuchando en todas las interfaces, puerto 8000
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
