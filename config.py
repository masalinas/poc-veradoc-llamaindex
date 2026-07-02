# config.py
import os
from dotenv import load_dotenv

# Cargar el archivo .env local
load_dotenv()

# Configuramos las variables de entorno necesarias para LanceDB/MinIO
os.environ["AWS_ENDPOINT_URL"] = os.getenv("MINIO_ENDPOINT_URL", "http://localhost:9000")
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("MINIO_SECRET_KEY", "minioadmin")
os.environ["AWS_REGION"] = "us-east-1"  # Región dummy para bypass de AWS

# Variables de configuración exportables
MINIO_ENDPOINT = os.environ["AWS_ENDPOINT_URL"]
ACCESS_KEY_ID = os.environ["AWS_ACCESS_KEY_ID"]
SECRET_ACCESS_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]
OLLAMA_ENDPOINT = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL_NAME = "hf.co/Ralriki/multilingual-e5-large-instruct-GGUF:latest"