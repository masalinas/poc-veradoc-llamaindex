import os
import logging
import argparse
import sys
import urllib.parse
import lancedb
import pyarrow as pa

from llama_index.core.readers.base import BaseReader
from llama_index.core import VectorStoreIndex, Document as LlamaDocument
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.readers.s3 import S3Reader
from llama_index.readers.file import PDFReader
from llama_index.readers.legacy_office import LegacyOfficeReader
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.lancedb import LanceDBVectorStore

from PIL import Image
import pytesseract

from config import *

# Configuración del Logger para producción en VeraDoc
logger = logging.getLogger("veradoc.ingestion")
logging.basicConfig(level=logging.INFO)

TEST_BUCKET = "veradoc-datalake"

# List all global metadata fields you want your database to support
UNIVERSAL_METADATA_FIELDS = ["file_name", "file_path", "page_label"]

# Extractor personalizado para Imágenes (OCR)
class OCRReader(BaseReader):
    def load_data(self, file_path: os.PathLike, extra_info: dict = None, fs = None, **kwargs) -> list[LlamaDocument]:
        # If a file system is passed, we should read it through the fs object
        if fs is not None:
            with fs.open(file_path, "rb") as f:
                image = Image.open(f)
                text = pytesseract.image_to_string(image, lang='spa')
        else:
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image, lang='spa')
        
        return [LlamaDocument(text=text, metadata=extra_info or {})]
 
def _get_minio_metadata(bucket_name: str, object_key: str) -> tuple[set[str], str]:
    """
    Función Mock que simula la extracción de metadatos (tags y owner_id) 
    de un objeto almacenado en MinIO (boto3 head_object en producción).
    """
    logger.info(f"[MOCK] Recuperando metadatos para s3://{bucket_name}/{object_key}")
    
    # Valores por defecto para la simulación
    owner_id = "user_miguel_2026"
    tags = {"documentacion", "arquitectura", "local-rag"}
    
    # Simulación dinámica opcional basada en la extensión o carpetas
    if "manuales" in object_key.lower():
        tags.add("manual-tecnico")
    if ".pdf" in object_key.lower():
        tags.add("pdf-format")
        
    logger.info(f"[MOCK] Metadatos obtenidos -> owner_id: {owner_id}, tags: {tags}")
    
    return tags, owner_id
    
def get_llamaindex_file_extractor() -> dict:
    """Mapea cada extensión con su lector específico en LlamaIndex"""
    return {
        ".pdf": PDFReader(),
        ".doc": LegacyOfficeReader(),        
        ".png": OCRReader(),
        ".jpg": OCRReader(),
        ".jpeg": OCRReader(),
        #".doc": AsposeDocReader(),        
        # Formatos como .docx, .txt, .md, .csv se manejan automáticamente 
        # por los defaults de LlamaIndex si no los especificas aquí.
    }

def split_doc_by_chunks(bucket_name: str, object_key: str) -> list:
    sanitized_key = urllib.parse.unquote_plus(object_key).lstrip("/")
    
    # 1. Obtener los extractores unificados
    file_extractor = get_llamaindex_file_extractor()
    
    # 2. Configurar el lector conectado a tu MinIO
    reader = S3Reader(
        bucket=bucket_name,
        key=sanitized_key,
        aws_endpoint_url=MINIO_ENDPOINT,
        file_extractor=file_extractor
    )
    
    try:
        # Descarga y parsea el archivo según su extensión automáticamente
        documents = reader.load_data()
    except Exception as e:
        logger.error(f"Error extrayendo datos de '{sanitized_key}': {e}")
        return []
        
    if not documents:
        logger.warning(f"No se recuperó contenido para '{sanitized_key}'.")
        return []

    # 3. Chunking estratégico (Cambiamos a TokenTextSplitter)
    # Usamos un tamaño de 2500 para mantener el contexto semántico completo de bloques complejos
    splitter = TokenTextSplitter(chunk_size=2500, chunk_overlap=40)
    nodes = splitter.get_nodes_from_documents(documents)
    
    # 4. Inyección homogénea de metadatos del Data Lake
    # Simulamos la recuperación previa de tags y owner_id desde MinIO (boto3)
    tags, owner_id = _get_minio_metadata(bucket_name, sanitized_key)
    
    # Convertimos la lista/set de tags a un string plano separado por comas
    # Ejemplo: {'arquitectura', 'local-rag'} -> "arquitectura,local-rag"
    tags_flat_string = ",".join(list(tags))

    for node in nodes:
        node.metadata["source"] = sanitized_key
        node.metadata["tags"] = tags_flat_string
        node.metadata["owner_id"] = owner_id
        
        # Le decimos a LlamaIndex que no use el ID de propietario ni los tags 
        # al calcular el embedding, evitando "ruido" en la distancia vectorial.
        node.excluded_embed_metadata_keys = ["owner_id", "tags"]
        
    return nodes

def embedding_chunks_by_model(nodes: list, bucket_name: str, table_name: str = "document_embeddings"):
    """
    Toma los nodes procesados por LlamaIndex, genera embeddings locales con Ollama
    y los persiste directamente en LanceDB utilizando MinIO como datalake.
    """
    if not nodes:
        logger.warning("No se recibieron nodos para indexar.")
        return None

    logger.info(f"Iniciando pipeline de ingesta para {len(nodes)} chunks...")

    # 1. Configuración del modelo de Embeddings Local (Ollama)
    # n_predict, timeouts o urls se configuran de manera nativa aquí
    ollama_embedding = OllamaEmbedding(
        model_name=EMBED_MODEL_NAME,  # Reemplaza por tu modelo local preferido (ej. bge-large, mxbai-embed-large)
        base_url=OLLAMA_ENDPOINT,
        embed_batch_size=10,  # Ajusta según la capacidad de tu GPU (RTX 5090 / M1)
        #query_instruction="query: "
    )

    # 1. Ensure EVERY node contains EVERY universal field to match the global schema structure
    # If a field (like page_label) isn't present, we force it to None so Apache Arrow creates a valid null entry.
    for node in nodes:
        for field in UNIVERSAL_METADATA_FIELDS:
            if field not in node.metadata or node.metadata[field] is None:
                node.metadata[field] = ""  # Crucial: Forces explicit Utf8 text type assignment!

    # 2. Configuración de la conexión nativa de LanceDB hacia MinIO
    # LanceDB utiliza internamente la infraestructura de almacenamiento de Apache Arrow (ObjectStore), 
    # por lo que mapear un URI 's3://' funcionará directamente contra tu MinIO.
        
    # Construimos el URI del almacenamiento híbrido de LanceDB en tu MinIO
    # Ejemplo: s3://veradoc-datalake/vector_indices/
    lancedb_uri = f"s3://{bucket_name}/vector_indices"

    storage_options = {
        "endpoint": os.getenv("MINIO_ENDPOINT_URL"),
        "access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
        "secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),        
        "allow_http": "true",
        "region": "us-east-1"
    }

    try:
        logger.info(f"Conectando LanceDB con almacenamiento en Data Lake: {lancedb_uri}")

        # vector_store = LanceDBVectorStore(
        #     uri=lancedb_uri,
        #     table_name=table_name,
        #     api_key=None, # No se requiere API Key para entornos locales
        #     mode="append",  # Añade nuevos chunks sin sobreescribir la tabla existente
        #     storage_options=storage_options
        # )   

        db_connection = lancedb.connect(
            lancedb_uri,
            storage_options=storage_options
        )

        # Check if the table already exists in your MinIO Data Lake
        if table_name in db_connection.list_tables().tables:
            logger.info(f"La tabla '{table_name}' ya existe. Modo: APPEND (Añadiendo datos...)")
            
            # Connect normally without overwriting anything
            vector_store = LanceDBVectorStore(
                connection=db_connection,
                table_name=table_name,
                mode="append"
            )
        else:
            logger.info(f"La tabla '{table_name}' no existe. Creando esquema unificado por primera vez...")
            
            # Define the uniform blueprint schema
            custom_schema = pa.schema([
                ("id", pa.string()),                           
                ("doc_id", pa.string()),                       
                ("text", pa.string()),                         
                ("vector", pa.list_(pa.float32(), 1024)),     
                ("file_name", pa.string()),                    
                ("file_path", pa.string()),                    
                ("page_label", pa.string()),  # Nullable automatically for images/docs
            ])

            # Initialize the table ONLY the very first time
            vector_store = LanceDBVectorStore(
                connection=db_connection,
                table_name=table_name,
                mode="overwrite", # Used only to build the initial empty table structure
                schema=custom_schema
            )
  
    except Exception as e:
        logger.error(f"Error crítico al conectar LanceDB con MinIO: {e}")
        raise

    # 3. Creación del Ingestion Pipeline unificado
    # Aquí puedes añadir más transformaciones en el futuro (ej. extractores de palabras clave adicionales)
    pipeline = IngestionPipeline(
        transformations=[
            # This will automatically break down your massive OCR text into 512-token pieces
            #TokenTextSplitter(chunk_size=512, chunk_overlap=50),            
            ollama_embedding  # El pipeline se encarga de llamar a Ollama en lotes óptimos
        ],
        vector_store=vector_store
    )

    try:
        # 4. Ejecución del Pipeline
        # Esto calcula los embeddings localmente e inyecta los datos vectoriales y metadatos en MinIO
        pipeline.run(nodes=nodes)
        logger.info("¡Pipeline ejecutado con éxito! Chunks y vectores persistidos en el Data Lake.")
        
    except Exception as e:
        logger.error(f"Error durante la ejecución del pipeline de Ollama/LanceDB: {e}")
        raise

    # 5. Opcional: Retornar el índice listo para consultas de VeraDoc
    # Esto te permitirá realizar búsquedas inmediatas si lo necesitas en el mismo flujo
    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=ollama_embedding)

    return index

def main():
    parser = argparse.ArgumentParser(description="Ingestion pipeline into Minio Data Lake.")

    parser.add_argument(
        "--file-path", 
        dest='file_path',
        help="Path to the target image file (e.g., docs/image.jpg)"
    )

    args = parser.parse_args()
    
    try:
        print("--- Extracción y Chunking con LlamaIndex ---")
        chunks = split_doc_by_chunks(TEST_BUCKET, args.file_path)
        print(f"Se han generado {len(chunks)} chunks listos para procesar.")
        
        print("--- Procesamiento de Embeddings e Ingesta ---")
        indice_resultado = embedding_chunks_by_model(
            nodes=chunks, 
            bucket_name=TEST_BUCKET,
            table_name="document_embeddings"
        )
        print("--- ¡Flujo completado con éxito en VeraDoc! ---")
    except Exception as e:
        logging.error(f"Execution failed: {e}")

        sys.exit(1)    

if __name__ == "__main__":
    main()