# search.py
import logging
import os
import lancedb
import ollama

from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding

from config import *

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

TEST_BUCKET = "veradoc-datalake"
TABLE_NAME = "document_embeddings"
TASK_DESCRIPTION = "Given a web search query, retrieve relevant passages that answer the query"

logger = logging.getLogger("veradoc.search")

# this instruct is particular for multilingual-e5-large-instruct embedding model
def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'

def query_veradoc_hybrid_search(
    bucket_name: str, 
    table_name: str, 
    query_str: str, 
    target_tag: str, 
    owner_id: str,
    top_k: int = 2
) -> list:
    storage_options = {
        "endpoint": os.getenv("MINIO_ENDPOINT_URL"),
        "access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
        "secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),        
        "allow_http": "true",
        "region": "us-east-1"
    }

    lancedb_uri = f"s3://{bucket_name}/vector_indices"

    try:
        db_connection = lancedb.connect(
            lancedb_uri,
            storage_options=storage_options
        )

        table = db_connection.open_table(table_name)

        resp = ollama.embed(model=EMBED_MODEL_NAME, input=get_detailed_instruct(TASK_DESCRIPTION, query_str))
        query_vector = resp["embeddings"][0]

        # Búsqueda híbrida: vector + BM25, fusionados automáticamente (RRF)
        search = (
            table.search(query_type="hybrid")
                .vector(query_vector)
                .text(query_str)
                .limit(2)
        )     

        # is exist some tag filter by then
        if target_tag:
            sql_filter = f"metadata.owner_id = '{owner_id}' AND metadata.tags LIKE '%{target_tag}%'"

            search = search.where(sql_filter)        

        results = search.to_pandas()            
    except Exception as e:
        logger.error(f"Error retrieving best chunks from lanceDB en MinIO: {e}")

        return []

    return results

def query_veradoc_vector_store(
    bucket_name: str, 
    table_name: str, 
    query_str: str, 
    target_tag: str, 
    owner_id: str,
    top_k: int = 2
) -> list:
    """
    Conecta de forma nativa a LanceDB sobre MinIO, aplica pre-filtros 
    por metadatos (owner_id y tags) y realiza una búsqueda vectorial.
    """
    logger.info(f"Iniciando búsqueda semántica para el propietario: {owner_id}")

    # 1. Configurar opciones de almacenamiento nativas de Rust para LanceDB
    storage_options = {
        "endpoint": os.getenv("MINIO_ENDPOINT_URL"),
        "access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
        "secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),        
        "allow_http": "true",
        "region": "us-east-1"
    }

    lancedb_uri = f"s3://{bucket_name}/vector_indices"

    try:
        # 2. Instanciar el modelo de embedding local para la query
        embed_model = OllamaEmbedding(
            model_name=EMBED_MODEL_NAME,
            base_url=OLLAMA_ENDPOINT
        )

        # 3. Establecer conexión e indexar el Vector Store en LlamaIndex
        db_connection = lancedb.connect(
            lancedb_uri,
            storage_options=storage_options
        )

        vector_store = LanceDBVectorStore(
            connection=db_connection,
            table_name=table_name,
            mode="overwrite"
        )
        
        # 4. bing vector store with the embedded model to be used
        index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

        # 5. Construir filtros estrictos en formato plano (Usa operador 'contains' para los tags como string)
        # 1. CONSTRUIR LA CLÁUSULA SQL NATIVA
        # Como 'tags' se guardó como string "arquitectura,local-rag", usamos LIKE con comodines
        # Los metadatos en LanceDB se guardan anidados bajo la columna 'metadata'
        sql_filter = f"metadata.owner_id = '{owner_id}' AND metadata.tags LIKE '%{target_tag}%'"
        logger.info(f"Aplicando filtro SQL nativo: {sql_filter}")

        # 6. retrieve best k chunks from lanceDB using the native filter too
        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=top_k,        
            vector_store_kwargs={"where": sql_filter} # 'vector_store_kwargs' pasa el argumento 'where' directo al cliente .query() de LanceDB
        )

        return retriever.retrieve(get_detailed_instruct(TASK_DESCRIPTION, query_str))            
    except Exception as e:
        logger.error(f"Error retrieving best chunks from lanceDB en MinIO: {e}")

        return []

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def main():        
    # print("\n--- PASO 1:Busqueda nativa ---")
    # resultados = query_veradoc_hybrid_search(
    #     bucket_name=TEST_BUCKET,
    #     table_name=TABLE_NAME,
    #     query_str="Enumera todas las secciones llamadas: Preparación de oferta, incluyendo todos los argumentos y valores de cada una de ellas. Dales formato del tipo secciones y subsecciones",
    #     #query_str="Dime la calle del Juzgado de 1ª instancia Nº68 de Madrid",
    #     target_tag="local-rag",
    #     owner_id="user_miguel_2026"
    # )

    # print(resultados[["doc_id", "text"]])

    print("\n--- PASO 1: Ejecución de Consulta desde el nuevo Módulo ---")
    resultados = query_veradoc_vector_store(
        bucket_name=TEST_BUCKET,
        table_name=TABLE_NAME,
        query_str="Enumera todas las secciones llamadas: Preparación de oferta, incluyendo todos los argumentos y valores de cada una de ellas. Dales formato del tipo secciones y subsecciones",
        #query_str="Dime la calle del Juzgado de 1ª instancia Nº68 de Madrid",
        target_tag="local-rag",
        owner_id="user_miguel_2026"
    )
    
    # Imprimir los fragmentos recuperados
    print("\n--- PASO 2: Imprimir resultados ---")

    for i, match in enumerate(resultados):
        print(f"\n[Resultado #{i+1}] Similitud: {match.score:.4f}")
        #print(f"Texto del Chunk: {match.node.get_content()[:150]}...")
        print(f"Texto del Chunk: {match.node.get_content()}...")

if __name__ == "__main__":
    main()        