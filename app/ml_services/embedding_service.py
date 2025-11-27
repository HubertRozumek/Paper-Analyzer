from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
import logging
import uuid

logger = logging.getLogger(__name__)

class EmbeddingService:

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = SentenceTransformer(model_name)

        self.chroma_client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory="./chroma_db"
        ))

        logger.info(f"Initialized embedding service with model: {model_name}")

    def create_embeddings(self, texts: List[str]) -> List[List[float]]:

        try:
            embeddings = self.model.encode(texts, show_progress_bar=True)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Failed to create embeddings: {e}")
            raise

    def create_collection(self, collection_name: str) -> chromadb.Collection:

        try:

            try:
                self.chroma_client.delete_collection(collection_name)
            except:
                pass

            collection = self.chroma_client.create_collection(
                name = collection_name,
                matadata={"hnsw:space": "cosine"}
            )

            logger.info(f"Created collection: {collection_name}")
            return collection

        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            raise

    def add_chunks_to_collection(self,
                                 collection_name: str,
                                 chunks: List[Dict]
                                 ) -> List[str]:

        try:
            collection = self.chroma_client.get_collection(collection_name)

            texts = [chunk['content'] for chunk in chunks]
            embeddings = self.create_embeddings(texts)

            ids = [str(uuid.uuid4()) for _ in chunks]

            metadatas = []
            for chunk in chunks:
                metadata = {
                    'chunk_index': chunk['chunk_index'],
                    'page_number': chunk.get('page_number', 0),
                    'section': chunk.get('metadata', {}).get('section', 'other'),
                }
                metadatas.append(metadata)

            collection.add(
                embeddings = embeddings,
                documents = texts,
                metadata = metadatas,
                ids = ids,
            )

            logger.info(f"Added {len(chunks)} chunks to collection: {collection_name}")
            return ids

        except Exception as e:
            logger.error(f"Failed to add chunks to collection: {e}")
            raise

    def search(self,
               collection_name: str,
               query: str,
               top_k: int = 5,
               filter_metadata: Optional[Dict] = None
               ) -> List[Dict]:

        try:
            collection = self.chroma_client.get_collection(collection_name)

            query_embedding = self.model.encode([query])[0].tolist()

            results = collection.query(
                query_embeddings = [query_embedding],
                n_retries = top_k,
                where=filter_metadata
            )

            formatted_results = []
            for idx in range(len(results['ids'][0])):
                formatted_results.append({
                    'id': results['ids'][0][idx],
                    'content': results['documents'][0][idx],
                    'metadata': results['metadatas'][0][idx],
                    'distance': results['distance'][0][idx],
                    'similarity_score': 1 - results['distance'][0][idx],
                })

            return formatted_results

        except Exception as e:
            logger.error(f"Failed to search results: {e}")
            raise

    def delete_collection(self, collection_name: str):

        try:
            self.chroma_client.delete_collection(collection_name)
            logger.info(f"Deleted collection: {collection_name}")
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")