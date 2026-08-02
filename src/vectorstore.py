import os
import faiss
import numpy as np
import pickle

from typing import List, Any, Optional
from sentence_transformers import SentenceTransformer

from src.embedding import EmbeddingPipeline


class FaissVectorStore:

    def __init__(
        self,
        persist_dir: str = "faiss_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):

        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)

        self.index: Optional[faiss.Index] = None
        self.metadata: List[Any] = []

        self.embedding_model = embedding_model

        self.model = SentenceTransformer(
            embedding_model
        )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        print(
            f"[INFO] Loaded embedding model: {embedding_model}"
        )


    def exists(self) -> bool:

        faiss_path = os.path.join(
            self.persist_dir,
            "faiss.index"
        )

        meta_path = os.path.join(
            self.persist_dir,
            "metadata.pkl"
        )

        return (
            os.path.exists(faiss_path)
            and os.path.exists(meta_path)
        )


    def build_from_documents(
        self,
        documents: List[Any]
    ):

        print(
            f"[INFO] Building vector store from {len(documents)} documents..."
        )

        embedding_pipeline = EmbeddingPipeline(
            model_name=self.embedding_model,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )


        chunks = embedding_pipeline.chunk_documents(
            documents
        )


        print(
            f"[INFO] Created {len(chunks)} chunks"
        )


        embeddings = embedding_pipeline.embed_chunks(
            chunks
        )


        embeddings = np.array(
            embeddings
        ).astype("float32")


        metadata = [
            {
                "text": chunk.page_content,
                "source": chunk.metadata.get(
                    "source",
                    "unknown"
                )
            }
            for chunk in chunks
        ]


        self.add_embeddings(
            embeddings,
            metadata
        )


        self.save()



        print(
            "[INFO] Vector store created successfully"
        )



    def add_embeddings(
        self,
        embeddings: np.ndarray,
        metadatas: List[Any]
    ):

        dimension = embeddings.shape[1]


        if self.index is None:

            self.index = faiss.IndexFlatL2(
                dimension
            )


        self.index.add(
            embeddings
        )


        self.metadata.extend(
            metadatas
        )


        print(
            f"[INFO] Added {len(embeddings)} vectors"
        )



    def save(self):

        if self.index is None:

            raise RuntimeError(
                "Cannot save empty FAISS index"
            )


        faiss_path = os.path.join(
            self.persist_dir,
            "faiss.index"
        )

        meta_path = os.path.join(
            self.persist_dir,
            "metadata.pkl"
        )


        faiss.write_index(
            self.index,
            faiss_path
        )


        with open(meta_path, "wb") as f:

            pickle.dump(
                self.metadata,
                f
            )


        print(
            f"[INFO] Saved vector store: {self.persist_dir}"
        )



    def load(self):

        if not self.exists():

            raise FileNotFoundError(
                "FAISS index does not exist. "
                "Run build_from_documents() first."
            )


        faiss_path = os.path.join(
            self.persist_dir,
            "faiss.index"
        )

        meta_path = os.path.join(
            self.persist_dir,
            "metadata.pkl"
        )


        self.index = faiss.read_index(
            faiss_path
        )


        with open(meta_path, "rb") as f:

            self.metadata = pickle.load(f)



        print(
            f"[INFO] Loaded FAISS store from {self.persist_dir}"
        )



    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ):

        if self.index is None:

            raise RuntimeError(
                "FAISS index not loaded"
            )


        distances, indices = self.index.search(
            query_embedding,
            top_k
        )


        results = []


        for idx, distance in zip(
            indices[0],
            distances[0]
        ):

            if idx == -1:
                continue


            results.append(
                {
                    "index": int(idx),
                    "distance": float(distance),
                    "metadata": self.metadata[idx]
                }
            )


        return results



    def query(
        self,
        query_text: str,
        top_k: int = 5
    ):

        print(
            f"[INFO] Query: {query_text}"
        )


        embedding = self.model.encode(
            [query_text]
        )


        embedding = embedding.astype(
            "float32"
        )


        return self.search(
            embedding,
            top_k
        )