from dotenv import load_dotenv
from langchain_groq import ChatGroq

from src.vectorstore import FaissVectorStore

load_dotenv()


class RAGSearch:

    def __init__(
        self,
        vector_store: FaissVectorStore,
        llm_model: str = "llama-3.3-70b-versatile"
    ):

        self.vectorstore = vector_store

        import os
        groq_api_key = os.getenv("GROQ_API_KEY")


        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=llm_model
        )


        print(
            f"[INFO] Groq LLM initialized: {llm_model}"
        )



    def search_and_summarize(
        self,
        query: str,
        top_k: int = 5
    ) -> str:


        print(
            f"[INFO] Searching for: {query}"
        )


        results = self.vectorstore.query(
            query,
            top_k=top_k
        )


        texts = []


        for result in results:

            metadata = result.get(
                "metadata"
            )

            if metadata:

                texts.append(
                    metadata.get(
                        "text",
                        ""
                    )
                )


        context = "\n\n".join(
            texts
        )


        if not context:

            return "No relevant documents found."



        prompt = f"""
You are a helpful AI assistant.

Answer the query using only the provided context.

Query:
{query}


Context:
{context}


Answer:
"""


        response = self.llm.invoke(
            prompt
        )


        return response.content