import asyncio
from typing import Any, Dict, List, Tuple
import numpy as np

from flow import Node, Flow

# --- Placeholder for actual ML/NLP libraries ---
class PlaceholderEmbeddingModel:
    def __init__(self, model_name="text-embedding-ada-002"):
        self.model_name = model_name
        print(f"Initialized PlaceholderEmbeddingModel: {self.model_name}")

    def get_embedding(self, text: str) -> List[float]:
        """Simulates generating an embedding for a text string."""
        # Simple hash-based embedding for consistent but dummy results
        val = hash(text) % 1000
        return [float(val) / 1000.0] * 5  # 5-dimensional dummy embedding

class PlaceholderLLM:
    def __init__(self, model_name="gpt-3.5-turbo"):
        self.model_name = model_name
        print(f"Initialized PlaceholderLLM: {self.model_name}")

    def generate(self, prompt: str) -> str:
        """Simulates generating text from a prompt."""
        return f"LLM Answer based on: '{prompt[:100]}...'"

class SimpleVectorStore:
    def __init__(self):
        self.vectors: Dict[str, List[float]] = {} # content_id -> embedding
        self.documents: Dict[str, str] = {}      # content_id -> original_text_chunk
        self.next_id = 0
        print("Initialized SimpleVectorStore")

    def add_vector(self, text_chunk: str, vector: List[float]):
        content_id = f"doc_chunk_{self.next_id}"
        self.next_id += 1
        self.vectors[content_id] = vector
        self.documents[content_id] = text_chunk
        print(f"Stored vector for chunk: {content_id}")

    def find_most_similar(self, query_vector: List[float], top_k: int = 1) -> List[Tuple[str, str, float]]:
        """Finds the most similar vector(s) using cosine similarity (simulated)."""
        if not self.vectors:
            return []

        similarities = []
        for content_id, doc_vector in self.vectors.items():
            # Simulate cosine similarity: dot product of normalized vectors
            # For dummy embeddings, this will be simplistic
            dot_product = np.dot(np.array(query_vector), np.array(doc_vector))
            # Assuming embeddings are somewhat normalized for simplicity
            similarity = dot_product 
            similarities.append((content_id, self.documents[content_id], similarity))

        similarities.sort(key=lambda x: x[2], reverse=True)
        return similarities[:top_k]

# --- Global instances (or pass them around in context) ---
embedding_model = PlaceholderEmbeddingModel()
llm = PlaceholderLLM()
vector_store = SimpleVectorStore()

# --- Node Definitions ---

class LoadDocumentsNode(Node):
    async def pre(self, context: Dict[str, Any]) -> None:
        print(f"[{self.name}] Pre: Preparing to load documents.")
    
    async def exec(self, context: Dict[str, Any]) -> None:
        print(f"[{self.name}] Exec: Loading documents.")
        # Sample documents
        context["documents"] = [
            {"id": "doc1", "content": "The sky is blue. Clouds are white."},
            {"id": "doc2", "content": "Photosynthesis is the process by which green plants use sunlight, water, and carbon dioxide to create their own food."},
            {"id": "doc3", "content": "The capital of France is Paris. Paris is known for the Eiffel Tower. The sky is blue."},
            {"id": "doc4", "content": "Large Language Models (LLMs) are a type of artificial intelligence."}
        ]
        print(f"[{self.name}] Loaded {len(context['documents'])} documents.")

    async def post(self, context: Dict[str, Any]) -> str:
        print(f"[{self.name}] Post: Documents loaded.")
        return "documents_loaded"

class SplitDocumentsNode(Node):
    async def pre(self, context: Dict[str, Any]) -> None:
        print(f"[{self.name}] Pre: Preparing to split documents.")
        if "documents" not in context:
            raise ValueError("Documents not found in context for splitting.")
            
    async def exec(self, context: Dict[str, Any]) -> None:
        print(f"[{self.name}] Exec: Splitting documents into chunks.")
        all_chunks = []
        # Simple splitting strategy: by sentence (crude) or fixed length
        for doc in context["documents"]:
            # Splitting by sentence for this example
            sentences = doc["content"].split('. ')
            for i, sentence in enumerate(sentences):
                if sentence.strip(): # Avoid empty chunks
                    all_chunks.append({"doc_id": doc["id"], "chunk_id": f"{doc['id']}_chunk{i}", "text": sentence.strip() + "."})
        context["document_chunks"] = all_chunks
        print(f"[{self.name}] Split into {len(all_chunks)} chunks.")

    async def post(self, context: Dict[str, Any]) -> str:
        print(f"[{self.name}] Post: Documents split.")
        return "chunks_created"

class EmbedChunksNode(Node):
    async def pre(self, context: Dict[str, Any]) -> None:
        print(f"[{self.name}] Pre: Preparing to embed chunks.")
        if "document_chunks" not in context:
            raise ValueError("Document chunks not found in context for embedding.")

    async def exec(self, context: Dict[str, Any]) -> None:
        print(f"[{self.name}] Exec: Embedding document chunks.")
        embedded_chunks = []
        for chunk in context["document_chunks"]:
            embedding = embedding_model.get_embedding(chunk["text"])
            embedded_chunks.append({**chunk, "embedding": embedding})
            # print(f"[{self.name}] Embedded chunk: {chunk['chunk_id']}")
        context["embedded_chunks"] = embedded_chunks
        print(f"[{self.name}] Embedded {len(embedded_chunks)} chunks.")

    async def post(self, context: Dict[str, Any]) -> str:
        print(f"[{self.name}] Post: Chunks embedded.")
        return "chunks_embedded"

class StoreEmbeddingsNode(Node):
    async def pre(self, context: Dict[str, Any]) -> None:
        print(f"[{self.name}] Pre: Preparing to store embeddings.")
        if "embedded_chunks" not in context:
            raise ValueError("Embedded chunks not found in context for storing.")

    async def exec(self, context: Dict[str, Any]) -> None:
        print(f"[{self.name}] Exec: Storing embeddings in vector store.")
        for emb_chunk in context["embedded_chunks"]:
            vector_store.add_vector(emb_chunk["text"], emb_chunk["embedding"])
        context["vector_store_ready"] = True # Signal that store is populated
        print(f"[{self.name}] Stored {len(context['embedded_chunks'])} embeddings.")
    
    async def post(self, context: Dict[str, Any]) -> str:
        print(f"[{self.name}] Post: Embeddings stored.")
        return "default" # Terminate offline flow

class GetQueryNode(Node):
    async def pre(self, context: Dict[str, Any]) -> None:
        print(f"[{self.name}] Pre: Ready to get user query.")
    
    async def exec(self, context: Dict[str, Any]) -> None:
        print(f"[{self.name}] Exec: Getting user query.")
        if "user_query_override" in context: # For potential testing/chaining
            query = context["user_query_override"]
            print(f"[{self.name}] Using overridden query: {query}")
        else:
            query = input("Enter your query: ")
        context["user_query"] = query

    async def post(self, context: Dict[str, Any]) -> str:
        print(f"[{self.name}] Post: Query received.")
        return "query_received"

class EmbedQueryNode(Node):
    async def pre(self, context: Dict[str, Any]) -> None:
        print(f"[{self.name}] Pre: Preparing to embed query.")
        if "user_query" not in context:
            raise ValueError("User query not found in context for embedding.")

    async def exec(self, context: Dict[str, Any]) -> None:
        print(f"[{self.name}] Exec: Embedding user query.")
        query_embedding = embedding_model.get_embedding(context["user_query"])
        context["query_embedding"] = query_embedding
        print(f"[{self.name}] Query embedded.")

    async def post(self, context: Dict[str, Any]) -> str:
        print(f"[{self.name}] Post: Query embedded.")
        return "query_embedded"

class RetrieveRelevantDocumentNode(Node):
    async def pre(self, context: Dict[str, Any]) -> None:
        print(f"[{self.name}] Pre: Preparing to retrieve relevant document.")
        if "query_embedding" not in context:
            raise ValueError("Query embedding not found for retrieval.")
        if not context.get("vector_store_ready"):
            raise ValueError("Vector store is not ready/populated.")

    async def exec(self, context: Dict[str, Any]) -> None:
        print(f"[{self.name}] Exec: Retrieving relevant document from vector store.")
        similar_docs = vector_store.find_most_similar(context["query_embedding"], top_k=10)
        if similar_docs:
            context["retrieved_document_chunk"] = similar_docs[0][1] # Get the text of the most similar chunk
            context["retrieved_document_id"] = similar_docs[0][0]
            print(f"[{self.name}] Retrieved document chunk: {similar_docs[0][0]}")
        else:
            context["retrieved_document_chunk"] = "No relevant document found."
            print(f"[{self.name}] No relevant document found.")

    async def post(self, context: Dict[str, Any]) -> str:
        print(f"[{self.name}] Post: Document retrieval complete.")
        return "document_retrieved"

class GenerateAnswerNode(Node):
    async def pre(self, context: Dict[str, Any]) -> None:
        print(f"[{self.name}] Pre: Preparing to generate answer.")
        if "user_query" not in context or "retrieved_document_chunk" not in context:
            raise ValueError("Query or retrieved document not found for answer generation.")

    async def exec(self, context: Dict[str, Any]) -> None:
        print(f"[{self.name}] Exec: Generating answer using LLM.")
        prompt = f"User Query: {context['user_query']}\n\nRelevant Information: {context['retrieved_document_chunk']}\n\nAnswer:"
        context["llm_prompt"] = prompt
        answer = llm.generate(prompt)
        context["generated_answer"] = answer
        print(f"[{self.name}] Generated Answer: {answer}")

    async def post(self, context: Dict[str, Any]) -> str:
        print(f"[{self.name}] Post: Answer generated.")
        return "terminate:online_flow_complete" # Terminate online flow

# --- Flow Definitions ---

# Offline Flow: Indexing
load_docs = LoadDocumentsNode("LoadDocs")
split_docs = SplitDocumentsNode("SplitDocs")
embed_chunks = EmbedChunksNode("EmbedChunks")
store_embeddings = StoreEmbeddingsNode("StoreEmbeddings")

# Define connections for offline flow
load_docs - "documents_loaded" >> split_docs
split_docs - "chunks_created" >> embed_chunks
embed_chunks - "chunks_embedded" >> store_embeddings

offline_phase = Flow("OfflineIndexingFlow", root_node=load_docs)

# Online Flow: Retrieval and Generation
get_query = GetQueryNode("GetQuery")
embed_query = EmbedQueryNode("EmbedQuery")
retrieve_doc = RetrieveRelevantDocumentNode("RetrieveDoc")
generate_answer = GenerateAnswerNode("GenerateAnswer")

# Define connections for online flow
get_query - "query_received" >> embed_query
embed_query - "query_embedded" >> retrieve_doc
retrieve_doc - "document_retrieved" >> generate_answer

online_phase = Flow("OnlineRAGFlow", root_node=get_query)

offline_phase >> online_phase

rag = Flow("RAGFlow", root_node=offline_phase)

async def main():
    print("Starting RAG Flow...")
    # Initial context can be empty or pre-populated
    initial_context = {"app_name": "RAG Demo"} 
    last_output, final_context = await rag.start(initial_context)
    print(f"\nMain RAG Flow finished with output: {last_output}")
    # print(f"Final context: {final_context}")

if __name__ == "__main__":
    asyncio.run(main())
