# Example showcasing RAG (Retrieval-Augmented Generation) flow using graphlib

from graphlib import flow, node

# Define nodes for each step in the RAG process
query_node = node("Query", lambda query: query)
retrieval_node = node("Retrieval", lambda query: f"Retrieved documents based on query: {query}")
generation_node = node("Generation", lambda query, documents: f"Generated response based on query '{query}' and documents '{documents}'")
response_node = node("Response", lambda response: response)

# Define the RAG flow
rag_flow = flow(
    query_node,
    retrieval_node,
    generation_node,
    response_node
)

# Example usage
query = "What is the capital of France?"
documents = "Paris is the capital of France. It is a beautiful city."

response = rag_flow(query, documents)
print(f"Final Response: {response}")
