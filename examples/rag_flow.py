# Example showcasing RAG (Retrieval-Augmented Generation) flow using graphlib

from graphlib import flow, node

# Define a Node for the Query
class QueryNode(node):
    def __init__(self, name):
        super().__init__(name)

    def run(self, query):
        return query

# Define a Node for Retrieval
class RetrievalNode(node):
    def __init__(self, name):
        super().__init__(name)

    def run(self, query):
        # Simulate retrieval of documents based on the query
        return f"Retrieved documents based on query: {query}"

# Define a Node for Generation
class GenerationNode(node):
    def __init__(self, name):
        super().__init__(name)

    def run(self, query, documents):
        # Simulate generation of a response based on the query and documents
        return f"Generated response based on query '{query}' and documents '{documents}'"

# Define a Node for the final Response
class ResponseNode(node):
    def __init__(self, name):
        super().__init__(name)

    def run(self, response):
        return response

# Define the RAG flow
rag_flow = flow(
    QueryNode("Query"),
    RetrievalNode("Retrieval"),
    GenerationNode("Generation"),
    ResponseNode("Response")
)

# Example usage
query = "What is the capital of France?"
documents = "Paris is the capital of France. It is a beautiful city."

response = rag_flow(query, documents)
print(f"Final Response: {response}")
