from rag_utils import create_or_load_vectorstore, setup_rag_chain, process_query

def main():
    # Load the existing vector store
    print("Loading vector store...")
    vectorstore = create_or_load_vectorstore()
    
    # Set up the RAG chain
    qa_chain = setup_rag_chain(vectorstore)
    
    print("\nRAG system ready! Type 'quit' to exit.")
    
    while True:
        # Get query from user
        query = input("\nEnter your question: ").strip()
        
        if query.lower() == 'quit':
            break
        
        if not query:
            continue
        
        # Process the query
        try:
            result = process_query(qa_chain, query)
            
            print("\nAnswer:", result["answer"])
            
            # Print sources
            print("\nSources:")
            for i, doc in enumerate(result["source_documents"], 1):
                print(f"\n{i}. From document: {doc.metadata.get('source', 'Unknown source')}")
                print(f"Relevant excerpt: {doc.page_content[:200]}...")
        
        except Exception as e:
            print(f"\nError processing query: {str(e)}")

if __name__ == "__main__":
    main() 