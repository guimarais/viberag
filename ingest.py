import os
from rag_utils import load_document, create_or_load_vectorstore

def main():
    # Create documents directory if it doesn't exist
    if not os.path.exists("documents"):
        os.makedirs("documents")
        print("Created 'documents' directory. Please add your documents there and run this script again.")
        return
    
    # Get all files from the documents directory
    document_files = []
    for file in os.listdir("documents"):
        if file.endswith((".pdf", ".txt")):
            document_files.append(os.path.join("documents", file))
    
    if not document_files:
        print("No supported documents found in the 'documents' directory.")
        print("Please add PDF or text files to the 'documents' directory.")
        return
    
    # Process all documents
    all_chunks = []
    for file_path in document_files:
        print(f"Processing {file_path}...")
        chunks = load_document(file_path)
        all_chunks.extend(chunks)
    
    print(f"\nProcessed {len(document_files)} documents into {len(all_chunks)} chunks.")
    
    # Create and persist the vector store
    print("\nCreating vector store...")
    vectorstore = create_or_load_vectorstore(all_chunks)
    print("Vector store created and persisted successfully!")

if __name__ == "__main__":
    main() 