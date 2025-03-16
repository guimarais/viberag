import streamlit as st
from rag_utils import create_or_load_vectorstore, setup_rag_chain, process_query

# Set page config
st.set_page_config(
    page_title="RAG Query System",
    page_icon="🤖",
    layout="wide"
)

# Add title and description
st.title("🤖 RAG Query System")
st.markdown("""
This system allows you to query your document collection using RAG (Retrieval-Augmented Generation).
Simply enter your question below, and the system will search through your documents to provide a relevant answer.
""")

# Initialize session state for the RAG system
if 'rag_initialized' not in st.session_state:
    with st.spinner("Initializing RAG system..."):
        # Load the vector store and setup the chain
        vectorstore = create_or_load_vectorstore()
        qa_chain = setup_rag_chain(vectorstore)
        st.session_state['qa_chain'] = qa_chain
        st.session_state['rag_initialized'] = True

# Create the query interface
query = st.text_area("Enter your question:", height=100)
submit_button = st.button("Submit Query")

if submit_button and query:
    with st.spinner("Processing your query..."):
        try:
            # Process the query
            result = process_query(st.session_state['qa_chain'], query)
            
            # Display the answer
            st.markdown("### Answer")
            st.write(result["answer"])
            
            # Display sources
            st.markdown("### Sources")
            for i, doc in enumerate(result["source_documents"], 1):
                with st.expander(f"Source {i}: {doc.metadata.get('source', 'Unknown source')}"):
                    st.markdown(f"```\n{doc.page_content}\n```")
        
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")

# Add footer with instructions
st.markdown("---")
st.markdown("""
### How to use
1. Make sure your documents are in the `documents` folder
2. Run `python ingest.py` to process new documents
3. Use this interface to query your document collection
""") 