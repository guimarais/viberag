import os
from typing import List, Dict
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()

# Initialize embeddings
embeddings = OpenAIEmbeddings()

def load_document(file_path: str) -> List[str]:
    """Load a document and split it into chunks."""
    # Determine the loader based on file extension
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path)
    
    # Load the document
    documents = loader.load()
    
    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    return text_splitter.split_documents(documents)

def create_or_load_vectorstore(docs: List[str] = None, persist_directory: str = "vectorstore"):
    """Create a new vector store or load an existing one."""
    if docs:
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        vectorstore.persist()
    else:
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
    
    return vectorstore

def setup_rag_chain(vectorstore):
    """Set up the RAG chain with a custom prompt."""
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    
    prompt_template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
    {context}
    
    Question: {question}
    Answer: """
    
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    
    chain_type_kwargs = {"prompt": PROMPT}
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs=chain_type_kwargs,
        return_source_documents=True
    )
    
    return qa_chain

def process_query(qa_chain, query: str) -> Dict:
    """Process a query through the RAG chain."""
    result = qa_chain({"query": query})
    
    return {
        "answer": result["result"],
        "source_documents": result["source_documents"]
    } 