import os
import json
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitters import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import Ollama
from langchain.chains import RetrievalQA

# --- Configuration ---
PERSIST_DIR = "./chroma_db_rag"
CORPUS_DIR = "./corpus"
MODEL_NAME = "mistral"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def setup_rag_pipeline(chunk_size=500, chunk_overlap=50, persist_dir=PERSIST_DIR):
    """
    Sets up the RAG pipeline components: Document Loading, Chunking, Embedding, 
    Vector Store (ChromaDB), and LLM (Ollama).
    """
    print(f"--- Setting up RAG Pipeline with Chunk Size: {chunk_size} ---")

    # 1. Load Documents from the corpus folder
    # Use a DirectoryLoader to load all .txt files in the corpus/ folder
    try:
        loader = DirectoryLoader(CORPUS_DIR, glob="**/*.txt", loader_cls=TextLoader)
        documents = loader.load()
    except Exception as e:
        print(f"Error loading documents from {CORPUS_DIR}. Please ensure the directory exists and contains files.")
        print(f"Details: {e}")
        return None

    # 2. Split the text into manageable chunks
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator="\n",
        length_function=len,
    )
    texts = text_splitter.split_documents(documents)
    print(f"Loaded {len(documents)} documents, split into {len(texts)} chunks.")

    # 3. Create Embeddings
    # Using HuggingFace Embeddings for local operation
    print("Initializing HuggingFace Embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # 4. Create and persist the Vector Store
    print(f"Creating/Loading ChromaDB at {persist_dir}...")
    vectorstore = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    vectorstore.persist()
    print("ChromaDB setup complete.")

    # 5. Initialize LLM (Ollama)
    # Assumes Ollama server is running locally and 'mistral' model is pulled
    print(f"Initializing Ollama with model: {MODEL_NAME}...")
    llm = Ollama(model=MODEL_NAME)

    # 6. Create the RetrievalQA Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}), # Retrieve top 5 chunks
        return_source_documents=True
    )
    
    return qa_chain

def main():
    """
    Main function for the command-line Q&A system.
    """
    # Use the medium chunk size as the default running size for the interactive demo
    qa_chain = setup_rag_pipeline(chunk_size=550, chunk_overlap=50, persist_dir=PERSIST_DIR)

    if not qa_chain:
        print("RAG setup failed. Exiting.")
        return

    print("\n--- AmbedkarGPT RAG System Ready ---")
    print(f"LLM: {MODEL_NAME}, Embeddings: {EMBEDDING_MODEL}")
    print("Enter 'exit' or 'quit' to end the session.")

    while True:
        query = input("\nYour Question: ")
        if query.lower() in ["exit", "quit"]:
            print("Exiting RAG system. Goodbye!")
            break
        
        if not query.strip():
            continue

        try:
            # Generate the answer
            result = qa_chain({"query": query})
            
            # --- Output Results ---
            print("\n" + "="*50)
            print("AI Answer:")
            print(result['result'])
            print("="*50)
            
            # Display Sources
            sources = set()
            for doc in result['source_documents']:
                # Extract filename from metadata source path
                filename = os.path.basename(doc.metadata.get('source', 'Unknown Source'))
                sources.add(filename)
            
            print("Sources Used (Files):")
            print(", ".join(sources))
            print("="*50 + "\n")

        except Exception as e:
            print(f"An error occurred during query processing: {e}")
            print("Ensure your Ollama server is running and the 'mistral' model is pulled.")

if __name__ == "__main__":
    main()