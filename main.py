import os
import json
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_classic.chains import RetrievalQA

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
    try:
        loader = DirectoryLoader(CORPUS_DIR, glob="**/*.txt", loader_cls=TextLoader)
        documents = loader.load()
        
        if len(documents) == 0:
            print(f"WARNING: No documents found in {CORPUS_DIR}")
            print("Please ensure the corpus folder contains the 6 speech files.")
            return None
            
    except Exception as e:
        print(f"Error loading documents from {CORPUS_DIR}.")
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
    print("Initializing HuggingFace Embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # 4. Create and persist the Vector Store
    print(f"Creating/Loading ChromaDB at {persist_dir}...")
    
    # Remove old database if exists to ensure fresh start
    if os.path.exists(persist_dir):
        import shutil
        shutil.rmtree(persist_dir)
        print(f"Removed old database at {persist_dir}")
    
    vectorstore = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    print("ChromaDB setup complete.")

    # 5. Initialize LLM (Ollama)
    print(f"Initializing Ollama with model: {MODEL_NAME}...")
    try:
        llm = Ollama(model=MODEL_NAME, temperature=0)
    except Exception as e:
        print(f"Error initializing Ollama: {e}")
        print("Make sure Ollama is running: ollama serve")
        print("And that mistral is pulled: ollama pull mistral")
        return None

    # 6. Create the RetrievalQA Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True
    )
    
    return qa_chain

def main():
    """
    Main function for the command-line Q&A system.
    """
    print("\n" + "="*60)
    print("  AmbedkarGPT - RAG Q&A System")
    print("="*60)
    
    # Check if corpus exists
    if not os.path.exists(CORPUS_DIR):
        print(f"\nERROR: Corpus directory '{CORPUS_DIR}' not found!")
        print("Please create the corpus folder with the 6 speech files.")
        return
    
    # Use the medium chunk size as the default
    qa_chain = setup_rag_pipeline(chunk_size=550, chunk_overlap=50, persist_dir=PERSIST_DIR)

    if not qa_chain:
        print("\nRAG setup failed. Please check the errors above.")
        return

    print("\n--- AmbedkarGPT RAG System Ready ---")
    print(f"LLM: {MODEL_NAME}, Embeddings: {EMBEDDING_MODEL}")
    print("Type your questions about Dr. Ambedkar's works.")
    print("Enter 'exit' or 'quit' to end the session.\n")

    while True:
        try:
            query = input("\n🔍 Your Question: ")
            
            if query.lower() in ["exit", "quit"]:
                print("\n👋 Exiting RAG system. Goodbye!")
                break
            
            if not query.strip():
                continue

            print("\n⏳ Generating answer...")
            
            # Generate the answer
            result = qa_chain({"query": query})
            
            # Output Results
            print("\n" + "="*60)
            print("📝 AI Answer:")
            print("-"*60)
            print(result['result'])
            print("="*60)
            
            # Display Sources
            sources = set()
            for doc in result['source_documents']:
                filename = os.path.basename(doc.metadata.get('source', 'Unknown'))
                sources.add(filename)
            
            if sources:
                print("\n📚 Sources Used:")
                for src in sorted(sources):
                    print(f"  • {src}")
            print("="*60)

        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("\nTroubleshooting:")
            print("  1. Ensure Ollama is running: ollama serve")
            print("  2. Check if mistral is available: ollama list")
            print("  3. Verify corpus files exist in ./corpus/")

if __name__ == "__main__":
    main()