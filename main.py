"""
AmbedkarGPT - Assignment 1: RAG Q&A System
"""
import os
from langchain_community.document_loaders import TextLoader
from langchain_community.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

SPEECH_FILE = "speech.txt"
CHROMA_DB_DIR = "./chroma_db"

def setup_rag_system():
    print("\n" + "=" * 80)
    print("🔧 SETTING UP AMBEDKARGPT")
    print("=" * 80)
    
    print(f"\n📄 Loading '{SPEECH_FILE}'...")
    if not os.path.exists(SPEECH_FILE):
        raise FileNotFoundError(f"Create {SPEECH_FILE}!")
    
    loader = TextLoader(SPEECH_FILE, encoding='utf-8')
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} document(s)")
    
    print(f"\n✂️  Splitting text...")
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50, separator="\n")
    texts = text_splitter.split_documents(documents)
    print(f"✅ Created {len(texts)} chunks")
    
    print(f"\n🧠 Loading embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print(f"✅ Embeddings loaded")
    
    print(f"\n💾 Creating vector store...")
    vectorstore = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory=CHROMA_DB_DIR)
    vectorstore.persist()
    print(f"✅ Vector store created")
    
    print(f"\n🤖 Connecting to Ollama...")
    llm = Ollama(model="mistral", temperature=0.1)
    print(f"✅ Connected")
    
    print(f"\n🔗 Creating QA chain...")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )
    print(f"✅ Ready!")
    print("=" * 80)
    return qa_chain

def main():
    print("\n🙏 AMBEDKARGPT - Dr. B.R. Ambedkar Q&A System\n")
    
    try:
        qa_chain = setup_rag_system()
        print("\n💬 Ask questions! (Type 'quit' to exit)\n")
        
        while True:
            question = input("🎤 Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!\n")
                break
            
            if question:
                print(f"\n🔍 Searching...\n")
                result = qa_chain({"query": question})
                print(f"💡 ANSWER:\n{'-'*80}")
                print(result['result'])
                print(f"{'-'*80}\n")
    
    except Exception as e:
        print(f"\n❌ Error: {e}\n")

if __name__ == "__main__":
    main()