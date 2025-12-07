import os
import json
import numpy as np
from typing import List, Dict, Any
import shutil

# LangChain Imports - FIXED for newer versions
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_classic.chains import RetrievalQA

# Evaluation Libraries
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.metrics.pairwise import cosine_similarity

# --- Configuration ---
CORPUS_DIR = "./corpus"
TEST_DATA_FILE = "test_dataset.json"
RESULTS_FILE = "test_results.json"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "mistral"

# Chunking strategies
CHUNK_CONFIGS = {
    "small": (250, 50),
    "medium": (550, 100),
    "large": (900, 150)
}

def setup_rag_pipeline(chunk_size: int, chunk_overlap: int, persist_dir: str):
    """Loads data, chunks it, embeds it, and creates a Chroma retriever."""
    print(f"\n  Setting up: chunk_size={chunk_size}, overlap={chunk_overlap}")
    
    # 1. Load Documents
    loader = DirectoryLoader(CORPUS_DIR, glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()
    
    if len(documents) == 0:
        raise ValueError(f"No documents found in {CORPUS_DIR}")

    # 2. Split the text
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator="\n",
        length_function=len,
    )
    texts = text_splitter.split_documents(documents)
    print(f"  Loaded {len(documents)} docs → {len(texts)} chunks")

    # 3. Create Embeddings
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # 4. Create Vector Store (delete old one first)
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir, ignore_errors=True)

    vectorstore = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    
    # 5. Initialize LLM
    llm = Ollama(model=LLM_MODEL, temperature=0)

    # 6. Create RetrievalQA Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True
    )
    
    return qa_chain, embeddings, vectorstore

def get_retrieved_documents(qa_chain: RetrievalQA, question: str) -> List[str]:
    """Retrieves context chunks and returns list of source filenames."""
    retriever = qa_chain.retriever
    # Use invoke instead of get_relevant_documents for newer versions
    docs = retriever.invoke(question)
    
    sources = set()
    for doc in docs:
        filename = os.path.basename(doc.metadata.get('source', ''))
        if filename:
            sources.add(filename)
    return list(sources)

def calculate_retrieval_metrics(retrieved: List[str], ground_truth: List[str], k: int = 5):
    """Calculates Hit Rate, MRR, and Precision@K."""
    is_hit = any(src in retrieved for src in ground_truth)
    
    # MRR: Find rank of first relevant document
    mrr = 0.0
    for i, src in enumerate(retrieved[:k], 1):
        if src in ground_truth:
            mrr = 1.0 / i
            break
    
    # Precision@K
    if len(retrieved) > 0:
        relevant_retrieved = sum(1 for src in retrieved[:k] if src in ground_truth)
        precision_at_k = relevant_retrieved / min(k, len(retrieved))
    else:
        precision_at_k = 0.0
        
    return is_hit, mrr, precision_at_k

def calculate_rouge(generated: str, reference: str) -> float:
    """Calculates ROUGE-L F1 score."""
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(reference, generated)
    return scores['rougeL'].fmeasure

def calculate_bleu(generated: str, reference: str) -> float:
    """Calculates BLEU score with smoothing."""
    ref = [reference.lower().split()]
    cand = generated.lower().split()
    
    smooth = SmoothingFunction()
    return sentence_bleu(ref, cand, smoothing_function=smooth.method1)

def calculate_cosine_sim(generated: str, reference: str, embeddings: HuggingFaceEmbeddings) -> float:
    """Calculates Cosine Similarity between embeddings."""
    vectors = embeddings.embed_documents([generated, reference])
    sim_matrix = cosine_similarity(np.array(vectors).reshape(2, -1))
    return float(sim_matrix[0][1])

def llm_judge(llm: Ollama, question: str, answer: str, context: str, metric_type: str) -> float:
    """
    Uses LLM to judge faithfulness and relevance.
    Returns a score between 0.0 and 1.0.
    """
    if metric_type == "faithfulness":
        prompt = f"""Rate the faithfulness of this answer to the context on a scale of 0.0 to 1.0.
Only output a single number.

Context: {context[:500]}...
Answer: {answer}

Faithfulness score (0.0-1.0):"""
    
    elif metric_type == "relevance":
        prompt = f"""Rate how well this answer addresses the question on a scale of 0.0 to 1.0.
Only output a single number.

Question: {question}
Answer: {answer}

Relevance score (0.0-1.0):"""
    
    else:
        return 0.5

    try:
        response = llm.invoke(prompt).strip()
        
        # Extract number from response
        import re
        numbers = re.findall(r'0\.\d+|1\.0|0|1', response)
        if numbers:
            score = float(numbers[0])
            return max(0.0, min(1.0, score))
        return 0.5
        
    except Exception as e:
        print(f"  Warning: LLM judge failed for {metric_type}: {e}")
        return 0.5

def evaluate_rag_system():
    """Runs comprehensive evaluation across all chunking strategies."""
    
    print("\n" + "="*70)
    print("  AMBEDKARGPT - COMPREHENSIVE RAG EVALUATION")
    print("="*70)
    
    # Download NLTK data if needed
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("\nDownloading NLTK punkt tokenizer...")
        nltk.download('punkt', quiet=True)
    
    # Load test dataset
    if not os.path.exists(TEST_DATA_FILE):
        print(f"\nERROR: {TEST_DATA_FILE} not found!")
        return
        
    with open(TEST_DATA_FILE, 'r', encoding='utf-8') as f:
        test_data = json.load(f)['test_questions']
    
    print(f"\n✓ Loaded {len(test_data)} test questions")
    
    # Check corpus
    if not os.path.exists(CORPUS_DIR):
        print(f"\nERROR: Corpus directory '{CORPUS_DIR}' not found!")
        return
    
    all_results = {}

    # Evaluate each chunking strategy
    for chunk_label, (chunk_size, chunk_overlap) in CHUNK_CONFIGS.items():
        
        persist_dir = f"./chroma_db_{chunk_label}"
        
        print("\n" + "="*70)
        print(f"  EVALUATING: {chunk_label.upper()} CHUNKS")
        print(f"  (size={chunk_size}, overlap={chunk_overlap})")
        print("="*70)

        try:
            # Setup RAG Pipeline
            qa_chain, embeddings, vectorstore = setup_rag_pipeline(
                chunk_size, chunk_overlap, persist_dir
            )
            llm = qa_chain.combine_documents_chain.llm_chain.llm
            
        except Exception as e:
            print(f"\n❌ Error setting up pipeline: {e}")
            continue
        
        chunk_results = []
        
        # Process each question
        for i, item in enumerate(test_data, 1):
            q_id = item['id']
            question = item['question']
            ground_truth = item['ground_truth']
            gt_sources = item.get('source_documents', [])
            answerable = item['answerable']
            
            print(f"\n  [{i}/{len(test_data)}] Q{q_id}: ", end='')
            
            # Skip unanswerable questions
            if not answerable:
                print("SKIPPED (unanswerable)")
                chunk_results.append({
                    "id": q_id,
                    "question": question,
                    "answerable": False,
                    "note": "Unanswerable question"
                })
                continue

            try:
                # Run RAG - use invoke instead of __call__
                result = qa_chain.invoke({"query": question})
                generated = result['result'].strip()
                context = " ".join([d.page_content for d in result['source_documents']])
                retrieved_files = get_retrieved_documents(qa_chain, question)
                
            except Exception as e:
                print(f"ERROR: {e}")
                chunk_results.append({
                    "id": q_id,
                    "question": question,
                    "answerable": True,
                    "error": str(e)
                })
                continue

            # Calculate all metrics
            is_hit, mrr, prec_k = calculate_retrieval_metrics(retrieved_files, gt_sources)
            rouge = calculate_rouge(generated, ground_truth)
            bleu = calculate_bleu(generated, ground_truth)
            cosine = calculate_cosine_sim(generated, ground_truth, embeddings)
            faith = llm_judge(llm, question, generated, context, "faithfulness")
            relev = llm_judge(llm, question, generated, context, "relevance")

            result_entry = {
                "id": q_id,
                "question": question,
                "ground_truth": ground_truth,
                "generated_answer": generated,
                "answerable": True,
                "retrieved_sources": retrieved_files,
                "ground_truth_sources": gt_sources,
                "metrics": {
                    "hit_rate": is_hit,
                    "mrr": mrr,
                    "precision_at_k": prec_k,
                    "rouge_l_f1": rouge,
                    "bleu_score": bleu,
                    "cosine_similarity": cosine,
                    "faithfulness": faith,
                    "answer_relevance": relev,
                }
            }
            chunk_results.append(result_entry)
            
            print(f"Hit:{int(is_hit)} ROUGE:{rouge:.2f} Faith:{faith:.2f}")

        # Calculate summary statistics
        valid = [r for r in chunk_results if r.get('answerable', False) and 'metrics' in r]
        
        if len(valid) == 0:
            print("\n⚠️  No valid results for this chunk size")
            continue
        
        summary = {
            "hit_rate": np.mean([r['metrics']['hit_rate'] for r in valid]),
            "mrr": np.mean([r['metrics']['mrr'] for r in valid]),
            "avg_precision_at_k": np.mean([r['metrics']['precision_at_k'] for r in valid]),
            "avg_faithfulness": np.mean([r['metrics']['faithfulness'] for r in valid]),
            "avg_answer_relevance": np.mean([r['metrics']['answer_relevance'] for r in valid]),
            "avg_rouge_l_f1": np.mean([r['metrics']['rouge_l_f1'] for r in valid]),
            "avg_bleu_score": np.mean([r['metrics']['bleu_score'] for r in valid]),
            "avg_cosine_similarity": np.mean([r['metrics']['cosine_similarity'] for r in valid]),
        }

        all_results[chunk_label] = {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "summary_metrics": {k: float(v) for k, v in summary.items()},
            "detailed_results": chunk_results
        }

        print(f"\n  📊 SUMMARY for {chunk_label.upper()}:")
        print(f"     Hit Rate: {summary['hit_rate']:.3f}")
        print(f"     MRR: {summary['mrr']:.3f}")
        print(f"     Faithfulness: {summary['avg_faithfulness']:.3f}")
        print(f"     Relevance: {summary['avg_answer_relevance']:.3f}")
        print(f"     ROUGE-L: {summary['avg_rouge_l_f1']:.3f}")

    # Save results
    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
        
    print("\n" + "="*70)
    print(f"✓ Evaluation complete! Results saved to {RESULTS_FILE}")
    print("="*70 + "\n")

if __name__ == "__main__":
    # Ensure corpus exists
    if not os.path.exists(CORPUS_DIR):
        print(f"\nERROR: Please create {CORPUS_DIR}/ with the 6 speech files!")
        print("Run the setup script first.")
        exit(1)
    
    # Check Ollama
    print("\nChecking Ollama...")
    try:
        test_llm = Ollama(model=LLM_MODEL)
        test_llm.invoke("test")
        print("✓ Ollama is working")
    except Exception as e:
        print(f"❌ Ollama error: {e}")
        print("\nPlease ensure:")
        print("  1. Ollama is running: ollama serve")
        print("  2. Mistral is pulled: ollama pull mistral")
        exit(1)
    
    evaluate_rag_system()