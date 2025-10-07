import os
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import tiktoken
from sentence_transformers import CrossEncoder


load_dotenv()

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import Document

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

os.environ["GROQ_API_KEY"] = GROQ_API_KEY
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

llm = ChatGroq(model="gemma2-9b-it")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def load_docs(docs_dir: Path):
    """Load .txt and .pdf from docs_dir -> LangChain Documents list."""
    docs = []
    for p in docs_dir.glob("**/*"):
        if p.is_file() and p.stat().st_size > 0: 
            if p.suffix.lower() == ".txt":
                docs += TextLoader(str(p), encoding="utf-8").load()
            elif p.suffix.lower() == ".pdf":
                docs += PyPDFLoader(str(p)).load()
    return docs

def build_splits(docs, chunk_size=800, chunk_overlap=120):
    """Build splits with token-aware chunking."""
    try:
        encoding = tiktoken.encoding_for_model("gemma2-9b-it")
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    def token_length(text):
        return len(encoding.encode(text))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=token_length,
        separators=["\n\n", "\n", "。", "！", "?", "!", ".", " ", ""]
    )
    return splitter.split_documents(docs)

def ensure_vectorstore(splits, embeddings: Embeddings, persist_path: Path):
    """Create or load FAISS index."""
    if persist_path.exists():
        return FAISS.load_local(str(persist_path), embeddings, allow_dangerous_deserialization=True)
    vs = FAISS.from_documents(splits, embeddings)
    vs.save_local(str(persist_path))
    return vs

docs_dir = Path("docs")
docs = load_docs(docs_dir)
splits = build_splits(docs)

persist_path = Path("vectorstore")
vectorstore = ensure_vectorstore(splits, embeddings, persist_path)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "lambda_mult": 0.7} 
)

system = SystemMessagePromptTemplate.from_template(
    "ඔබ සිංහලෙන් පිළිතුරු දෙන සහායකයෙක් වනවා. සපයන ලද මූලාශ්‍ර පමණක් භාවිතා කර පිළිතුරු දෙන්න. "
    "පිළිතුර මූලාශ්‍රවල නොමැති නම්, මම නොදනිමි කියා සහ සොයන්නට හෝ වැඩි විස්තර ඉල්ලන්නට යෝජනා කරන්න. "
    "පිළිතුරු විස්තරාත්මකව සහ පහසුවෙන් තේරුම් ගත හැකි ආකාරයකින් සකස් කරන්න. උදාහරණ සහ විස්තර සමඟ පිළිතුරු දෙන්න."
)

human = HumanMessagePromptTemplate.from_template(
    "පරිශීලක ප්‍රශ්නය: {question}\n\nපහත සන්දර්භය භාවිතා කරන්න: {context}\n\nමූලාශ්‍ර රේඛාගතව උපුටා දක්වන්න: [1], [2] වැනි ආකාරයකින්."
)

prompt = ChatPromptTemplate.from_messages([system, human])

class EnhancedQAChain:
    """Enhanced QA chain with re-ranking and better formatting."""

    def __init__(self, llm, retriever, cross_encoder):
        self.llm = llm
        self.retriever = retriever
        self.cross_encoder = cross_encoder

    def retrieve_and_rerank(self, query):
        candidates = self.retriever.get_relevant_documents(query)
        reranked = self.rerank_with_cross_encoder(query, candidates, top_k=3)
        return reranked

    def rerank_with_cross_encoder(self, query, docs, top_k=3):
        """Re-rank documents using cross-encoder."""
        if not docs:
            return []

        pairs = [[query, doc.page_content] for doc in docs]

        scores = self.cross_encoder.predict(pairs)

        scored_docs = list(zip(docs, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        return [doc for doc, score in scored_docs[:top_k]]

    def format_docs(self, docs):
        """Format documents with citations."""
        formatted = []
        source_map = {
            "lesson1.txt": "පාඩම 1: AI හි මූලික කරුණු"
        }
        for i, doc in enumerate(docs, 1):
            filename = doc.metadata.get("source", "unknown").split("/")[-1]
            source = source_map.get(filename, filename)
            formatted.append(f"[{i}] {doc.page_content} (Source: {source})")
        return "\n\n".join(formatted)

    def __call__(self, query_dict):
        query = query_dict["query"]

        docs = self.retrieve_and_rerank(query)
        context = self.format_docs(docs)

        full_prompt = prompt.format_messages(question=query, context=context)

        response = self.llm.invoke(full_prompt)

        return {
            "result": response.content,
            "source_documents": docs
        }

qa_chain = EnhancedQAChain(llm, retriever, cross_encoder)

app = Flask(__name__)
CORS(app)

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({'error': 'Question is required'}), 400
    
    question = data['question']
    result = qa_chain({'query': question})
    
    source_map = {
        "lesson1.txt": "පාඩම 1: AI හි මූලික කරුණු"
    }
    sources = []
    for doc in result['source_documents']:
        filename = doc.metadata.get('source', 'Unknown').split('/')[-1]
        sources.append(source_map.get(filename, filename))
    
    response = {
        'answer': result['result'],
        'sources': sources
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
