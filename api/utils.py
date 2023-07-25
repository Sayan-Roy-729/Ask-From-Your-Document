from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceHub, OpenAI
from langchain.chains.question_answering import load_qa_chain

def load_documents(file_path):
    loader = TextLoader(file_path)
    documents = loader.load()
    return documents

def create_chunks(documents, chunk_size: int = 1_000, chunk_overlap: int = 100):
    chunks = CharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap
    )
    docs = chunks.split_documents(documents)
    return docs

def create_embeddings(chunks, service_type: str):
    if service_type == "HuggingFace":
        embeddings = HuggingFaceEmbeddings()
    elif service_type == "OpenAI":
        embeddings = OpenAIEmbeddings(model = "text-embedding-ada-002")

    db = FAISS.from_documents(chunks, embeddings)
    return db

def similarity_search(db, query):
    docs = db.similarity_search(query)
    return docs

def question_answering(query_similarity_docs, query, service_type: str):
    if service_type == "HuggingFace":
        llm = HuggingFaceHub(
            repo_id = "google/flan-t5-xxl",
            model_kwargs = {"temperature": 0.9, "max_length": 512}
        )
    elif service_type == "OpenAI":
        llm = OpenAI()

    chains = load_qa_chain(
        llm = llm,
        chain_type = "stuff"
    )
    response = chains.run(
        input_documents = query_similarity_docs,
        question = query
    )
    return response