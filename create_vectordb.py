from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.embeddings import GPT4AllEmbeddings


pdf_data_path = "data"
vector_db_path = "vectorstores/db_faiss"

def load_documents(folder_path: str):
    loader = DirectoryLoader(folder_path, glob="BENH DA LIEU.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50
    )
    
    chunks = text_splitter.split_documents(extracted_data)
    return chunks

def create_vector_db(chunks):
    embeddings_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    db = FAISS.from_documents(chunks, embeddings_model)

    db.save_local(vector_db_path)


documents = load_documents(pdf_data_path)

# print(f"Loaded {len(documents)} documents")

chunks = create_chunks(documents)
# print(f"Created {len(chunks)} chunks")

create_vector_db(chunks)






