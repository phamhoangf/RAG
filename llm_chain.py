from langchain_community.llms import CTransformers
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS


# Cau hinh
model_file = "models/vinallama-7b-chat_q5_0.gguf"


# Load LLM
def load_llm(model_file):
    llm = CTransformers(
        model=model_file,
        model_type = "llama",
        max_new_tokens=1024,
        temperature=0.01
    )
    return llm

# Tao prompt template
def create_prompt(template):
    prompt = PromptTemplate(template = template, input_variables=["context", "question"])
    return prompt


# Load db
db_faiss_path = "vectorstores/db_faiss"
embeddings_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.load_local(db_faiss_path, embeddings_model, allow_dangerous_deserialization=True)

# Chay thu chain

template = """<|im_start|>system\nSử dụng thông tin sau đây để trả lời câu hỏi. Nếu bạn không biết câu trả lời, hãy nói không biết, đừng cố tạo ra câu trả lời\n
    {context}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant"""

prompt = create_prompt(template)
llm = load_llm(model_file)

qa_chain = RetrievalQA.from_chain_type(
    llm,
    chain_type = "stuff",
    retriever = db.as_retriever(search_kwargs={"k": 3}, max_tokens_limit=1024),
    return_source_documents = False,
    chain_type_kwargs = {"prompt":prompt}
)

# Now invoke with a single query
user_query=input("Write Query Here: ")
response=qa_chain.invoke({'query': user_query})
print("RESULT: ", response["result"])
# print("SOURCE DOCUMENTS: ", response["source_documents"])
