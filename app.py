import fitz  # PyMuPDF
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.schema import Document
import os


# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text_content = ""
    for page in doc:
        text_content += page.get_text()
    doc.close()
    return text_content

# Split the extracted text by warning titles
def split_by_warning_sections(text_content):
    # Splitting by common patterns in warning titles
    warning_sections = text_content.split('\n\n')
    documents = []
    for section in warning_sections:
        if "Fault" in section or "Warning" in section:  # Simplistic check for warnings
            documents.append(Document(page_content=section.strip()))
    return documents

# Create embeddings and automatically persist in Chroma
def create_and_persist_embeddings(documents, openai_api_key, persist_directory="chroma_db"):
    texts = [doc.page_content for doc in documents]
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    chroma_store = Chroma.from_texts(texts=texts, embedding=embeddings, persist_directory=persist_directory)
    return chroma_store

# Updated setup_retrieval_qa function
def setup_retrieval_qa(chroma_store, openai_api_key):
    retriever = chroma_store.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        openai_api_key=openai_api_key
    )

     # Create the retrieval QA chain
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return rag_chain

# Function to run a query using the RetrievalQA system
def run_query(rag_chain, query):
    result = rag_chain({"query": query})
    return result

def main_setup():
    openai_api_key = os.getenv("OPENAI_API_KEY")  # Get API key from environment variable
    if not openai_api_key:
        raise ValueError("Please set the OPENAI_API_KEY environment variable")
    
    file_path = "file_path"

    text_content = extract_text_from_pdf(file_path)
    documents = split_by_warning_sections(text_content)
    chroma_store = create_and_persist_embeddings(documents, openai_api_key)
    return chroma_store, openai_api_key

# Run the setup once and store necessary components
chroma_store, openai_api_key = main_setup()

# Initialize the RetrievalQA system (run once after setup)
rag_chain = setup_retrieval_qa(chroma_store, openai_api_key)

# Example
query = "I can see the check engine fault on my dashboard. What does this mean and what should I do about it?"
result = run_query(rag_chain, query)
print(result['result'])  # Print the answer
