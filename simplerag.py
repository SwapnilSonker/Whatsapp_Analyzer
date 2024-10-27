from langchain_community.document_loaders import TextLoader
from langchain.chains import create_retrieval_chain
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from langchain.chains.combine_documents import create_stuff_documents_chain


loader = TextLoader("speech.txt")
text_loader = loader.load()
text_loader


# loader = WebBaseLoader("https://python.langchain.com/docs/integrations/document_loaders/web_base/")
loader = WebBaseLoader(web_paths=("https://python.langchain.com/docs/integrations/document_loaders/web_base/",),
                       bs_kwargs=dict(parse_only=bs4.SoupStrainer(
                        
                       )))
docs = loader.load()
docs[0]

loader = PyPDFLoader("attention.pdf")
dox = loader.load()
dox

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split = text_splitter.split_documents(dox)


## now using the vector embeddings and vector store


embeddings = OllamaEmbeddings(
    model="llama3.2",
)

vector_store = Chroma.from_documents(documents=split, embedding=embeddings)

retriever = vector_store.as_retriever()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def ollama_llm(question, context):
    formatted_prompt = f"Question: {question}\n\nContext: {context}"
    response = ollama.chat(model="llama3.2", messages=[{'role':'user', 'content':formatted_prompt}])
    return response['message']['content']

def rag_chain(question):
    retrieved_docs = retriever.invoke(question)
    formatted_context = format_docs(retrieved_docs)
    return ollama_llm(question, formatted_context)

result = rag_chain("Define Attention?")
print("result" , result)




llm = OllamaLLM(model="llama3.2")

prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context
  Think step by step before answering any question.
  <context>
  {context}
  </context>
  Question: {input}""")



document_chain= create_stuff_documents_chain(llm, prompt)


retirever = vector_store.as_retriever()
retirever


retireval_chain = create_retrieval_chain(retirever, document_chain)

respons=retireval_chain.invoke({"input": "The Attention mechanism"})

respons['answer']