from langchain_aws import ChatBedrock  # Changed from BedrockLLM
from langchain_aws import BedrockEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import boto3

session = boto3.Session(profile_name="bedrock")
bedrock = session.client(service_name="bedrock-runtime", region_name="us-west-2")

# Use ChatBedrock for Claude models
model = ChatBedrock(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    client=bedrock,
    model_kwargs={"max_tokens": 1000}
)

bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v1", client=bedrock
)

question = "What is my current employer?"

loader = PyPDFLoader("assets/linkedin.pdf")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""]
)
docs = loader.load()
splitted_docs = splitter.split_documents(docs)

# print(f"Total chunks created: {len(splitted_docs)}")

vector_store = FAISS.from_documents(splitted_docs, bedrock_embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

results = retriever.invoke(question)
results_string = [result.page_content for result in results]

# Simple prompt - Claude is smart enough to figure it out
template = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are analyzing a resume. Answer the question based on the context provided.
        
        Context from resume:
        {context}"""
    ),
    ("user", "{input}"),
])

chain = template | model
response = chain.invoke({"input": question, "context": results_string})

# ChatBedrock returns a message object, extract the content
print(f"\nBot: {response.content}")