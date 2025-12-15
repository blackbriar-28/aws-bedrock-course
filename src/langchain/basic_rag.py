from langchain_aws import BedrockLLM as Bedrock
from langchain_aws import BedrockEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
import boto3

session = boto3.Session(profile_name="bedrock")
bedrock = session.client(service_name="bedrock-runtime", region_name="us-west-2")

my_data = [
    "The weather is nice today.",
    "Last night's game ended in a tie.",
    "Don likes to eat pizza.",
    "Don likes to eat pasta.",
]

question = "What does Don like to eat?"

model = Bedrock(model_id="amazon.titan-text-express-v1", client=bedrock)

bedrock_embeddings = BedrockEmbeddings(
  model_id="amazon.titan-embed-text-v1", client=bedrock
)

vectorstore = FAISS.from_texts(my_data, bedrock_embeddings)

retriever = vectorstore.as_retriever(
  search_kwargs={"k": 2}
)

results = retriever.invoke(question)

results_string = []
for result in results:
    results_string.append(result.page_content)
    
template = ChatPromptTemplate.from_messages(
  [
    (
      "system",
      "Answer the users question based on the following context: {context}",
    ),
    ("user", "{input}"),
  ]
)

chain = template.pipe(model)

response = chain.invoke({"input": question, "context": results_string})
print(response)                         