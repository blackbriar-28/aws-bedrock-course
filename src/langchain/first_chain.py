from langchain_aws import BedrockLLM as Bedrock
from langchain_core.prompts import PromptTemplate
import boto3

session = boto3.Session(profile_name="bedrock")
bedrock = session.client(service_name="bedrock-runtime", region_name="us-west-2")

model = Bedrock(model_id="amazon.titan-text-express-v1", client=bedrock)

def invoke_model():
  response = model.invoke("What is the most populated city in the world?")
  print(response)
  
def first_chain():
  prompt = PromptTemplate.from_template("Write a short, compelling product description for: {product}")
  chain = prompt | model
  
  response = chain.invoke({"product": "bycycle"})
  print(response)
  
first_chain()  
# invoke_model()