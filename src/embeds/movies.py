import boto3
import json

session = boto3.Session(profile_name="bedrock")
bedrock = session.client(service_name="bedrock-runtime", region_name="us-west-2")

model_id = "amazon.titan-embed-text-v1"

prompt = "Please recommend movies with a theme similar to the movie 'The Shining'."

native_request = {"inputText": prompt}

request = json.dumps(native_request)

response = bedrock.invoke_model(modelId=model_id,body=request)

model_response = json.loads(response["body"].read())

embedding = model_response["embedding"]
input_token_count = model_response["inputTextTokenCount"]

print("\nYour input:")
print(prompt)
print(f"Number of input tokens: {input_token_count}")
print(f"Size of the generated embedding: {len(embedding)}")
print("Embedding:")
print(embedding)