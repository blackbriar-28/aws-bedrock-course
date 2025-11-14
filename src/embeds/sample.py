import boto3
import json

from similarity import cosineSimilarity as cosine_similarity

session = boto3.Session(profile_name="bedrock")
bedrock = session.client(service_name="bedrock-runtime", region_name="us-west-2")

facts = [
    "The Eiffel Tower is located in Paris.",
    "The Great Wall of China is visible from space.",
    "The Amazon River is the longest river in the world.",
    "John F. Kennedy was the 35th President of the United States.",
    "The first computer was invented in the 1940s.",
    "Mount Everest is the highest mountain on Earth.",
]

# newFact = "I like to play computer games."
question = 'Who is the president of the United States?'

# Funciton to generate embeddings
def generate_embedding(text):
    embedding_request = {
        "inputText": text
    }

    response = bedrock.invoke_model(
        body=json.dumps(embedding_request),
        modelId="amazon.titan-embed-text-v1",
        accept="application/json",
        contentType="application/json"
    )

    response_body = json.loads(response.get("body").read())
    embedding = response_body.get("embedding")
    return embedding
  
factsWithEmbeddings = []

for fact in facts:
    embedding = generate_embedding(fact)
    factsWithEmbeddings.append({
        "fact": fact,
        "embedding": embedding
    })

newFactEmbedding = generate_embedding(question)

similarities = []

for fact in factsWithEmbeddings:
    similarity = cosine_similarity(newFactEmbedding, fact["embedding"])
    similarities.append({
        "fact": fact["fact"],
        "similarity": similarity
    })
    
print(f"Similarities for fact: '{question}' with:")
similarities.sort(key=lambda x: x["similarity"], reverse=True)
for similarity in similarities:
  print(f" '{similarity['fact']}': {similarity['similarity']:.2f}")
