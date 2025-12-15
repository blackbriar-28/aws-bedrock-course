import boto3
import json
import base64

from similarity import cosineSimilarity

session = boto3.Session(profile_name="bedrock")
bedrock = session.client(service_name="bedrock-runtime", region_name="us-west-2")

images = [
    'images/1.png',
    'images/2.png',
    'images/3.png',
]

def getImagesEmbedding(imagePath: str):
  with open(imagePath, "rb") as f:
    base_image = base64.b64encode(f.read()).decode("utf-8")
    
  response = bedrock.invoke_model(
      body=json.dumps({
          "inputImage": base_image
      }),
      modelId="amazon.titan-embed-image-v1",
      accept="application/json",
      contentType="application/json")
  
  response_body = json.loads(response.get("body").read())
  return response_body.get("embedding")

imagesWithEmbeddings = []

for image in images:
    embedding = getImagesEmbedding(image)
    imagesWithEmbeddings.append({
        "image": image,
        "embedding": embedding
    })
    
newImageEmbedding = getImagesEmbedding('images/cute_cartoon_cat.png')

similarities = []

for image in imagesWithEmbeddings:
    similarity = cosineSimilarity(newImageEmbedding, image["embedding"])
    similarities.append({
        "image": image["image"],
        "similarity": similarity
    })

print(f"Similarities for image: 'images/cute_cartoon_cat.jpg' with:")
similarities.sort(key=lambda x: x["similarity"], reverse=True)
for similarity in similarities:
  print(f" '{similarity['image']}': {similarity['similarity']:.2f}")