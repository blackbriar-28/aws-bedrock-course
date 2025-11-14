import boto3
import json
import base64

session = boto3.Session(profile_name="bedrock")
bedrock = session.client(service_name="bedrock-runtime", region_name="us-west-2")

stability_image_config = json.dumps(
{
    "taskType": "TEXT_IMAGE",
    "textToImageParams": {
        "text": "a photo of a beautiful banshee in a forest",      
        "negativeText": "Clouds"
    },
    "imageGenerationConfig": {
        "quality": "standard",
        "numberOfImages": 1,
        "height": 512,
        "width": 512,
        "cfgScale": 8.0
    }
}
)

response = bedrock.invoke_model(
  body=stability_image_config,
  modelId="amazon.titan-image-generator-v2:0",
  accept="application/json",
  contentType="application/json"
)

response_body = json.loads(response.get("body").read())
base64_image = response_body.get("images")[0]

base_64_image = base64.b64decode(base64_image)

file_path = "banshee_dark_forest.png"
with open(file_path, "wb") as f:
  f.write(base_64_image)