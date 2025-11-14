import boto3
import json
import base64

session = boto3.Session(profile_name="bedrock")
bedrock = session.client(service_name="bedrock-runtime", region_name="us-west-2")

stability_image_config = json.dumps({
  "prompt": "A photo of a cute cat on a space suit"
})

response = bedrock.invoke_model(
    body=stability_image_config,
    modelId="stability.stable-image-core-v1:1",
    accept="application/json",
    contentType="application/json"
)

response_body = json.loads(response.get("body").read())
base64_image = response_body.get("images")[0]

base_64_image = base64.b64decode(base64_image)

file_path = "cute_cat.png"
with open(file_path, "wb") as f:
    f.write(base_64_image)