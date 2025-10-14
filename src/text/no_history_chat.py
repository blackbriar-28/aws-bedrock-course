import boto3
import json

session = boto3.Session(profile_name="bedrock")
client = session.client(service_name="bedrock-runtime", region_name="us-west-2")

def get_configuration(prompt: str):
  body = json.dumps({
      "inputText": prompt,
      "textGenerationConfig": {
          "maxTokenCount": 4096,
          "stopSequences": [],
          "temperature": 0,
          "topP": 1
      }
  })
  return body

print("Bot: Hello! I am a chatbot. I can help you with anything related to Movies. How can I assist you today?")

while True:
  user_input = input("You: ")
  if user_input.lower() == "exit":
      print("Bot: Goodbye! Have a great day!")
      break

  body = get_configuration(user_input)

  titan_model_id = "amazon.titan-text-express-v1"

  accept = "application/json"
  content_type = "application/json"

  response = client.invoke_model(
      body=body, modelId=titan_model_id, accept=accept, contentType=content_type
  )
  response_body = json.loads(response.get("body").read())
  ai_response = response_body.get("generatedText", "").strip()
  print(f"Bot: {ai_response}")
  



            
            
  
