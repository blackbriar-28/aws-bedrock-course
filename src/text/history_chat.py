import boto3
import json

session = boto3.Session(profile_name="bedrock")
client = session.client(service_name="bedrock-runtime", region_name="us-west-2")

history = []

def get_history():
    return "\n".join(history)

def get_configuration():
  body = json.dumps({
      "inputText": get_history(),
      "textGenerationConfig": {
          "maxTokenCount": 2048,
          "stopSequences": [],
          "temperature": 0,
          "topP": 1
      }
  })
  return body

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Bot: Goodbye! Have a great day!")
        break
    history.append(f"You: {user_input}")
    body = get_configuration()

    titan_model_id = "amazon.titan-text-express-v1"
    accept = "application/json"
    content_type = "application/json"

    response = client.invoke_model(
        body=body, modelId=titan_model_id, accept=accept, contentType=content_type
    )
    response_body = json.loads(response.get("body").read())
    cleaned_response = response_body['results'][0]['outputText'].replace('\n', ' ')
    print(f"Bot: {cleaned_response}")
    history.append(cleaned_response)



            
            
  
