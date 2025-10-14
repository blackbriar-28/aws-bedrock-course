import boto3
import pprint

session = boto3.Session(profile_name="bedrock")
bedrock = session.client(service_name="bedrock", region_name="us-west-2")

pp = pprint.PrettyPrinter(indent=4).pprint

def list_foundation_models():
  models = bedrock.list_foundation_models()
  for model in models['modelSummaries']:
      pp(model)
      pp("----------------------------")
      
def get_foundation_model(modelIdentifier):
  model = bedrock.get_foundation_model(modelIdentifier=modelIdentifier)
  pp(model)      

# list_foundation_models()
get_foundation_model(modelIdentifier='mistral.mistral-large-2402-v1:0')