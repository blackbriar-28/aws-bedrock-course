import boto3
import json

session = boto3.Session(profile_name="bedrock")
client = session.client(service_name="bedrock-runtime", region_name="us-west-2")

prompt =   """
  Below is a phone conversation between a couple, Jeff and Martha.

  Jeff: Hey babe, how’s your day going?
  Martha: Hey! It’s been okay. Super busy though. You?
  Jeff: Same. Meetings nonstop since 9 AM. I swear my brain’s fried.
  Martha: Haha, I feel that. Did you have lunch at least?
  Jeff: Just grabbed a sandwich between calls. You?
  Martha: I made pasta. Tried that recipe I sent you last week. Turned out actually good
  Jeff: Oh nice! You should’ve saved me some.
  Martha: I would’ve, but someone never said what time they were getting home
  Jeff: I told you! I said around 7.
  Martha: No, you said maybe 7. Last time that meant 9.
  Jeff: That was one time!
  Martha: Two times, actually.
  Jeff: Okay, fine, twice. Still, you know how work is lately.
  Martha: Yeah, but you could at least text if you’re running late. I end up waiting and then eating alone.
  Jeff: You’re right, sorry. I’ll do better next time.
  Martha: Okay… thanks. Just don’t make me feel like I’m eating for two when it’s just me
  Jeff: Haha, noted. Speaking of which—what are we watching tonight?
  Martha: I was thinking that new thriller on Netflix.
  Jeff: Ugh, not another thriller. Can we watch something fun for once?
  Martha: You always say that! You pick some dumb comedy and I sit there pretending to laugh.
  Jeff: Dumb comedy?? At least I don’t pick movies where everyone dies in the first 10 minutes!
  Martha: Maybe if you actually paid attention, you’d realize those are good.
  Jeff: Yeah, “good” as in nightmare-fuel before bed.
  Martha: Fine, then watch your comedy alone.
  Jeff: Maybe I will.
  Martha: Great.
  Jeff: Great.

  From the call transcript above, generate a concise summary of the main points discussed in the conversation.
"""

titan_model_id = "amazon.titan-text-express-v1"

accept = "application/json"
content_type = "application/json"

body = json.dumps({
    "inputText": prompt,
    "textGenerationConfig": {
        "maxTokenCount": 2048,
        "stopSequences": [],
        "temperature": 0,
        "topP": 0.9
    }
})

response = client.invoke_model(
    body=body, modelId=titan_model_id, accept=accept, contentType=content_type
)
response_body = json.loads(response.get("body").read())
print(f"Input token count: {response_body['inputTextTokenCount']}")

for result in response_body['results']:
    print(f"Token count: {result['tokenCount']}")
    print(f"Output text: {result['outputText']}")
    print(f"Completion reason: {result['completionReason']}")
