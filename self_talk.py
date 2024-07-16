from RealtimeTTS import TextToAudioStream, GTTSEngine, GTTSVoice
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Constants and configurations
AGENT_PROMPT = 'Act as a call center agent and help me with my issue. Speak in Spanish.'
CALLER_PROMPT = 'Act as if you are calling a customer service line to seek help with an imaginary issue. Speak in Spanish.'
MAX_HISTORY_LENGTH = 10

def main():
  client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

  engine_agent = GTTSEngine(voice=GTTSVoice(language="es", tld="es"))
  engine_caller = GTTSEngine(voice=GTTSVoice(language="es", tld="com"))
  stream_agent = TextToAudioStream(engine_agent, language="es")
  stream_caller = TextToAudioStream(engine_caller, language="es")

  history_agent = []
  history_caller = []

  while True:
    agent_response = generate_response(client, messages=[{'role': 'system', 'content': AGENT_PROMPT}] + history_agent[-MAX_HISTORY_LENGTH:])
    print(f'Operator: {agent_response}')
    stream_agent.feed(agent_response).play()
    update_history(history_agent, 'assistant', agent_response)
    update_history(history_caller, 'user', agent_response)

    caller_response = generate_response(client, messages=[{'role': 'system', 'content': CALLER_PROMPT}] + history_caller[-MAX_HISTORY_LENGTH:])
    print(f'Client: {caller_response}')
    stream_caller.feed(caller_response).play()
    update_history(history_caller, 'assistant', caller_response)
    update_history(history_agent, 'user', caller_response)

def generate_response(client, messages):
  """
  Generate a response using OpenAI API based on given messages.
  """
  try:
    response = client.chat.completions.create(
      messages=messages,
      model="gpt-3.5-turbo",
    ).choices[0].message.content
  except Exception as e:
    print(f'Error generating response: {e}')
    response = "Perdón, no pude entender tu solicitud. ¿Podrías repetirlo?"
  return response

def update_history(history, role, content):
  """
  Update history with a new message.
  """
  history.append({'role': role, 'content': content})

if __name__ == '__main__':
    main()
