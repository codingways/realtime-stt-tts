import os
from dotenv import load_dotenv
from openai import OpenAI
from RealtimeTTS import TextToAudioStream, GTTSEngine, GTTSVoice
from RealtimeSTT import AudioToTextRecorder

# Load environment variables from .env file
load_dotenv()

# Constants and configurations
AGENT_PROMPT = 'Act as a call center agent and help me with my issue. Speak in Spanish.'
MAX_HISTORY_LENGTH = 10

def recording_start_callback(stream_agent, recorder):
  """Callback function to stop the recorder when the agent is speaking."""
  if stream_agent.is_playing():
    stream_agent.stop()
    recorder.stop()

def initialize_clients():
  """Initialize and return the OpenAI client and TTS engines."""
  client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
  engine_agent = GTTSEngine(voice=GTTSVoice(language="es", tld="es"))
  return client, engine_agent

def generate_response(client, messages):
  """Generate a response using the OpenAI API based on given messages."""
  try:
    response = client.chat.completions.create(
      messages=messages,
      model="gpt-3.5-turbo",
    ).choices[0].message.content
  except Exception as e:
    print(f'Error generating response: {e}')
    response = "Sorry, there was an error processing your request."
  return response

def main():
  # Initial setup
  client, engine_agent = initialize_clients()
  stream_agent = TextToAudioStream(engine_agent, language="es")

  recorder = AudioToTextRecorder(
    model="medium", 
    language="es", 
    on_recording_start=lambda: recording_start_callback(stream_agent, recorder)
  )

  history = []

  while True:
    text = recorder.text()
    if text:
      print(f'Client: {text}')
      history.append({'role': 'user', 'content': text})

      # Generate agent's response
      agent_response = generate_response(client, [{'role': 'system', 'content': AGENT_PROMPT}] + history[-MAX_HISTORY_LENGTH:])
      print(f'Operator: {agent_response}')
      stream_agent.feed(agent_response).play_async()
      history.append({'role': 'assistant', 'content': agent_response})

if __name__ == '__main__':
    main()
