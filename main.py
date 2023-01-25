import os
import openai
import wandb
import speech_recognition as sr
from pydub import AudioSegment
# from pydub.effects import pitch_shift
import pyttsx3
from gtts import gTTS


r = sr.Recognizer()

with sr.Microphone() as source:
    print("Say something!")
    audio = r.listen(source)

try:
    print("You said: " + r.recognize_google(audio))
except sr.UnknownValueError:
    print("Google Speech Recognition could not understand audio")
except sr.RequestError as e:
    print("Could not request results from Google Speech Recognition service; {0}".format(e))

#Login to OpenAI
with open('OPENAI_API_KEY.txt') as f:
    openai.api_key = f.readline()
f.close()

#Run wandb
# run = wandb.init(project='GPT-3 in Python')
# prediction_table = wandb.Table(columns=["prompt", "completion"])

#prompt prediction
gpt_prompt = r.recognize_google(audio)

response = openai.Completion.create(
  engine="text-davinci-002",
  prompt=gpt_prompt,
  temperature=0.5,
  max_tokens=1024,
  n = 1,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0
)


print(response["choices"][0]["text"])
text = response["choices"][0]["text"]
tts = gTTS(text, lang='de', slow=False)
tts.save("hello.mp3")

# audio = AudioSegment.from_file("original.mp3")
# shifted_audio = pitch_shift(audio, n_semitones=2)
# shifted_audio.export("pitch_shifted.mp3", format="mp3")

os.system("start hello.mp3")
# prediction_table.add_data(gpt_prompt,response['choices'][0]['text'])
# wandb.log({'predictions': prediction_table})
# wandb.finish()
