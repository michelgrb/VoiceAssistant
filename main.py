import os
import openai
import wandb

#Login to OpenAI
with open('OPENAI_API_KEY.txt') as f:
    openai.api_key = f.readline()
f.close()

#Run wandb
run = wandb.init(project='GPT-3 in Python')
prediction_table = wandb.Table(columns=["prompt", "completion"])

#prompt prediction
gpt_prompt = "Correct this to standard English:\n\nShe no went to the market."

response = openai.Completion.create(
  engine="text-davinci-002",
  prompt=gpt_prompt,
  temperature=0.5,
  max_tokens=256,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0
)

print(response['choices'][0]['text'])

prediction_table.add_data(gpt_prompt,response['choices'][0]['text'])
wandb.log({'predictions': prediction_table})
wandb.finish()
