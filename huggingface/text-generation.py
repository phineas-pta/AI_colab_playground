# -*- coding: utf-8 -*-

from transformers import pipeline
import torch

model = pipeline(device="cuda:0", task="text-generation", model="VietAI/gptho")
input_txt = "<|endoftext|> thu sang "
output_txt = model(input_txt, max_new_tokens=1000)[0]["generated_text"]
print(output_txt)

model = pipeline(device="cuda:0", task="text-generation", model="asi/gpt-fr-cased-base")
input_txt = "Longtemps je me suis couché de bonne heure."
output_txt = model(input_txt, max_new_tokens=1000)[0]["generated_text"]
print(output_txt)

# at least 8GB VRAM
model = pipeline(device="cuda:0", task="text-generation", model="gpt2-xl", torch_dtype=torch.float16)
input_txt = "Replace me by any text you’d like."
output_txt = model(input_txt, max_new_tokens=1000)[0]["generated_text"]
print(output_txt)

# at least 12GB VRAM
model = pipeline(device="cuda:0", task="text-generation", model="VietAI/gpt-j-6B-vietnamese-news", torch_dtype=torch.float16)
input_txt = "Tiềm năng của trí tuệ nhân tạo"
output_txt = model(input_txt, max_new_tokens=1000)[0]["generated_text"]
print(output_txt)

# at least 24GB VRAM
model = pipeline(device="cuda:0", task="text-generation", model="Cedille/fr-boris", torch_dtype=torch.float16)
input_txt = "Il était une fois"
output_txt = model(input_txt, max_new_tokens=1000)[0]["generated_text"]
print(output_txt)

# at least 24GB VRAM
model = pipeline(device="cuda:0", task="text-generation", model="EleutherAI/gpt-j-6B", torch_dtype=torch.float16)
input_txt = "Once upon a time,"
output_txt = model(input_txt, max_new_tokens=1000)[0]["generated_text"]
print(output_txt)

# at least 24GB VRAM
model = pipeline(device="cuda:0", task="text-generation", model="facebook/galactica-6.7b", torch_dtype=torch.float16)
input_txt = "Title: Impact on office dust on sneezing in office workers\n\nAbstract:"
output_txt = model(
	input_txt, max_new_tokens=1000,
	top_k=50, # how many top tokens does the model use
	do_sample=True, # to avoid generating same things over and over again
	temperature=0.5 # randomness in words
)[0]["generated_text"]
print(output_txt)

###################
# A.I. dungeon adventure with KoboldAI

# "KoboldAI/GPT-J-6B-Adventure" # default model with dungeon
# "KoboldAI/GPT-J-6B-Janeway"   # novel literature
# "KoboldAI/GPT-J-6B-Skein"     # adventure + novel
# "KoboldAI/GPT-J-6B-Shinen"    # explicit NSFW
