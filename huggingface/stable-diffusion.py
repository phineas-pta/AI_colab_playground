# -*- coding: utf-8 -*-

import torch
from diffusers import (
	StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline,
	LMSDiscreteScheduler, # sampler suggested by DreamStudio
	PNDMScheduler, # default in v1
	DDIMScheduler, # default in v2, great results at a blazing fast speed
	EulerDiscreteScheduler, EulerAncestralDiscreteScheduler, # changes generation style a lot at each step
	DPMSolverMultistepScheduler # fastest
)

model_id = "runwayml/stable-diffusion-v1-5"
pipe_text2img = StableDiffusionPipeline.from_pretrained(
	model_id, scheduler=DDIMScheduler.from_pretrained(model_id, subfolder="scheduler"),
	revision="fp16", torch_dtype=torch.float16, safety_checker=None
).to("cuda")

images_text2img = pipe_text2img(
	prompt="a photo of an astronaut riding a horse on mars",
	# negative_prompt="",
	# height=512, width=512,
	num_images_per_prompt=5,
	guidance_scale=7.5, # should be between 0 and 20
	num_inference_steps=50 # should be between 10 and 150
).images

for i in range(len(images_text2img)):
	images_text2img[i].save(f"output/img{i:02d}.jpg")

###############################################################################

from PIL import Image

pipe_img2img = StableDiffusionImg2ImgPipeline(**pipe_text2img.components).to("cuda")

my_img = Image.open("photo.png")

images_img2img = pipe_img2img(
	prompt="a photo of an astronaut riding a horse on mars",
	# negative_prompt="",
	init_image=my_img, strength=0.8,
	# height=512, width=512,
	num_images_per_prompt=5,
	guidance_scale=7.5, # should be between 0 and 20
	num_inference_steps=50 # should be between 10 and 150
).images

###############################################################################

pipe_inpaint = StableDiffusionInpaintPipeline(**pipe_text2img.components).to("cuda")

init_image = Image.open("photo.png").resize((512, 512))
mask_image = Image.open("yolo.jpeg").resize((512, 512))

images_inpaint = pipe_inpaint(
	prompt="a photo of an astronaut riding a horse on mars",
	# negative_prompt="",
	image=init_image, mask_image=mask_image,
	# height=512, width=512,
	num_images_per_prompt=5,
	guidance_scale=7.5, # should be between 0 and 20
	num_inference_steps=50 # should be between 10 and 150
).images
