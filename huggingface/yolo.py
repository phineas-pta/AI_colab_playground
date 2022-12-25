#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""download all models weights to cache"""

import torch, transformers, diffusers

transformers.utils.logging.disable_progress_bar()
diffusers.utils.logging.disable_progress_bar()

yolo_trans = [
	"gpt2-xl",
	"EleutherAI/gpt-j-6B",
	"asi/gpt-fr-cased-base",
	"Cedille/fr-boris",
	"facebook/galactica-6.7b",
	"stanford-crfm/pubmedgpt",
	"KoboldAI/GPT-J-6B-Adventure",
	"KoboldAI/GPT-J-6B-Janeway",
	"KoboldAI/GPT-J-6B-Skein",
	"KoboldAI/GPT-J-6B-Shinen",
	"plguillou/t5-base-fr-sum-cnndm",
	"etalab-ia/camembert-base-squadFR-fquad-piaf",
	"VietAI/gptho",
	"VietAI/gpt-j-6B-vietnamese-news",
	"VietAI/vit5-large",
	"VietAI/vit5-large-vietnews-summarization"
]

for i in yolo_trans:
	print(i, end=": ")
	try:
		_ = transformers.pipeline(i, torch_dtype=torch.float16)
		print("good")
	except:
		print("bad")

yolo_diff = [
	"runwayml/stable-diffusion-v1-5",
	"hakurei/waifu-diffusion",
	"stabilityai/stable-diffusion-2-base", # 512×512
	"stabilityai/stable-diffusion-2", # 768×768
	# all models below don’t need `revision`
	"Langboat/Guohua-Diffusion", #  traditional Chinese paintings, token: guohua style
	"Aybeeceedee/knollingcase", # token: knollingcase
	"ProGamerGov/knollingcase-embeddings-sd-v2-0",
	"mattthew/technicolor-50s-diffusion-v2", # token: "tchnclr" and/or "1950s"
	"proxima/darkvictorian_artstyle", # token: darkvictorian artstyle
	"proxima/halloween_diffusion", # token: halloweenv1 artstyle
	"wavymulder/Analog-Diffusion", # token: "analog style"
	"nitrosocke/redshift-diffusion", # 3D artwork, token: redshift style
	"nitrosocke/redshift-diffusion-768", # SD v2
	"nitrosocke/Ghibli-Diffusion", # token: ghibli style
	"nitrosocke/mo-di-diffusion", # token: modern disney style
	"nitrosocke/classic-anim-diffusion", # token: classic disney style
	"questcoast/SD-Kurzgesagt-style-finetune", # token: "kurzgesagt style"
	"Fireman4740/kurzgesagt-style-v2-768", # SD v2
	"Guizmus/MosaicArt", # token: Mosaic Art
	"sd-dreambooth-library/fashion" # no need token
	"sd-dreambooth-library/disco-diffusion-style", # token: ddfusion style
	"Fictiverse/Stable_Diffusion_Microscopic_model", # token: Microscopic
	"Fictiverse/Stable_Diffusion_BalloonArt_Model", # token: BalloonArt
	"Fictiverse/Stable_Diffusion_PaperCut_Model", # token: PaperCut
	"plasmo/woolitize-768sd1-5", # token: woolitize
	"plasmo/colorjizz-768px" # token: colorjizz
	"plasmo/vox2", # token: voxel-ish
	"DGSpitzer/Cyberpunk-Anime-Diffusion",
	"ogkalu/Comic-Diffusion" # token: [artist] style, with artist: charliebo/holliemengert/marioalberti/pepelarraz/andreasrocha/jamesdaly
]

for i in yolo_diff:
	print(i, end=": ")
	try:
		_ = diffusers.DiffusionPipeline.from_pretrained(i, revision="fp16", torch_dtype=torch.float16)
		print("good")
	except:
		_ = diffusers.DiffusionPipeline.from_pretrained(i, torch_dtype=torch.float16)
		print("bad")
