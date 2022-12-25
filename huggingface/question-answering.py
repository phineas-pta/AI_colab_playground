# -*- coding: utf-8 -*-

from transformers import pipeline

model = pipeline(device="cuda:0", task="question-answering", model="etalab-ia/camembert-base-squadFR-fquad-piaf")
input_txt = {
	"question": "Qui est Claude Monet?",
	"context": "Claude Monet, né le 14 novembre 1840 à Paris et mort le 5 décembre 1926 à Giverny, est un peintre français et l’un des fondateurs de l'impressionnisme."
}
output_txt = model(input_txt)["answer"]
print(output_txt)

model = pipeline(device="cuda:0", task="question-answering", model="VietAI/vit5-large")
input_txt = {
	"question": "??",
	"context": "??"
}
output_txt = model(input_txt)["answer"]
print(output_txt)
