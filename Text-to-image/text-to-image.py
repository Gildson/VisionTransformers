from diffusers import StableDiffusionPipeline
import torch
from IPython.display import display

# Carregar o modelo
model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# Gerar a imagem a partir do texto
prompt = "um lindo pôr do sol sobre as montanhas"
image = pipe(prompt).images[0]

#Mostrar a imagem
display(image)