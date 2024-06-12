from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image

# If want use a Google files
from google.colab import files
files.upload()

# image preprocessing
img = Image.open('images.jpg')
img = img.resize((224, 224))

# object instances
model_name = 'google/vit-base-patch16-224'
# Download the the processor
processor = ViTImageProcessor.from_pretrained(model_name)
# Download of the model
model = ViTForImageClassification.from_pretrained(model_name)

# processing the image
inputs = processor(images=img, return_tensors="pt")
# passes the input through the model
outputs = model(**inputs)
# Get the array with the probabilities
logits = outputs.logits
# model predicts one of the 1000 ImageNet classes
# Get the index with the highest probability value
predicted_class_idx = logits.argmax(-1).item()
# Get the class corresponding to index
class_ = model.config.id2label[predicted_class_idx]

print("Predicted class:", class_)