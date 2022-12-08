from transformers import DetrFeatureExtractor, DetrForObjectDetection
import torch
from PIL import Image, ImageDraw, ImageFont
import requests

url = "https://www.paradisebirmingham.co.uk/wp-content/uploads/2022/01/ARG003_Paradise_N8238_press-2048x1386.jpg"
image = Image.open(requests.get(url, stream=True).raw)

contents = []

feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

inputs = feature_extractor(images=image, return_tensors="pt")
outputs = model(**inputs)

target_sizes = torch.tensor([image.size[::-1]])
results = feature_extractor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    
    if score > 0.9:
        item =  [model.config.id2label[label.item()],box,round(score.item(),3)]
        contents.append(item)
        
font2 = ImageFont.truetype(r'C:\Users\System-Pc\Desktop\arial.ttf', 50) 
font = ImageFont.truetype(r'C:\Users\System-Pc\Desktop\arial.ttf', 20) 

for i in range(len(contents)-1):


    img = ImageDraw.Draw(image)
    img.rectangle(contents[i][1],outline="#ffff33")
    
    img.text((contents[i][1][0] + ((contents[i][1][2]+contents[i][1][0])/2),contents[i][1][1]+2) ,contents[i][0], fill="green", font=font2)

    img.text((contents[i][1][2]+ 2,contents[i][1][3]- 20), str(contents[i][2]), fill="green",font=font)


image.show()


