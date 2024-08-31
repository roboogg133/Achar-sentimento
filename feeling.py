from PIL import Image
import pytesseract
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os


tokenizer = AutoTokenizer.from_pretrained("alcalazans/bert-base-cased-squad-v1.1-portuguese_v1.1.9")
model = AutoModelForSequenceClassification.from_pretrained("alcalazans/bert-base-cased-squad-v1.1-portuguese_v1.1.9")

os.system('clear')
print('Bem-vindo, as respostas variam de negativo ate positivo')

tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')


 #extrai image, isso dai e bem mole     
def extract(foto):
    image = Image.open(foto)

    text = pytesseract.image_to_string(image, lang='por')


    return text 


#ver o sentimento ai
def inteligencia(texto):
    inputs = tokenizer(texto, return_tensors='pt', truncation=True, padding=True)
    outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
    feeling = torch.argmax(probabilities).item() + 1  # Avaliação de 1 a 5
    return feeling



sair = False

while sair == False:

    picture = input('\n\nFala o local ai, seu melda (ex: /home/robo/Downlaods/picture.png)\n')



    result = extract(picture)
    sentimento = inteligencia(result)
    print(f'\n    Interpretaçao do resultado\n1.Muito negativo\n2.Negativo\n3.Neutro\n4.Positivo\n5.Muito positivo\nsentimento detectado nivel: {sentimento}')

