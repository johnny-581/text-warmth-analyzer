from transformers import pipeline
pipeline = pipeline("text-classification", model="nlptown/bert-base-multilingual-uncased-sentiment")
data = ["let me do that", "where is this?"] 
result = pipeline(data)
print(result)