from sentence_transformers import SentenceTransformer

model = SentenceTransformer("intfloat/multilingual-e5-large")
model.save("./saved_model/multilingual-e5-large")
