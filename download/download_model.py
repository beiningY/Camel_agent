from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-large-zh-v1.5")
model.save("models/bge-large-zh-v1.5")
