import json
from vr_chunking import chunk_data_by_title
from camel.embeddings import SentenceTransformerEncoder
from camel.storages import QdrantStorage
from camel.retrievers import VectorRetriever
from sentence_transformers import SentenceTransformer
import time

class RAG:
    def __init__(self, collection_name=None):
        self.load_config()
        self.init_models()
        if collection_name is None:
            collection_name = [self.config.get("collection_name")]
        self.collection_name = collection_name
        self.init_vector_store()

    def load_config(self):
        with open("config.json", "r", encoding="utf-8") as f:
            self.config = json.load(f)

    def init_models(self):
        #self.embedding_model = SentenceTransformerEncoder(model_name=self.config.get("embedding_model", "intfloat/multilingual-e5-large"))  
        #self.tokenizer = AutoTokenizer.from_pretrained(self.config.get("tokenizer_model", "intfloat/multilingual-e5-large"))
        self.embedding_model = SentenceTransformerEncoder(model_name="./saved_model/multilingual-e5-large")
        self.tokenizer = SentenceTransformer("./saved_model/multilingual-e5-large")

    def init_vector_store(self):
        """初始化向量存储"""
        self.vector_storage = QdrantStorage(
            vector_dim=self.embedding_model.get_output_dim(),
            path="local_data",  
            collection_name=self.collection_name,
        )
        self.vr = VectorRetriever(embedding_model=self.embedding_model, storage=self.vector_storage)

    def embedding(self, data_path = None, data = None, chunk_type = chunk_data_by_title, max_tokens = 500, overlap = None):
        """向量化"""
        if data is None:
            with open(data_path, "r") as f:
                structured_data = json.load(f)
        else:
            structured_data = data
        chunks = chunk_type(
            structured_data,
            MAX_TOKENS=max_tokens,
            tokenizer=self.tokenizer,
        )
        # start_time = time.time()
        for chunk in chunks:
            self.vr.process(
                content=chunk["content"],
                should_chunk=False,
                extra_info={"id": chunk["chunk_id"], "title": chunk["title"], "type": chunk["type"]}
            )
        # end_time = time.time()
        # print(f"embedding处理时间: {end_time - start_time:.4f} 秒")

    def rag_retrieve(self, query, topk=None):
        """进行检索"""
        results = self.vr.query(
            query=query, 
            top_k=topk if topk is not None else self.config.get("vector_top_k", 5), 
            similarity_threshold=self.config.get("similarity_threshold", 0.6)
            )   
        retrieved = []
        for i, info in enumerate(results):
            retrieved.append(f"{i+1}. {info['text']}\n\n")  
        return retrieved          
    

    