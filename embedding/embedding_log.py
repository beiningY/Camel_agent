from vr_chunking import chunk_data_for_log
from RAG import RAG
from pathlib import Path
import time

def clean_log(log):
    clean_text = log.replace('\n', '') 
    clean_text = clean_text.replace(' ', '')
    return clean_text

def clean_log_file(file_path):
    path = Path(file_path)
    filename = path.name
    chunks = []
    with open(file_path, "r", encoding="utf-8") as f:
        title = f.readline().strip()
        log = f.read().strip()
        clean_text = clean_log(log)
        chunk = {
            "chunk_id": filename,
            "chapter": "操作日志",
            "title1": title,
            "content": clean_text,
            "type": "log"
        }
        chunks.append(chunk)
    return chunks

def embedding_log(dataname,log_path):
    rag = RAG(collection_name=dataname)
    rag.embedding(data=clean_log_file(log_path), chunk_type=chunk_data_for_log, max_tokens=500, overlap=None)

def process_all_logs():
    """处理所有日志文件"""
    vector_name = "vector_data"
    log_dir = Path("data/log")
    
    for log_file in log_dir.iterdir():
        if log_file.is_file():      
            embedding_log(vector_name, str(log_file))


def embedding_log_file(log_file):
    """处理单个日志文件"""
    vector_name = "vector_data"
    embedding_log(vector_name, log_file)    
    print(f"已处理文件: {log_file}")

start_time = time.time()
embedding_log_file("data/log/2025_06_16.txt")
end_time = time.time()
print(f"单个日志处理时间: {end_time - start_time:.4f} 秒")


