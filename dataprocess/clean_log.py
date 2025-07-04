from pathlib import Path
import json

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


if __name__ == "__main__":
    chunks = []
    OUTPUT_JSON = "../data/cleand_data/data_json_log.json"
    log_dir = Path("../data/raw_data/log")
    for log_file in log_dir.iterdir():
        if log_file.is_file():      
            chunk = clean_log_file(log_file)
            chunks.extend(chunk)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"\n共提取 {len(chunks)} 个有效内容块，已保存至 {OUTPUT_JSON}")
