import pdfplumber
import re
import json

PDF_PATH = "循环水南美白对虾养殖系统设计及操作手册张驰v3.0.pdf"
OUTPUT_JSON = "structured_data_zh.json"

chapter_pattern = re.compile(r"^第\d+章")
title1_pattern = re.compile(r"^\d+\.\d+\s?")
title2_pattern = re.compile(r"^\d+\.\d+\.\d+\s?")

def extract_pdf_text(pdf_path, start_page=7, end_page=61):
    with pdfplumber.open(pdf_path) as pdf:
        # 合并所有页面的文本
        all_text = []
        for i in range(start_page-1, end_page):
            page = pdf.pages[i]
            text = page.extract_text() or ""
            all_text.append(text)
        
        # 将所有页面文本合并，并按段落分割
        full_text = "\n".join(all_text)
        # 将连续的多个换行符替换为两个换行符，确保段落间只有一个分隔符
        full_text = re.sub(r'\n\s*\n+', '\n\n', full_text)
        # 按双换行符分割成段落
        paragraphs_raw = [p.strip() for p in full_text.split('\n\n') if p.strip()]
        print(paragraphs_raw)
        # 处理文本内容
        paragraphs = []
        current_paragraph = None
        chunk_id = 1

        def save_current_paragraph():
            nonlocal chunk_id, current_paragraph
            if current_paragraph and current_paragraph.get("content"):
                current_paragraph["chunk_id"] = chunk_id
                paragraphs.append(current_paragraph.copy())
                chunk_id += 1

        # 逐段分析内容
        for para in paragraphs_raw:
            # 获取段落的第一行作为可能的标题
            first_line = para.split('\n')[0].strip()
            
            if chapter_pattern.match(first_line):
                save_current_paragraph()
                current_paragraph = {
                    "chapter": first_line,
                    "content": "\n".join(para.split('\n')[1:]).strip()
                }
            elif title1_pattern.match(first_line) and current_paragraph:
                save_current_paragraph()
                current_paragraph = {
                    "chapter": current_paragraph.get("chapter", ""),
                    "title1": first_line,
                    "content": "\n".join(para.split('\n')[1:]).strip()
                }
            elif title2_pattern.match(first_line) and current_paragraph:
                save_current_paragraph()
                current_paragraph = {
                    "chapter": current_paragraph.get("chapter", ""),
                    "title1": current_paragraph.get("title1", ""),
                    "title2": first_line,
                    "content": "\n".join(para.split('\n')[1:]).strip()
                }
            elif current_paragraph:
                if current_paragraph["content"]:
                    current_paragraph["content"] += "\n\n"
                current_paragraph["content"] += para

        # 保存最后一个段落
        save_current_paragraph()
        return paragraphs

if __name__ == "__main__":
    paragraphs = extract_pdf_text(PDF_PATH)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(paragraphs, f, ensure_ascii=False, indent=2)
    print(f"\n共提取 {len(paragraphs)} 个有效内容块，已保存至 {OUTPUT_JSON}")

