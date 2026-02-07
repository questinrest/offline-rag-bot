from pathlib import Path
from  pypdf import PdfReader
import hashlib
from langchain_core.documents import Document

def doc_id(input_path):
    return hashlib.sha256(str(input_path.resolve()).encode('utf-8')).hexdigest()[:16]


def detect_law_from_source(source):
    source = source.lower()
    if "ccpa" in source:
        return "CCPA"
    if "gdpr" in source:
        return "GDPR"
    if "ddpa" in source:
        return "DDPA"
    if "lgpd" in source:
        return "LGPD"
    return "UNKNOWN"



def load_doc(input_path):
    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist for the path: {input_path}")
    

    if path.is_file():
        files = [path]
    else:
        files = sorted(path.rglob("*"))

    docs = []

    for file_path in files:
        if not file_path.is_file():
            continue

        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            docs.extend(load_pdf(file_path))
        elif suffix == ".txt":
            text = file_path.read_text(encoding="utf-8", errors = "ignore")
            docs.append({'page_content' : text,
                         'metadata' : {
                            'doc_id' : doc_id(file_path),
                            'source' : file_path.name,
                            'page' : 1
                         }})
    return docs
    
def load_pdf(file_path):
    reader = PdfReader(str(file_path))
    document_id = doc_id(file_path)
    pages = []

    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if not text.strip():
            continue
        pages.append({'page_content' : text,
                         'metadata' : {
                            'doc_id' : doc_id(file_path),
                            'source' : file_path.name,
                            'page' : 1,
                         }})
    return pages

def add_law_metadata(file_path):
    docs = load_doc(file_path)
    for doc in docs:
        doc['metadata']['law'] = detect_law_from_source(doc["metadata"]["source"])
    return docs

def to_langchain_docs(file_path):
    lang_docs = []
    for doc in add_law_metadata(file_path):
        lang_docs.append(Document(
        page_content=doc['page_content'],
        metadata = doc['metadata']
    ))
    return lang_docs
        

        