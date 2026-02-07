from langchain_text_splitters import RecursiveCharacterTextSplitter




def chunker(lang_docs, chunk_size = 350, chunk_overlap = 50):
    splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", r"(?<=[.?!])\s+"],
    chunk_size = chunk_size,
    chunk_overlap = chunk_overlap)

    return splitter.split_documents(lang_docs)