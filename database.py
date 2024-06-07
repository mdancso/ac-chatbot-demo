import io
import streamlit as st
from PyPDF2 import PdfReader
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings


@st.cache_resource
def load_db():
    return VectorDB("archicad_db")

class VectorDB:
    def __init__(self, db_name: str):
        self.embeddings = OpenAIEmbeddings()
        self.db_name = db_name
        self.db = self.load_db(db_name)

    def as_retriever(self, k: int):
        return self.db.as_retriever(search_kwargs={'k': k})
    
    def add_pdf_to_db(self, id: str, bytes_file: bytes):
        reader = PdfReader(io.BytesIO(bytes_file), strict=False)
        documents = [
            Document(page_content=page.extract_text(), metadata={'id': id, 'source': id, 'page': i})
            for i, page in enumerate(reader.pages, start=1)
        ]

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            separators=["\n\n", "\n", "(?<=\. )", " ", ""]
        )
        docs = text_splitter.split_documents(documents)

        if len(docs) > 0:       
            db2 = FAISS.from_documents(docs, self.embeddings)
            self.db.merge_from(db2)

    def delete_file_from_db(self, id):
        chunks_to_remove = [k for k, doc in self.db.docstore._dict.items() if doc.metadata.get('id') == id]
        self.db.delete(chunks_to_remove)

    def get_known_documents(self):
        db_dict = self.db.docstore._dict
        return list(set(
            doc.metadata.get('id')
            for doc in db_dict.values() if doc.metadata.get('id') is not None
        ))

    def save_db(self):
        self.db.save_local(self.db_name)

    def load_db(self, name):
        try:
            return FAISS.load_local(name, self.embeddings)
        except Exception as _:
            return FAISS.load_local(name, self.embeddings, allow_dangerous_deserialization=True)
        