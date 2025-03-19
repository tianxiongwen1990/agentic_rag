
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils import load_config
from retriever import Retriever
import os

config = load_config()
chunk_size = config['data_processing']['chunk_size']
chunk_overlap = config['data_processing']['chunk_overlap']

class DataProcessing:
    def __init__(self):
        # split documents into chunks
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap,
            length_function=len
        )
    def process_document(self,document_path='input_docs/'):
        if os.path.isdir(document_path):
            docs = [os.path.join(document_path,f) for f in os.listdir(document_path) if f.endswith('.pdf')]
            documents = []
            for doc in docs:
                loader = PyPDFLoader(doc)
                document = loader.load()
                documents += document
        elif os.path.isfile(document_path) and document_path.endswith('.pdf'):
            loader = PyPDFLoader(doc)
            documents = loader.load()

        ### TODO: add more metadata by using LLM sentiment.
        documents = self.text_splitter.split_documents(documents)

        return documents

if __name__=="__main__":
    data_process= DataProcessing()
    documents = data_process.process_document(document_path='input_docs/')
    retriever = Retriever()
    if retriever.need_upload:
        print('uploading index')
        retriever.upload_index(documents)
    response = retriever.search('hi')
    print(response)