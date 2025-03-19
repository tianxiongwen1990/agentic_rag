# https://python.langchain.com/docs/integrations/vectorstores/faiss/

from uuid import uuid4
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from llm_func import *

from utils import load_config
from dotenv import load_dotenv

import os
import numpy as np
load_dotenv()

qwen_api_key = os.getenv('qwen_api_key')
qwen_base_url = os.getenv('qwen_base_url')
embedding_model = QwenModel(qwen_api_key,qwen_base_url).embedding()
dimension = len(embedding_model.embed_query('hi'))

class Retriever:
    def __init__(self,index_name='faiss',index_dir = 'vec_db',db_name='faiss.db'):
        self.embedding_model = embedding_model
        self.index_name = index_name
        self.index_dir = index_dir
        self.db_name = db_name
        self.index_path = os.path.join(index_dir,self.index_name)
        self.db_path = os.path.join(index_dir,self.db_name)

        # read existed db file from local
        if os.path.isdir(self.db_path):
            print('loading index and vector db.')
            self.index = faiss.read_index(self.index_path)
            self.vector_store = FAISS.load_local(folder_path=self.db_path,
                                                 embeddings = self.embedding_model,
                                                 index_name = self.index_name,
                                                 allow_dangerous_deserialization=True)
            self.need_upload = 0
        else:
            print('creating index and vector db.')
            self.index = faiss.IndexFlatL2(dimension)
            self.vector_store = FAISS(
                embedding_function=self.embedding_model,
                index=self.index,
                docstore = InMemoryDocstore(),
                index_to_docstore_id={},
            )
            self.need_upload = 1

    def upload_index(self,documents):
        uuids = [str(uuid4()) for _ in range(len(documents))]
        self.vector_store.add_documents(documents=documents, ids=uuids)

        self.vector_store.save_local(folder_path=self.db_path,
                                         index_name=self.index_name)
        faiss.write_index(self.index,self.index_path)

    def search(self,query,top_k = 3):
        results = self.vector_store.similarity_search(
            query,
            k=top_k,
            # filter={"source": " tweet"},
        )
        return results

