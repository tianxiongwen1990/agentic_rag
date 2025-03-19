from langchain_openai import ChatOpenAI,AzureChatOpenAI
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_ollama.embeddings import OllamaEmbeddings
from openai import OpenAI
import openai
import os
from dotenv import load_dotenv
load_dotenv()


class QwenModel:
    def __init__(self,api_key,base_url):
        self.api_key = api_key
        self.base_url = base_url

    def chat(self,model_name="qwen-plus"):
        print('using qwen-plus api as llm')
        client = ChatOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            model=model_name,
        )
        return client

    def embedding(self):
        print('using qwen-plus api as embedding model')
        client = DashScopeEmbeddings(
            dashscope_api_key = self.api_key,
            model="text-embedding-v2",
        )
        return client


class AzureOpenAIModel:
    def __init__(self,api_key,base_url):
        self.api_key = api_key
        self.base_url = base_url
        self.api_version = "2023-03-15-preview"
        self.azure_deployment = "gpt-4o"

    def chat(self):
        print('using gpt-4o as llm')
        client = AzureChatOpenAI(azure_endpoint = self.base_url,
                            api_key = self.api_key,
                            api_version=self.api_version,
                            azure_deployment = self.azure_deployment
                            )
        return client
    # def embedding(self):
    #     client = openai.AzureOpenAI(
    #         api_key=self.api_key,
    #         api_version=self.api_version,
    #         azure_endpoint=self.base_url,
    #         azure_deployment=self.azure_deployment)
    #     return client


class LocalModel:
    def __init__(self,model_name):
        self.model_name = model_name

    def chat(self):
        print(f'using model {self.model_name} as llm')
        client = ChatOpenAI(
            api_key = 'teddy',
            base_url = # The string `'http://localhost:11434/v1'` is a URL representing a local server
            # endpoint. In the context of the provided code, this URL is used as the base
            # URL for a local model in the `LocalModel` class. When this URL is used, it
            # typically means that the application is trying to communicate with a service
            # or model that is running on the local machine at port 11434 and has an API
            # version of v1.
            'http://localhost:11434/v1',
            model=self.model_name,
            max_tokens=2048
        )
        return client

    def embedding(self):
        print(f'using model {self.model_name} as embedding model')
        client = OllamaEmbeddings(
        model=self.model_name,
            )
        return client


if __name__=="__main__":
    qwen_api_key = os.getenv('qwen_api_key')
    qwen_base_url = os.getenv('qwen_base_url')
    gpt_api_key = os.getenv('AZURE_OPENAI_API_KEY')
    gpt_base_url= os.getenv('AZURE_OPENAI_ENDPOINT')

    # gpt = AzureOpenAIModel(gpt_api_key,gpt_base_url)
    # client = gpt.chat()
    model_name ='llama8b_q4_K_M'
    client = LocalModel(model_name).chat()
    response = client.invoke(
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Does Azure OpenAI support customer managed keys?"},
                {"role": "assistant", "content": "Yes, customer managed keys are supported by Azure OpenAI."},
                {"role": "user", "content": "Do other Azure Cognitive Services support this too?"}
            ]
        )
    print(response)

