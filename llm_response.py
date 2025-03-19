import os
os.environ['USER_AGENT'] = 'agentic_rag'
from llm_func import *
from dotenv import load_dotenv
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts.prompt import PromptTemplate

from data_processing import DataProcessing
from retriever import Retriever
from utils import *

llm = init_llm()

class SimpleRAG:
    def __init__(self):
        self.llm = llm
        self.retriever = Retriever()
        self.data_process = DataProcessing()
        self.simple_rag_prompt = PromptTemplate(
            input_variables=['context','question'],
            template="""
                You are an assistant for question-answering tasks, your name is Teddy. Use the following pieces of retrieved context to answer the question.
                If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\n

                Question: {question} \n

                Context: {context} \n

                Answer:
                """
        )
        self.chain = self.simple_rag_prompt | self.llm

    def process(self,query,top_k):
        docs = self.retriever.search(query,top_k=top_k)

        context = ""
        for i,doc in enumerate(docs):
            source = doc.metadata['source'].split('/')[1]
            page_num = doc.metadata['page']+1
            context += f'\n\n--- File {i+1} name: **{source}**, **page**: **{page_num}** ---\n\n'
            context += doc.page_content

        input_data = {
            "question":query,
            "context":doc.page_content
        }

        response = self.chain.invoke(input_data)

        response = {
            "result":response.content,
            "context":context
        }

        return response


class AgenticRAG(SimpleRAG):
    def __init__(self):
        super().__init__()
        self.relevance_chain = init_relevance_chain(llm)
        self.rewrite_chain = init_rewrite_chain(llm)
        self.simplerag = SimpleRAG()

    def simple_rag_process(self,query,top_k):
        response = self.simplerag.process(query,top_k=top_k)
        return response

    def call_relevant_chain(self,result,context):
        input_data = {
            "result":result,
            "context":context
        }
        try:
            relevant = self.relevance_chain.invoke(input_data).response.lower()
        except:
            relevant = 'relevant'
        return relevant

    def call_rewrite_chain(self,query):
        try:
            result = self.rewrite_chain.invoke({
                        'query':query
                    }).query
        except:
            result = query
        return result

    def reply_by_llm(self,query):
        response_chain = init_llm_response_chain(self.llm)
        context = "Could not find the references, answered by LLM"
        try:
            response= response_chain.invoke({
                "query":query,
            }).response
            result = {
                "result":response,
                "context": context
            }
        except:
            result = {
                "result":"Sorry, I don't know the answer yet.",
                "context": context
            }
        return result


    def process(self,query,top_k=3,max_repeat_time = 1):
        print('\n\n----- pipeline starts from here -----\n\n')
        # 1. get the SimpleRAG's result
        # 2. see if the question to the query is relevant.
        # 3. if yes, generate results, if no, rewrite query and regenerate until N times.
        response = self.simple_rag_process(query,top_k=top_k)

        relevant = self.call_relevant_chain(response['result'],response['context'])

        print(f'SIMPLE RAG: the answer is {relevant} to the context')
        print(f"SIMPLE RAG ANSWER: {response['result']}")
        REQUERY_NOTE="no need to rewrite the query." if relevant=='relevant' else ""
        print(f"\n\n--- SIMPLE RAG ANSWER is {relevant} to the context. {REQUERY_NOTE}---\n\n")
        self.simple_rag_pass= False
        if relevant == 'relevant':
            result = response
            pass_flag = True
            self.simple_rag_pass=True
        else:
            self.simple_rag_pass=False
            pass_flag=False
            repeat_time = 0
            print('will rewrite the query.')
            while repeat_time<max_repeat_time:
                print(f'old query is {query}')
                query = self.call_rewrite_chain(query)
                print(f'new query is: {query}')

                response = self.simple_rag_process(query,top_k=top_k)
                relevant = self.call_relevant_chain(response['result'],response['context'])
                if relevant == 'relevant':
                    print('\n\n --- After rewritting the query, the result and the context are relevant. ---\n\n')
                    # print(response['result'])
                    # print(response['context'])
                    pass_flag = True
                    break
                else:
                    repeat_time+=1
            result = response
        if pass_flag==False:
            response['result'] = bad_response
        return result




def final_process(query):
    # 1. route1 agentic rag
    # 2. route2 web serch
    # 3. reranking above if not work then:
    # 4. direct answer
    all_contexts = []
    all_indexs = []

    # get agentic results
    agentic_rag = AgenticRAG()
    simple_rag = SimpleRAG()
    agentic_response= agentic_rag.process(query,top_k=5)

    rag_contexts,rag_indexs = reverse_context(agentic_response['context'])
    all_contexts.extend(rag_contexts)
    all_indexs.extend(rag_indexs)

    # get web search results
    web_results = web_search(llm,query,max_results=5)

    web_contexts,web_indexs = reverse_context(web_results)
    all_contexts.extend(web_contexts)
    all_indexs.extend(web_indexs)

    # reranking
    rerank_contexts, rerank_indexs = reranking(llm,query,all_contexts,all_indexs, top_n=3)

    final_contexts = ""
    for i in range(len(rerank_contexts)):
        index = rerank_indexs[i]
        context = rerank_contexts[i]
        final_contexts += " "+index+" "
        final_contexts += context

    # get results
    input_data = {
        "question":query,
        "context":final_contexts
    }

    response = simple_rag.chain.invoke(input_data)

    response = {
        "result":response.content,
        "context":final_contexts,
        "response_type":'RAG'
    }

    # see if relevant
    relevant = agentic_rag.call_relevant_chain(response['result'],response['context'])
    print(f"\n\n--- after reranking, the result is {relevant} to the context. ---\n\n")

    if relevant=='relevant':
        final_result = response
    else:
        final_result = agentic_rag.reply_by_llm(query)
        final_result['response_type']='llm_answer'
    return final_result

if __name__=='__main__':
    # query = 'who are you?'
    query = 'what is the advantages of Kimi?'
    response = final_process(query)
    print(response)