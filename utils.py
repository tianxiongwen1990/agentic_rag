import toml
from langchain_core.pydantic_v1 import BaseModel,Field
from langchain_core.prompts.prompt import PromptTemplate
from langchain.tools import DuckDuckGoSearchResults
from duckduckgo_search import DDGS
from typing import List,Tuple
import json
import re
import os
from dotenv import load_dotenv
from llm_func import *

def load_config(config_path='config.toml'):
    return toml.load(config_path)

config = load_config()
bad_response= config['llm_response']['bad_response']
load_dotenv()

def init_llm():

    model_name = config['general']['model_name']
    if model_name =='qwen-plus':
        qwen_api_key = os.getenv('qwen_api_key')
        qwen_base_url = os.getenv('qwen_base_url')
        llm = QwenModel(qwen_api_key,qwen_base_url).chat()

    elif model_name=='gpt-4o':
        gpt_api_key = os.getenv('AZURE_OPENAI_API_KEY')
        gpt_base_url= os.getenv('AZURE_OPENAI_ENDPOINT')
        llm = AzureOpenAIModel(gpt_api_key,gpt_base_url).chat()

    elif model_name=='llama3.2:3b':
        llm = LocalModel(model_name).chat()
    elif model_name=='llama8b_q4_K_M':
        llm = LocalModel(model_name).chat()
    return llm

class LLMResponse(BaseModel):
    response: str = Field(...,
                          title="You are QA System Agent, your name is Teddy, you will answer user's query",
                          description="reply for the user's question, don't return reasons, only reply the answer, if you don't know the answer, just say you don't know, don't make up the answer.")

def init_llm_response_chain(llm):
    prompt = PromptTemplate(
        input_variables=['query'],
        template="""
        You are QA System Agent, your name is Teddy.
        Given the query '{query}', provide the answers based on your expertise, don't return the reasons, only reply the answer, don't make up the answer.
        """
    )
    reply_chain = prompt | llm.with_structured_output(LLMResponse)
    return reply_chain

class RelevanceResponse(BaseModel):
    response: str = Field(...,
                          title="Determines if context is relevant to the answer",
                          description="Output only 'Relevant' or 'Irrelevant'. ")

def init_relevance_chain(llm):
    relevance_prompt = PromptTemplate(
        input_variables=['result','context'],
        template = """
            Determine if the context and the answer are semantically relevant or not.
            Output 'Relevant' if the context and the answer are strongly semantically similar; Or 'Irrelevant' if the two are not that similar
            Output only 'Relevant' or 'Irrelevant'.
            Answer:
            {result}
            Context:
            {context}
        """
    )
    relevance_chain = relevance_prompt | llm.with_structured_output(RelevanceResponse)
    return relevance_chain

class QueryRewriterResponse(BaseModel):
    query: str = Field(...,
                       description='The query to rewrite')

def init_rewrite_chain(llm):
    rewrite_prompt = PromptTemplate(
        input_variables=['query'],
        template = """Given query '{query}' is not working well for the RAG pipeline, you need to consider the query's issue in below possibilities:
        1. Vague or Ambiguous.
        2. Too Broad or Overly Complex.
        3. Lack of Specificity

        Consider what could be the query issue and rewrite the query,
        The rewritten query must be different with the original query.

        Rewritten query:
        """
    )
    rewrite_chain = rewrite_prompt | llm.with_structured_output(QueryRewriterResponse)
    return rewrite_chain

class AccuracyScoreResponse(BaseModel):
    accuracy_score: str = Field(...,
                           description='The accuracy score, ranged from 0 to 10, of the generated answer and the ground truth answer')

def init_accuracy_chain(llm):
    accuracy_prompt = PromptTemplate(
        input_variables=['query','ground_truth_answer','generated_answer','context'],
        template = """You need to give a score to measure how accurate the generated answer is compared to the true answer.
        the score range is from 0 to 10, 0 represents that generated answer is super inaccurate and 10 stands for the score for a perfectly accurate generated answer comparing to the true answer,
        You need to consider query, context and the answers. Context information could be missing.
        query:
        {query}

        context:
        {context}

        true answer:
        {true_answer}

        generated answer:
        {generated_answer}

        The accuracy score is:
        """)
    accuracy_score_chain = accuracy_prompt | llm.with_structured_output(AccuracyScoreResponse)
    return accuracy_score_chain



def init_chain_rewrite_query_for_web_search(llm):
    prompt = PromptTemplate(
        input_variables=["query"],
        template="Rewrite the following query to make it more suitable for a web search:\n{query}\nRewritten query:"
    )
    rewrite_web_search_chain = prompt | llm.with_structured_output(QueryRewriterResponse)

    return rewrite_web_search_chain


class KnowledgeRefinementInput(BaseModel):
    key_points: str = Field(..., description="The document to extract key information from.")
def knowledge_refinement(llm,texts: str):
    prompt = PromptTemplate(
        input_variables=["texts"],
        template="""Summarize the below texts in bullet points.
        texts:
        {texts}

        bullet points:"""
    )
    chain = prompt | llm.with_structured_output(KnowledgeRefinementInput)
    input_variables = {"texts": texts}
    result = chain.invoke(input_variables).key_points
    return result

def parse_search_results(results_string: str) -> List[Tuple[str, str]]:
    """
    Parse a JSON string of search results into a list of title-link tuples.

    Args:
        results_string (str): A JSON-formatted string containing search results.

    Returns:
        List[Tuple[str, str]]: A list of tuples, where each tuple contains the title and link of a search result.
                               If parsing fails, an empty list is returned.
    """
    try:
        # Attempt to parse the JSON string
        results = json.loads(results_string)
        # Extract and return the title and link from each result
        return [(result.get('title', 'Untitled'), result.get('link', '')) for result in results]
    except json.JSONDecodeError:
        # Handle JSON decoding errors by returning an empty list
        print("Error parsing search results. Returning empty list.")
        return []

def web_search(llm,query,max_results=5):
    rewrite_web_search_chain = init_chain_rewrite_query_for_web_search(llm)
    try:
        new_query = rewrite_web_search_chain.invoke({
            'query':query
        }).query.strip()

        results = DDGS().text(new_query,region='wt-wt', max_results=5)

        web_results = []
        web_contexts = ""
        for i,result in enumerate(results):
            web_contexts += f'\n\n--- File name: **{result['title']}**, **link**: **{result['href']}** ---\n\n'
            web_contexts += result['body']

        # web_knowledge = knowledge_refinement(llm,web_results)
        # sources = parse_search_results(web_results)
    except:
        web_contexts = 'No relevant results are found from the internet.'

    return web_contexts

class RatingScore(BaseModel):
    relevance_score: float = Field(..., description="The relevance score of a document to a query.")

def reranking(llm,query,contexts,indexs, top_n=3):
    print("\n\n --- reranking the results --- \n\n")
    prompt_template = PromptTemplate(
        input_variables=["query", "context"],
        template="""On a scale of 1 to 100, rate the relevance of the following document to the query.
        Consider the specific context and intent of the query, not just keyword matches.
        Query: {query}
        Document: {context}
        Relevance Score:"""
    )
    try:
        llm_chain = prompt_template | llm.with_structured_output(RatingScore)

        scored_docs = []
        for context,index in zip(contexts,indexs):
            input_data = {"query": query, "context": context}
            try:
                score = llm_chain.invoke(input_data).relevance_score
                score = float(score)
                print('reranking score: ', score)
            except ValueError:
                score = 0  # Default score if parsing fails
            scored_docs.append((context, index, score))

        reranked_items = sorted(scored_docs, key=lambda x: x[2], reverse=True)
        reranked_contexts = [d for d,_,_ in reranked_items[:top_n]]
        reranked_indexs = [i for _,i,_ in reranked_items[:top_n]]

    except:
        reranked_contexts = contexts[:top_n]
        reranked_indexs = indexs[:top_n]

    return reranked_contexts,reranked_indexs

def pretty_print(response):
    try:
        context = response['context']
        context = re.sub('#','',context)
        final_result = ""
        final_result += "## result\n"
        final_result += response['result']
        final_result += '\n## references \n'
        final_result += context
    except:
        final_result = response
    return final_result

def reverse_context(contexts):
    pattern = r'\n\n.*?\n\n'  #

    replaced = []
    def replacer(match):
        replaced.append(match.group())
        return "['context']"
    new_text = re.sub(pattern, replacer, contexts)
    contexts_list = [c for c in new_text.split("['context']") if c != '']
    contexts_list = [c for c in contexts_list if len(set(c))>2]

    return contexts_list, replaced


def create_final_result(result):
    # `contexts_list` is a list that stores the extracted contexts from a given text. The
    # `reverse_context` function is used to extract and reverse the contexts from the input text based
    # on a specific pattern. The extracted contexts are then stored in the `contexts_list` for further
    # processing or analysis.
    # print('result')
    # print(result)
    contexts_list, replaced = reverse_context(result['context'])
    context = ""
    if len(contexts_list)==len(replaced) and len(contexts_list)>0:
        for i in range(len(contexts_list)):
            context += replaced[i]
            context += contexts_list[i]
    else:
        context = '\n'.join(contexts_list)

    response_type = result['response_type']
    if response_type=='llm_answer':
        final_result= result['result']
    elif response_type=='RAG':
        final_result = ""
        final_result += result['result']
        final_result += '\nReferences:\n'
        final_result += context
    else:
        if 'result' in result:
            final_result = result['result']
        else:
            final_result = result
    return final_result