from llm_response import final_process
from utils import create_final_result

def flow(query):
    result = final_process(query)
    final_result = create_final_result(result)
    return final_result
