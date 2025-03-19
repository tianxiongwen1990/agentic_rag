import json
import pandas as pd
import random
from langchain_core.pydantic_v1 import BaseModel,Field
from langchain_core.prompts.prompt import PromptTemplate
from llm_func import *
import os

# data = []
# with open("distillation/input_data/data_with_llm_results_0213.json",'r') as f:
#     for line in f.readlines():
#         data.append(json.loads(line))

# data2 = []
# with open("distillation/input_data/databricks-dolly-15k.json",'r') as f:
#     for line in f.readlines():
#         data2.append(json.loads(line))

# # 杨路的拿2000条
# # 我的数据拿1000条
# # 再加点技术数据。 ？
# # 先用Llama-3.1-8B-Instruct的模型表述过一遍，再蒸馏1.5B，为了保持风格的一致。
# # 再训练role-play
# # 做评估 evalscope

# # 88.4% general
# # 3.3% grammer
# # 3.3% science
# # 3.3% math
# # 1.6% coding

# # general       2630
# # grammar       100
# # science       100
# # math          100
# # PythonCode     50

# final_data1 = random.sample(data,1000)
# final_data2 = random.sample(data2,2000)

# final_data = []
# final_data+=final_data1
# final_data+=final_data2
# The code snippet you provided is writing the contents of the `final_data` list to a JSON file named
# "distill_3k_raw.json" in the "distillation/input_data" directory. The data is being dumped into the
# file with an indentation level of 2 for better readability (`indent=2`) and ensuring that non-ASCII
# characters are written as-is without escaping (`ensure_ascii=False`).

# with open("distillation/input_data/distill_3k_raw.json", "w", encoding="utf-8") as f:
#     json.dump(final_data, f, indent=2, ensure_ascii=False)

qwen_api_key = os.getenv('qwen_api_key')
qwen_base_url = os.getenv('qwen_base_url')
gpt_api_key = os.getenv('AZURE_OPENAI_API_KEY')
gpt_base_url= os.getenv('AZURE_OPENAI_ENDPOINT')


model = LocalModel('llama3.1') # 7B
# model = LocalModel('qwen2.5:0.5b')
llm = model.chat()

# llm = QwenModel(qwen_api_key,qwen_base_url).chat()


class StandardFormat(BaseModel):
    response: str = Field(...,
                          title="Standardize the output",
                          description="standardize the output with technique style")

def init_standard_format_chain(llm):
    prompt = PromptTemplate(
        input_variables=['query'],
        template="""
            Standardize the output with more llama style and technique style:
            Requirements:
            1. Simplify redundant terms (e.g., The whole → The full)
            2. Use technical terms like sequence/completes
            3. Prefer roughly over exactly
            4. Preserve contextual terms like process → sequence
        Given the query '{query}', provide the answers based on your expertise, don't return the reasons, only reply the answer, don't make up the answer.

        for example, input sentence:
        "The entire process takes around 10 seconds."

        output sentence:
        "The system initialization phase will be completed in a time frame of ±10 seconds."
        """
    )
    reply_chain = prompt | llm.with_structured_output(StandardFormat)
    return reply_chain

chain = init_standard_format_chain(llm)

from tqdm import tqdm
with open("distillation/input_data/distill_3k_raw_clean.json",'r') as f:
    final_data = json.load(f)


i=0
for data in tqdm(final_data):

    i+=1
    input_data = {
        "query":data['output']
    }
    if data['output_clean']=="":
        print("working on %s"%i)
        try:
            output = chain.invoke(input_data)
            data['output_clean'] = output.response
        except:
            print('erorr in %s'%i)
            pass
    else:
        pass

    if i%10==0:
        print('saving in %d'%i)
        with open("distillation/input_data/distill_3k_raw_clean.json", "w", encoding="utf-8") as f:
            json.dump(final_data, f, indent=2, ensure_ascii=False)
print('done')