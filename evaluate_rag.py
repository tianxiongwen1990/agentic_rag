from tqdm import tqdm
import pandas as pd
from utils import *
from llm_func import *
from llm_response import *
from langchain_core.prompts.prompt import PromptTemplate
from dotenv import load_dotenv
from evalscope.run import run_task
from evalscope.utils.logger import get_logger


load_dotenv()
config = load_config()
llm = init_llm()
qwen_api_key = os.getenv('qwen_api_key')
qwen_base_url = os.getenv('qwen_base_url')

num_of_data_to_eval = 30
data = pd.read_excel('qa_list_with_answer.xlsx')

def generate_answers():
    print('generating answers..')
    for i in tqdm(range(len(data))[:num_of_data_to_eval]):
        question = data.loc[i,'questions']
        print('question')
        print(question)
        llm_answer = final_process(question)
        answer = llm_answer['result']
        context = llm_answer['context']
        data.loc[i,tag+'_llm_answer']= answer
        data.loc[i,tag+'_llm_context'] = context
    data.to_excel('qa_list_with_answer.xlsx',index=False)

# def evaluation_bak():
#     qwen_api_key = os.getenv('qwen_api_key')
#     qwen_base_url = os.getenv('qwen_base_url')
#     eval_llm = QwenModel(qwen_api_key,qwen_base_url).chat()
#     print('evaluating...')
#     data = pd.read_excel('qa_list_with_answer.xlsx')
#     for i in tqdm(range(len(data))[:num_of_data_to_eval]):
#         question = data.loc[i,'questions']
#         true_answer = data.loc[i,'answers']
#         llm_answer = data.loc[i,'llm_answer']
#         llm_context = data.loc[i,'llm_context']

#         input_data = {
#             "query":question,
#             "context":llm_context,
#             "true_answer":true_answer,
#             "generated_answer":llm_answer
#         }

#         accuracy_score_chain = init_accuracy_chain(eval_llm)

#         pass_flag = 0
#         repeat_time = 0
#         while repeat_time < 3 and pass_flag==False:
#             print('trying %d time'%repeat_time)
#             try:
#                 response = accuracy_score_chain.invoke(input_data)
#                 print(response)
#                 accuracy_score = response.accuracy_score
#                 accuracy_score = float(accuracy_score)
#                 pass_flag=1
#             except:
#                 accuracy_score = 0
#             repeat_time+=1
#         data.loc[i,'accuracy_score'] = accuracy_score
#     data.to_excel('qa_list_with_score.xlsx',index=False)

#     print(data['accuracy_score'][:num_of_data_to_eval].mean())


def prepare_eval_data(tag='llama3'):
    ## tag in ['qwen','gpt','llama3','distill-llama3.2-3b']
    df = pd.read_excel("qa_list_with_answer.xlsx")
    collect = []
    for i in range(len(df)):
        js = {}
        # query
        js['user_input'] = df.loc[i,'questions']

        # contexts
        contexts_list, _ = reverse_context(df.loc[i,'%s_llm_context'%tag])
        js['retrieved_contexts'] = [c for c in contexts_list if len(c)>2]

        # llm answer
        js["response"] = df.loc[i,'%s_llm_answer'%tag]

        # true answer
        js["reference"] = df.loc[i,'answers']
        collect.append(js)

    with open("eval/%s_dataset.json"%tag, "w", encoding="utf-8") as f:
        json.dump(collect, f, indent=2, ensure_ascii=False)


def evaluate_evalscope(tag='llama3'):
    prepare_eval_data(tag=tag)

    eval_task_cfg = {
        "eval_backend": "RAGEval",
        "eval_config": {
            "tool": "RAGAS",
            "eval": {
                "testset_file": "eval/%s_dataset.json"%tag,
                "critic_llm": {
                    "model_name":"qwen-plus",
                    "api_base":qwen_base_url,
                    "api_key":qwen_api_key
                },
                "embeddings": {
                    "model_name_or_path":"nomic-ai/nomic-embed-text-v1.5",
                },
                "metrics": [
                    "Faithfulness",
                    "AnswerRelevancy",
                    "ContextPrecision",
                    "AnswerCorrectness",
                ],
                "language": "english"
            },
        },
    }
    # logger = get_logger()

    # Run task
    run_task(task_cfg=eval_task_cfg)
    print('done')

if __name__=="__main__":
    tag='distill-llama'
    generate_answers()
    # evaluate_evalscope(tag)