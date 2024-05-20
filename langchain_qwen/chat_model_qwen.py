from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from openai import OpenAI

llm = OpenAI(api_key='EMPTY',
             base_url="http://183.195.106.189:10340/v1")

messages = [
    ("system", "You are a helpful assistant that translates English to French."),
    ("human", "Translate this sentence from English to French. I love programming."),
]
res = llm.chat.completions.create(model='/home/zhijue/project/huggingface/Qwen1.5-72B-Chat-AWQ',
                                  temperature=0.8,
                                  # top_p=0.95, if set temperature, could skip top_p
                                  # top_p=self.llm_conf['top_p'],
                                  max_tokens=7000,
                                  # Feb 19 2024 avoid repreat chars
                                  presence_penalty=1.2,
                                  # model_kwargs={"stop": ["OUTPUT_JSON_END", "##Input", "##Comment", "##Comments", "##End", "##Error", "##Note", "##Summary", "##Feedback", "##输入", "##Hints",  "INPUT_STOP","."]}
                                  # model_kwargs={"stop": ["OUTPUT_JSON_END", "##Input", "##Comment", "##Comments", "##End", "##Error", "##Note", "##Summary", "##Feedback", "##输入", "##Hints"]}
                                  # model_kwargs={"stop": ["OUTPUT_STOP", "##Input", "##Comment", "##Comments", "##End", "##Error", "##Note", "##Summary", "##Feedback", "##输入", "##Hints", "##Hint", "##Description", "##Task", "##Rules", "##Explanation", "#Other", "#About", " ##Profile"]}
                                  # stop=["OUTPUT_STOP", "##Input", "##Comment", "##Comments", "##End", "##Error",
                                  #       "##Note", "##Summary",
                                  #       "##Feedback", "##输入", "##Hints", "##Hint", "##Description", "##Task",
                                  #       "##Rules", "##Explanation",
                                  #       "#Other", "#About", " ##Profile"],

                                  messages=[
                                      {"role": "system", "content": "You are a helpful assistant."},
                                      {"role": "user", "content": "Translate this sentence from English to French. I love programming."},
                                  ])
print(res)
