import json
import os
import re
import time
from datetime import datetime
from loguru import logger
from openai import OpenAI
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.prompts.loading import load_prompt

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

# from mics import Mics
# from pgdb import PgDb

ROOT_LIBPARSE = "."
os.environ["OPENAI_API_KEY"] = "EMPTY"
openai_api_key = "EMPTY"
debug_mode = True


def get_llm_model_config():
    """
    get llm model configuration from configuration file: llm_conf.json

    Returns:
        a dict with llm configuration information
    """
    path = "/llm_model_conf.json"
    with open(ROOT_LIBPARSE + path, "r") as f:
        config = json.load(f)
    return config


class LLMPredictQwen:
    """
    leverage the LLM to parse the content, usually content is a list of text.
    the parse result usually is a json string.
    """

    def __init__(self):
        # LLM configuration, is a dict with fields: llm_base, model_name,
        #   temperature, top_p, max_tokens...
        self.llm_conf = None

        # a LLM client object to call remote LLM by http/https address.
        self.llm = None

        time_stamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")

    def set_llm_conf(self, llm_conf):
        """
        set self.llm_conf value
        :param llm_conf: a dict of llm client configuration,
          include fields: llm_base, model_name, temperature, top_p, max_tokens.
        :return: None
        """
        self.llm_conf = llm_conf

    def get_llm_conf(self):
        return self.llm_conf

    def init_llm_client(self):
        """
        init a llm client by llm configruation (a dict) self.llm_conf
        set self.llm laso.
        :return: a llm client object.
        """
        llm = OpenAI(
            api_key=openai_api_key,
            base_url=self.llm_conf['llm_base'],
        )

        self.llm = llm
        return llm

    # TODO TODO: change the caller code, because we add folder param
    def get_prompt_from_file(self, folder, prompttemplate_file):
        """
        get prompt from file self.promp
        :param folder: input, the prompt template file's folder
        :param prompttemplate_file : input, prompttemplate_file, a string
        :return prompt:  a prompt (langchain PromptTemplate object)
        """
        f = folder + '/' + prompttemplate_file
        prompt = load_prompt(f)
        return prompt

    def do_predict(self, prompt, d_input_var):
        """
        do LLM predict for prompt and the input variables,
        return the response string
        :param prompt: input, a langchain PromptTemplate object
        :param d_input_var: input, a dict input var,
            e.g. {'rules':rules,'s_list_str':s_list_str}
        :return: a response string from remote LLM
        """
        # input params (a list of string) inside prompt's template
        input_variables = prompt.input_variables
        print(input_variables)

        # template of langchain PromptTemplate,  is a string
        template = prompt.template
        print(template)

        input = {
            k: v
            for k, v in d_input_var.items() if k in prompt.input_variables
        }
        print("input")
        print(input)

        prompt_value = prompt.invoke(input)
        print(prompt_value)
        ps = prompt_value.to_string()

        chat_response = self.llm.chat.completions.create(
            model=self.llm_conf['model_name'],
            temperature=self.llm_conf['temperature'],
            # top_p=0.95, if set temperature, could skip top_p
            # top_p=self.llm_conf['top_p'],
            max_tokens=self.llm_conf['max_tokens'],
            # Feb 19 2024 avoid repreat chars
            presence_penalty=self.llm_conf['presence_penalty'],
            # model_kwargs={"stop": ["OUTPUT_JSON_END", "##Input", "##Comment", "##Comments", "##End", "##Error", "##Note", "##Summary", "##Feedback", "##输入", "##Hints",  "INPUT_STOP","."]}
            # model_kwargs={"stop": ["OUTPUT_JSON_END", "##Input", "##Comment", "##Comments", "##End", "##Error", "##Note", "##Summary", "##Feedback", "##输入", "##Hints"]}
            # model_kwargs={"stop": ["OUTPUT_STOP", "##Input", "##Comment", "##Comments", "##End", "##Error", "##Note", "##Summary", "##Feedback", "##输入", "##Hints", "##Hint", "##Description", "##Task", "##Rules", "##Explanation", "#Other", "#About", " ##Profile"]}
            stop=["OUTPUT_STOP", "##Input", "##Comment", "##Comments", "##End", "##Error", "##Note", "##Summary",
                  "##Feedback", "##输入", "##Hints", "##Hint", "##Description", "##Task", "##Rules", "##Explanation",
                  "#Other", "#About", " ##Profile"],

            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": ps},
            ]
        )

        # print("Chat response:", chat_response)

        # print("\n")
        # print(f"type choices {type(chat_response.choices)}")
        # print(f"type message {type(chat_response.choices[0].message)}")
        # print(f"type content {type(chat_response.choices[0].message.content)}")

        response = ''
        if chat_response is not None and chat_response.choices is not None:
            choice = chat_response.choices[0]
            if choice.message is not None and choice.message.content is not None:
                response = chat_response.choices[0].message.content
        # print(" response_start ")
        # print(response)
        # print(" response_end ")

        return response


def test5():
    """
    In this function, operations just for test langchain ability, DO NOT USE IT.
    please run it at root dir of the project, e.g.
    (base) yihua@yoga:~/project/law_doc_parse$PYTHONPATH=. python3 ./docparse/libparse/llm_predict_qwen.py
    """
    llmpt = LLMPredictQwen()

    config = get_llm_model_config()
    llm_conf = config['llm_conf']
    llmpt.set_llm_conf(llm_conf)
    llmpt.init_llm_client()

    """
    # test OK for Qwen1.5-72B-chat-4bits-AWQ
    # https://python.langchain.com/docs/expression_language/cookbook/prompt_llm_parser
    prompt = ChatPromptTemplate.from_template("tell me a joke about {foo}")
    chain = prompt | llmpt.llm
    response = chain.invoke({"foo": "frog"})
    print(response)
    """


    # test OK for Qwen1.5-72B-chat-4bits-AWQ
    # https://python.langchain.com/docs/expression_language/cookbook/prompt_llm_parser
    prompt = ChatPromptTemplate.from_template("tell me a joke about {foo}")
    chain = prompt | llmpt.llm
    input = {"foo":"frog"}
    response = chain.invoke(input)
    print(response)


    """
    # test OK for multi-input-vars
    # test OK for Qwen1.5-72B-chat-4bits-AWQ
    # https://python.langchain.com/docs/expression_language/cookbook/prompt_llm_parser
    prompt = ChatPromptTemplate.from_template("tell me a joke about {foo} and {y}")
    chain = prompt | llmpt.llm
    input = {"foo":"frog", "y":"pig"}
    response = chain.invoke(input)
    print(response)
    #print(dir(response))
    """

    """
    # test OK for multi-input-vars, the key point is string format function.
    # test OK for Qwen1.5-72B-chat-4bits-AWQ
    # https://python.langchain.com/docs/expression_language/cookbook/prompt_llm_parser
    s = " tell me a joke about {foo} and {y}"
    input = {"foo":"frog", "y":"pig"}
    #s1 =  s.format(foo = "frog", y = "pig")
    s1 =  s.format(**input)
    prompt = ChatPromptTemplate.from_template(s1)
    chain = prompt | llmpt.llm
    # the var for invoke() is necessary.
    response = chain.invoke(input)
    print(response)
    #print(dir(response))
    """

    """
    # test OK for multi-input-vars, the key point is string format function.
    # test OK for Qwen1.5-72B-chat-4bits-AWQ
    # https://python.langchain.com/docs/expression_language/cookbook/prompt_llm_parser
    # leverage the ChatPromptTemplate to format the string 
    s = " tell me a joke about {foo} and {y}"
    tmp_prompt = ChatPromptTemplate.from_template(s)
    input = {"foo":"frog", "y":"pig"}
    prompt_value = tmp_prompt.invoke(input)
    # Maybe the failed reason is  "Human:"
    # Human:  tell me a joke about frog and pig
    s1 = prompt_value.to_string()
    print(s1)
    prompt = ChatPromptTemplate.from_template(s1)
    #chain = prompt | llmpt.llm
    # the var for invoke() is necessary.
    #response = chain.invoke(input)
    response = llmpt.llm.invoke(s1)
    print(response)
    #print(dir(response))
    """

    """
    # test KO for multi-input-vars
    # test KO for Qwen1.5-72B-chat-4bits-AWQ
    # https://python.langchain.com/docs/expression_language/cookbook/prompt_llm_parser
    s = " tell me a joke about {foo} and {y} "
    s2 = "请不要输出其他与答案无关的字符."
    input = {"foo": "frog", "y": "pig"}
    prompt = ChatPromptTemplate.from_template(s, partial_variables=input)
    s3 = prompt.format(**input)
    print(prompt)
    print(dir(prompt))
    # input_variables=['foo', 'y'] messages=[HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['foo', 'y'], template=' tell me a joke about {foo} and {y} '))]
    chain = prompt | llmpt.llm
    response = chain.invoke(input)
    print(response)
    #print(dir(response))
    """

    """
    # test KO for Qwen1.5-72B-chat-4bits-AWQ : return is empty.
    # Integrate the model in an LLMChain
    # https://python.langchain.com/docs/integrations/llms/vllm
    prompt = PromptTemplate.from_template("tell me a joke about {foo}")
    chain = LLMChain(llm=llmpt.llm, verbose=True, prompt=prompt)
    #input = {"foo":"bee"}
    #response = chain.predict(**input)
    foo = "bee"
    response = chain.invoke(foo)
    print(response)
    """




def test6():
    """
    COULD USED IN PROD ENV.
    please run it at root dir of the project, e.g.
    (base) yihua@yoga:~/project/law_doc_parse$PYTHONPATH=. python3 ./docparse/libparse/llm_predict_qwen.py
    OK for models: Qwen1.5-72B-chat-4bits-AWQ
    """
    llmpt = LLMPredictQwen()

    config = get_llm_model_config()
    llm_conf = config['llm_conf']

    llmpt.set_llm_conf(llm_conf)
    llmpt.init_llm_client()
    # prompttemplate_file = kg_prompt_config[libparse_type]['logical_element'][0]['introductory_part']['parse_prompt'][0]['prompttemplate_file']
    prompttemplate_file = "keep_test-prompttemplate-kg-criminal_judgement-introductory_part-v1.3.1.yaml"
    folder = "."
    tmp_prompt = llmpt.get_prompt_from_file(folder, prompttemplate_file)
    print("A" * 80)
    print(tmp_prompt)
    print("B" * 80)

    d_input_var = dict()

    s_list_str = """
    [
    '陕西省高级人民法院<br>',
    '刑 事 判 决 书<br>',
    '（2020）陕刑终267号<br>',
    '抗诉机关陕西省宝鸡市人民检察院。<br>',
    '原审被告人翁某甲，<br>',
    '男，<br>',
    '1966年2月2日生于陕西省平利县，<br>',
    '汉族，<br>',
    '小学文化，<br>',
    '矿工，<br>',
    '户籍所在地陕西省安康市平利县，<br>',
    '住山西省繁峙县狼涧矿山，<br>',
    '现羁押于扶风县看守所。<br>',
    '2019年11月9日因涉嫌抢劫被刑事拘留，<br>',
    '同年11月15日被监视居住，<br>',
    '2019年11月22日被刑事拘留，<br>',
    '同年12月28日被逮捕，<br>',
    '指定辩护人马某乙，<br>',
    '陕西某某律师事务所律师。<br>'
    ]
    """
    print(s_list_str)

    """
    d_input_var['s_list_str'] = s_list_str
    prompt_value = tmp_prompt.invoke(d_input_var)
    s1 = prompt_value.to_string()
    prompt = ChatPromptTemplate.from_template(s1)
    """
    d_input_var['s_list_str'] = s_list_str
    response = llmpt.do_predict(tmp_prompt, d_input_var)
    print(response)


def main():
    msg = """
    please run it at root dir of the project, e.g.
    (base) yihua@yoga:~/project/law_doc_parse$PYTHONPATH=. python3 ./docparse/libparse/llm_predict_qwen.py
    """
    print(msg)


if __name__ == '__main__':
    main()
    test5()
    # test6()
