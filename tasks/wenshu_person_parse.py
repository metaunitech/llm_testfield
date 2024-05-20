# coding=utf-8
import json

from modules.llm_utils.langchain_zhipu import ChatZhipuAI
from modules.llm_utils.langchain_qwen import ChatQwenAI
from pathlib import Path
import yaml
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
import re
from tenacity import retry, stop_after_attempt, wait_random

from typing import List, Dict

LLM_CONFIG_PATH = Path(__file__).parent.parent.parent.parent / 'configs' / 'llm_configs.yaml'
with open(LLM_CONFIG_PATH, 'r', encoding='utf-8') as f:
    config_data = yaml.load(f, Loader=yaml.FullLoader)

PLATFORM_NAME = 'Qwen'
MODEL_NAME = list(config_data.get(PLATFORM_NAME, {}).get('models').keys())[0]
MODEL_PARAM = {'platform': PLATFORM_NAME,
               'model_name': MODEL_NAME,
               'api_base': config_data.get(PLATFORM_NAME, {}).get('api_base')}


class PersonRelated(BaseModel):
    person: str = Field(
        description='法院文书中涉及的人员的姓名'
    )
    personRole: List[str] = Field(
        description='法院文书中涉及的人员的身份，从选项中选择一个，无法确定的选择“不确定”',
        selection=['原告', '被告', '原告律师', '被告律师', '不确定']
    )
    personDetails: List[Dict] = Field(
        description='法院文书中涉及的人员的详细信息，包含但不仅限于：籍贯，出生年月，居住地，性别，本案中的行文等。'
    )


class PersonRelatedALL(BaseModel):
    allPersons: List[PersonRelated] = Field(
        description='所有与这个法律文书相关的人员。'
    )


class PersonRelation(BaseModel):
    person_from: str = Field(
        description='关系中的动作发起者'
    )
    person_to: str = Field(
        description='关系中的动作接收者'
    )
    relation: str = Field(
        description='两者的关系，可能是事件，也有可能是关系。'
    )


class PersonRelations(BaseModel):
    personRelations: List[PersonRelation] = Field(
        description='文中提到的所有人物之间的关系'
    )


class WenshuParser:
    def __init__(self, llm_api_key=None, transcript=None, lang='zh'):
        self.chat_model = ChatQwenAI(model_name=MODEL_PARAM.get('model_name'),
                                     qwen_api_base=MODEL_PARAM.get('api_base'))
        if llm_api_key:
            self.chat_model_better = ChatZhipuAI(model_name='chatglm_turbo', zhipuai_api_key=llm_api_key)
        self.transcript = transcript
        self.lang = lang

    @staticmethod
    def english_stringfy_string(input):
        return re.sub('，', ', ', input)

    @retry(wait=wait_random(min=1, max=3), stop=stop_after_attempt(3))
    def parse_person(self, transcript):
        parser = PydanticOutputParser(pydantic_object=PersonRelatedALL)
        retry_parser = OutputFixingParser.from_llm(parser=parser, llm=self.chat_model)
        prompt = ("# ROLE: 你是一个经验丰富的律师，你擅长从法律文书中提取出关键信息。"
                  "\n# TASK: 我会给你提供一篇法律文书，你需要帮助我从中提取出与本案相关的人员的关键信息。"
                  "\n# OUTPUT_FORMAT: 请将结果返回为JSON格式。{format_instructions}"
                  "\n# INPUT: {input_str}"
                  "\nYOUR ANSWER:")
        prompt = prompt.format(input_str=transcript,
                               format_instructions=parser.get_format_instructions())
        res_content = self.chat_model.predict(prompt)
        res_content = self.english_stringfy_string(res_content)
        extracted_kies = retry_parser.parse(res_content)
        persons = extracted_kies.dict()['allPersons']
        return persons

    @retry(wait=wait_random(min=1, max=3), stop=stop_after_attempt(3))
    def parse_relations(self, transcript, persons: dict):
        parser = PydanticOutputParser(pydantic_object=PersonRelations)
        retry_parser = OutputFixingParser.from_llm(parser=parser, llm=self.chat_model)
        prompt = ("# ROLE: 你是一个经验丰富的律师，你擅长从法律文书中提取人物之间的关系。"
                  "\n# TASK: 我会给你提供一篇法律文书以及本案的一些涉案人员关键信息，你需要帮助我从中文中归纳这些人物之间的关系。"
                  "\n# OUTPUT_FORMAT: 请将结果返回为JSON格式。{format_instructions}"
                  "\n# INPUT: "
                  "\n文书：{input_str}"
                  "\n人物：{persons}"
                  "\nYOUR ANSWER:")
        prompt = prompt.format(input_str=transcript,
                               format_instructions=parser.get_format_instructions(),
                               persons=json.dumps(persons, indent=2, ensure_ascii=False))
        res_content = self.chat_model.predict(prompt)
        res_content = self.english_stringfy_string(res_content)
        extracted_kies = retry_parser.parse(res_content)
        relations = extracted_kies.dict()['personRelations']
        return relations


if __name__ == "__main__":
    script = """中华人民共和国最高人民法院
驳 回 申 诉 通 知 书
（2023）最高法刑申100号
毛某1:
你因故意伤害一案，不服河北省成安县人民法院（2011）成刑初字第27号刑事附带民事判决、河北省邯郸市中级人民法院（2011）邯市刑终字第240号刑事附带民事判决和河北省高级人民法院（2013）冀刑监字第108号驳回申诉通知，以原审认定事实不清，证据不足，你没有作案目的，案发时不在现场，作案刀具去向不明，证据相互矛盾等为由,向本院提出申诉。
本院经组成合议庭阅卷审查认为，原审认定你因被人举报违反计划生育被拘留一事而记恨本村村支书赵某5河，在与毛某2、毛某3、李某、毛某4等人喝酒期间商量报复，并于当晚凌晨酒后用砖头、啤酒瓶等扔砸赵某5河家街门后逃离。以上事实你亦供认。赵某5河叫其弟赵某1等人在本村寻找扔砸街门之人时，找到你们。你们当事双方相互熟识，你虽然否认殴打被害人赵某1、不供认作案刀具下落，但在案证据不仅有被害人赵某1陈述指认你，且有现场目击证人赵某2、赵某3、赵某4等人证言、被告人毛某2的供述、法医学鉴定书等证据证实。案发后，毛某2的父亲曾找中间人调解你们打伤赵某1一事，以及公安机关办案人员与毛某2网上聊天记录等证据亦可印证，足以认定。原审认定事实清楚，证据确实、充分。
你与被告人毛某2对被害人赵某1实施殴打，造成赵某1右侧额顶部硬膜外血肿，额骨骨折、左侧顶骨骨折等处损伤，经法医鉴定损伤程度为重伤，伤残等级为拾级。原审定性准确，量刑适当。审判程序合法。你的申诉理由不能成立。
综上，你的申诉不符合《中华人民共和国刑事诉讼法》第二百五十三条规定的应当重新审判情形，本院决定驳回你的申诉。
特此通知。
二〇二四年三月十一日"""
    ins = WenshuParser()
    res = ins.parse_person(script)
    from pprint import pprint
    print("=>涉案人员")
    pprint(res)
    print("=>人物关系")
    res2 = ins.parse_relations(script, res)
    pprint(res2)