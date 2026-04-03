"""
工具函数定义:
get_vector:将输入的文本list转成向量list
LLM_summary_topic_and_answer:通过LLM将多个同一文本簇内的question_list和answer_list进行知识压缩和总结，生成"簇代表问题"和"簇代表答案"
"""


#将一个querys文本列表转成一个embedding向量嵌入列表
import os
import json
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
api_key=os.getenv("QWEN_API_KEY")
print(api_key)
def get_vector(querys):
    # 定义API信息
    client = OpenAI(
        api_key=api_key,  # 如果您没有配置环境变量，请在此处用您的API Key进行替换
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # 百炼服务的base_url
    )
    try:
        completion = client.embeddings.create(
            model="text-embedding-v2",
            input=querys, #'衣服的质量杠杠的，很漂亮，不枉我等了这么久啊，喜欢，以后还来这里买'
            dimensions=1024,  # 指定向量维度（仅 text-embedding-v3及 text-embedding-v4支持该参数）
            encoding_format="float"
        )
        result=json.loads(completion.model_dump_json())
        embed_result=[k.get("embedding") for k in result["data"]]
        # print(embed_result)
        return embed_result
    except Exception as e:
        print(f"e:{e}")
        return []  # 如果所有重试都失败，返回空列表

querys=["你好","今天天气真好"] #qwen向量化模型支持querys最多25一个批次
result=get_vector(querys=querys)#
print(len(result))


#%%
# 定义通过LLM进行簇的代表标题的总结与簇代表标题对应的答案的总结
def LLM_summary_topic_and_answer(question_list, answer_list):
    """
    :param question_list: 一个簇的所有问题
    :param answer_list: 一个簇的所有答案
    :return: 簇的代表性问题和完整的答案(高质量qa知识)
    """
    prompt = f"""你是一个数据处理大师，你的任务是将输入的question_list中的问题综述出一个能代表所有元素且表达正式的question，然后根据提供的answer_list中的答案进行回复，回复的内容尽可能丰富完善但不要重复。
    举例:
    输入:question_list:["什么是抵押贷款？", "抵押贷款是如何运作的？", "抵押贷款的基本机制是什么？"]
        answer_list:["抵押贷款是指借款人以其财产（如房产）作为抵押物向贷款人借款，如果借款人未能按时还款，贷款人有权收回并出售该抵押物。","借款人将其拥有的财产（如房产）抵押给贷款人（如银行）以获取贷款。如果借款人违约，贷款人有权依法处置该抵押物来收回贷款。","抵押贷款是借款人将资产（如不动产）作为担保物从贷款机构获得资金，如果违约，贷款人可依法收回并处置抵押物以补偿损失。"]
    输出:[问题:抵押贷款的核心概念及其运作机制？|答案:抵押贷款是一种以特定资产（主要为不动产如房产）作为债权担保的融资方式。其核心概念是借款人（抵押人）通过将资产法定权益暂时转移给贷款人（抵押权人）来获得资金，同时保留资产使用权。运作机制包含三个关键环节：首先，借款人以符合要求的资产向贷款机构提出抵押申请并完成价值评估；其次，双方签订合同明确贷款金额、利率、期限及抵押条款，并办理抵押登记确立法律效力；最后，在还款期间借款人按期还本付息，若发生违约，贷款人有权依法通过拍卖、变卖等处置抵押物的方式优先受偿，从而实现风险缓释。这种机制通过物权保障显著降低信贷风险，使借款人能获得更低利率和更高额度，同时规范化的登记制度保障了交易安全。]
    输入:question_list: ["什么是资本充足率？", "银行为何需要重视资本充足率？", "监管层面对银行的资本充足率有什么规定？"]
        answer_list: ["资本充足率是衡量银行自有资本能否覆盖其风险加权资产潜在损失的指标，反映了银行的财务稳健性和风险抵御能力。", "银行重视资本充足率是为了确保拥有足够的资本缓冲来吸收意外损失，维持债权人信心，满足监管要求，并保障银行持续稳健经营。", "监管机构（如央行或银保监会）会设定最低资本充足率标准（例如遵循巴塞尔协议的8%或更高），并定期监测，对不达标的银行采取限制业务、要求补充资本等措施，以维护整个金融体系的稳定。"]
    输出:[问题:资本充足率的核心要义、重要性及监管要求？|答案:资本充足率是评估银行体系健康度的核心监管指标，指银行持有的合格资本与其风险加权资产之间的比率，用于衡量银行用自有资本应对资产潜在损失的能力。其重要性体现在三个方面：一是风险缓冲功能，充足的资本能有效吸收经营中的意外亏损，避免资不抵债；二是信心维系功能，高的资本充足率能增强存款人、债权人和市场对银行的信任，维护金融稳定；三是经营约束功能，它限制了银行的过度风险扩张，促进审慎经营。监管层面，各国监管机构（如中国的国家金融监督管理总局）依据巴塞尔协议框架，对商业银行设定了分层次的最低资本要求（通常包括核心一级资本充足率、一级资本充足率和总资本充足率），并辅以资本留存缓冲、逆周期资本缓冲等附加要求。监管机构通过非现场监管和现场检查持续监测银行达标情况，对资本不足的机构采取限制分红、限制资产增长、要求提交资本补充计划乃至强制重组等干预措施，旨在系统性防范金融风险，保障银行业整体稳健运行。]


    输入:question_list:[{str(question_list)}]
        answer_list:[{str(answer_list)}]
    输出:"""
    client = OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    completion = client.chat.completions.create(
        # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        model="qwen-plus",
        messages=[
            {"role": "user", "content": prompt},
        ],
        extra_body={"enable_thinking": False},
    )
    qa_result_str = completion.choices[0].message.content
    try:
        qa_result_str = qa_result_str.replace("[", "").replace("]", "").replace("问题:", "").replace("答案:", "")
        topic_question = qa_result_str.split("|")[0]
        topic_answer = qa_result_str.split("|")[1]
    except:
        topic_question = []
        topic_answer = []
    return topic_question, topic_answer
