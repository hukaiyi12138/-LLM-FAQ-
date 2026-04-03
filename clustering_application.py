"""
本文件用于将聚类好的高质量qa数据集用于:
(1) 高频FAQ挖掘
(2) 知识库建设，相似问题检索
"""
##应用1:用聚类好的数据构建高频FAQ(假设认为热度3个的簇是热门簇需要进行筛选)
def sort_by_similarity(hot_freq=3):
    """
    根据相似数从大到小排序列表中的字典
    参数:data_list: 包含字典的列表，每个字典应包含'相似数'键
    返回:排序后的列表
    """
    data_list=[]
    with open('text_clustering.txt', mode='r',encoding='utf-8', newline='') as f:
        for lines in f:
            documents_list = []
            metadata_list = []
            # print("lines:",lines)
            if len(lines)<10: #过滤首位的[和]字符
                continue
            #将text转成dict
            json_lines=eval(lines.strip())[0]
            # 超过一定频率的问题才会被视为高频问题
            if json_lines.get('相似数')> hot_freq:
                data_list.append(json_lines)
    data_list=sorted(data_list, key=lambda x: x['相似数'], reverse=True)
    return data_list
# 使用示例
sorted_data = sort_by_similarity()
print("按热度排序高质量FAQ数据:\n")
for elem in sorted_data:
    print(elem)
#%%
#应用2:构建qa知识库，进行向量检索问答
#实际使用时由于对话类的数据可能较大，可以过滤低频簇，仅用高频簇进行知识库建设
from embedding_vector_search import *
def init_db():
    # 初始化
    chroma_qwen = ChromaDBWithQwen()
    print("chroma_qwen初始化成功")
    with open('text_clustering.txt', mode='r',encoding='utf-8', newline='') as f:
        for lines in f:
            documents_list = []
            metadata_list = []
            # print("lines:",lines)
            if len(lines)<10: #过滤首位的[和]字符只取簇的数据
                continue
            #将text转成dict
            json_lines=eval(lines.strip())[0]
            # print("json_lines:",json_lines)
            question=json_lines.get("簇代表问题")
            answer=json_lines.get("簇代表答案")
            documents_list.append(question)
            metadata_list.append({"answer":answer})
            # print("添加高质量文档到集合中...")
            chroma_qwen.add_documents(documents=documents_list,metadatas=metadata_list)
    # 查询集合信息
    info = chroma_qwen.get_collection_info()
    print(f"集合信息: 名称={info['name']}, 文档数量={info['count']}")
    return chroma_qwen

#%%
chroma_qwen=init_db()
query_text="期货合约的基本特征是什么？"
print("用户当前问题:",query_text)

# 检查知识库中是否有数据
info = chroma_qwen.get_collection_info()
print(f"集合信息: 名称={info['name']}, 文档数量={info['count']}")

if info['count'] > 0:
    try:
        results = chroma_qwen.query([query_text]) #相似度检索得到答案
        max_score = 0
        score_thred=0.8
        
        #进行相似度得分的过滤,余弦相似度阈值超过0.8才作为检索返回结果. 由于我们用的是chromadb的API，其不支持相似度得分返回值，只支持相似度距离返回值。因此我们需要将距离转成得分
        for i, (doc, distance) in enumerate(zip(results['documents'][0], results['distances'][0])):
            score = 1 - distance
            #记录最大得分
            if score > max_score and score>score_thred:
                max_score = score
                result_document = doc
                print(f"知识库检索结果: 问题:{doc} 答案:{results['metadatas'][0][0].get('answer')} (距离: {distance:.4f})，score:{(1 - distance):.4f}")
    except Exception as e:
        print(f"检索查询失败: {e}")
        print("提示：请检查API Key配置或确保text_clustering.txt文件存在")
else:
    print("警告：知识库为空，无法进行检索查询")
    print("提示：请先运行SinglePassClustering.py生成聚类结果")
