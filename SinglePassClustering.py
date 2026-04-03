"""
文本流式聚类主程序实现
"""
import os
import hnswlib
import numpy as np
from dotenv import load_dotenv
import time
#从当前向量目录下载入.env文件中记录的QWEN_API_KEY
load_dotenv()
from function_utils import get_vector,LLM_summary_topic_and_answer
api_key=os.getenv("QWEN_API_KEY")
print("api_key:",api_key)
K_Thr = 30 #向量引擎粗筛的topk阈值
theta2 = 0.8 #KNN检索的相似度阈值
dim = 1536 #向量维度,根据你使用的向量模型其维度是多少来决定值是多少，qwen的text_embedding_v2的向量维度:1536
num_elements =1000 #根据实际总数据量设置，高于总数据量即可。过高没必要占内存空间

#这段代码是使用 hnswlib 库来创建一个 HNSW（Hierarchical Navigable Small World）索引（p）。HNSW 是一种高效的近似最近邻搜索算法，特别适用于高维数据。
#max_elements指定了索引中最大可以容纳的元素数量
#ef_construction 控制了在构建每个节点时要访问的邻居数量.设置100意味着在构建每个节点时，索引将尝试找到100个最佳邻居
global p
p = hnswlib.Index(space='cosine', dim=dim)  #cosine  指定了每个数据点的维度（dim=1536维向量 qwen_embedding-v3模型的嵌入维度）
p.init_index(max_elements=num_elements, ef_construction=100, M=16)
#%%
#将问题和答案的数据存储起来,后续我们要用question去找answer，如果一个question对应多个answer我们将非重复的部分拼接起来组成这个问题的完整answer
#定义从file_path中读取原始数据的方法,返回每个query对应的询问频次数
def get_datasets(file_path):
    sum_count = 0 #统计总数据条目数
    qa_dict = {} #存放所有的question-answer的dict供后续使用
    query2fre = dict()
    with open(file_path, 'r', encoding='gbk') as file1:
        for i,line in enumerate(file1):
            # print(line)
            if i==0:
                continue
            #去除一些异常的数据(空白行)等，这个根据实际数据集决定是否需要
            if len(line)<10:
                continue
            # 去除行末的换行符和空格
            question=line.split(",")[1]
            answer = line.split(",")[2]
            #如果qa_dict中没有question则新建一个question，将answer存储在这个question key中
            if question not in qa_dict:
                qa_dict[question]=answer
            else:
                #如果已经有question了，那么将非重复的answer存放拼接进去作为这个问题的完整答案
                if answer not in qa_dict[question]:
                    qa_dict[question]+=answer
            sum_count += 1
            chatstr = question
            print(sum_count, chatstr) #打印每个聊天会话子串出现的文本序号和文本内容
            if chatstr in query2fre.keys():
                query2fre[chatstr] += 1 #记录每个字串的出现频率,相当于对子串进行了去重
            else:
                query2fre[chatstr] = 1
    print("原始问题-答案字典：",qa_dict)
    print("数据集总条目数:",sum_count, "数据集非重复条目数:",len(query2fre)) #586
    print("统计结束")
    return query2fre, qa_dict

file_path='./金融用户咨询.csv'
query2fre, qa_dict=get_datasets(file_path=file_path)
#%%
# topic就是簇
# sentence就是文章(文本序列)
# sent_index对应的原始索引
#文本聚类主体函数实现
def text_clustering_main_process(batch_size=25,K_Thr=K_Thr):
    print("流式聚类开始....")
    numTopic = 0  # 所属的簇编号
    sent_index = 0  # sentence index
    Index2Topic = dict() # 记录每篇文章自身index所对应的簇编号 {文章index1:簇编号1,文章index2:簇编号2...}
    Topic2Sent = dict()  # 记录簇编号和簇内各相近语义的sentence的字典 {"簇编号":[query1,query2,...]}
    costtime1 = 0
    costtime2 = 0
    querys = list()
    for query in query2fre.keys(): #将每个非重复子串取出来
        # print("query:",query) #单个query一直append到querys列表中直到25个数量批量进行embedding
        querys.append(query)
        if len(querys)==batch_size: #每隔25个发送给百炼embedding模型批量处理一次，处理完成后querys清空
            # try:
            time1 = time.time()
            vectors = get_vector(querys) #调用jina api embedding model获取对应的embedding返回2000个querys 的batch embedding的结果
            time2 = time.time()
            print(f"批处理{len(vectors)}条文本，共计耗时{time2-time1}秒") #每次批处理2000条文本，共计耗时7.026509046554565秒，但是p向量化索引是一直累计的
            for query, vector in zip(querys, vectors):
                # try:
                if numTopic == 0:
                    Index2Topic[sent_index] = numTopic
                    if numTopic not in Topic2Sent.keys(): #如果0这个簇编号不存在,则创建这个簇编号对应的键，并初始化0序号簇内对应的query list为空列表
                        Topic2Sent[numTopic]=list()
                    Topic2Sent[numTopic].append(query)
                    print("Topic2Sent_x:",Topic2Sent)
                    numTopic+=1 #簇编号自增给下面使用
                else:
                    # print("p:",p,"num_elements:",num_elements,"K_Thr:",K_Thr)
                    k_thr = min(p.element_count, num_elements, K_Thr) #取top30
                    #计算当前vector与簇索引p中各簇中心文本之间的相似度距离，并返回簇中心文本的簇编号
                    labels, distances = p.knn_query(vector, k=k_thr) #knn query 基于cosine相似度的邻近搜索取top30个临近的vector，p是向量化总索引
                    print("labels:",labels,"distances:",distances) #找到距离较近的其他label=index点与距离
                    # labels: [[3798   52 1206 2756 3442 3888   83 1278 1860  518 3837 1864 2199 3720
                    #           3443 3726 2518 3680 1781 1114 1168 1807 3195  488 2116 2769  743  898
                    #           3152 3842]]
                    # distances: [[0.28909642 0.3354559  0.33617133 0.33934987 0.34502316 0.35942817
                    #              0.3691535  0.38273215 0.389386   0.39132696 0.39207125 0.40506792
                    #              0.41521323 0.42058855 0.42749232 0.44370186 0.45656508 0.4600889
                    #              0.46420693 0.46443564 0.4697997  0.4767543  0.4776048  0.4780361
                    #              0.48445195 0.4896233  0.49245864 0.4929937  0.49535865 0.49959302]]
                    p_index = 0 #游遍遍历相似文本的index
                    maxIndex = -1 #记录相似度最高得分对应的簇中心文本的原始index
                    maxValue = 0 #记录相似度最高得分的分值
                    for label in labels[0]: #遍历每一个簇中心代表文本中的元素计算相似性距离，并转成相似度得分
                        score = 1 - distances[0][p_index] #距离越小,相似性得分越高
                        print("score:",score)
                        if score > maxValue: #找到这些距离近的sentence中与该元素的得分最高的sentence的score和label
                            maxValue = score
                            maxIndex = label
                        p_index += 1 #遍历下一个相似的元素位置
                    print("maxValue:",maxValue)
                    # 以第一条文本为种子，建立一个簇，将给定新闻文本分配到现有的、最相似的簇中
                    if maxValue > theta2: #如果高于阈值，则将当前sentence分配到已有簇(maxIndex 对应的文本所在的簇)
                        print("分配到已有簇:",maxIndex) #21
                        # print("Index2Topic:",Index2Topic)
                        print("检查点Topic2Sent:",Topic2Sent,"maxIndex:",maxIndex,"Index2Topic:",Index2Topic)
                        Index2Topic[sent_index] = Index2Topic.get(maxIndex) #{当前sentence index:找到的最相近的sentence所在的簇index}
                        if Index2Topic.get(maxIndex) not in Topic2Sent.keys(): #如果maxIndex本身的文本没有所在的簇
                            Topic2Sent[Index2Topic.get(maxIndex)] = list()
                        Topic2Sent[Index2Topic.get(maxIndex)].append(query) #将当前sentence(query)追加到所在最近的max_index所在簇的文章list中
                        # Topic2Sent: {0: ['还没'],
                        #                  1: ['可以的', '可以', '可以了吗\n', '可以了吗', '可以可以', '可以\n'],
                        #                  7: ['好了', '好了嘛', '好了\n', '好了嘛\n']}
                        print("Topic2Sent:", Topic2Sent)
                    #如果与当前query文章最相近的maxIndex对应的文章相似性sore没有超过阈值，说明当前文章属于新的簇，创建一个新的簇
                    else:
                        print("创建一个新的簇:",numTopic) # 699
                        Index2Topic[sent_index] = numTopic #{当前sentence_index:新的簇编号}
                        print("Index2Topic:",Index2Topic)
                        # Index2Topic: {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11,
                        #               12: 12, 13: 13, 14: 14, 15: 15, 16: 16, 17: 17, 18: 18, 19: 19, 20: 20,
                        #               21: 21, 22: 22, 23: 23, 24: 24, 25: 25, 26: 26, 27: 27, 28: 28, 29: 29,
                        #               30: 30, 31: 31, 32: 32, 33: 33, 34: 34, 35: 35, 36: 36, 37: 37, 38: 38,
                        #               39: 39, 40: 40, 41: 41, 42: 42, 43: 43, 44: 44, 45: 45, 46: 46, 47: 47,
                        #               48: 28, 49: 48, 50: 49, 51: 28, 52: 50, 53: 46, 54: 51, 55: 52, 56: 53}
                        print("Topic2Sent:",Topic2Sent)
                        if numTopic not in Topic2Sent.keys(): #如果新的簇内不存在任何文本元素，则创建文本元素list并将该文本加入作为首个文本
                            Topic2Sent[numTopic] = list()
                        Topic2Sent[numTopic].append(query)
                        # print("Topic2Sent:", Topic2Sent)
                        numTopic += 1 #簇编号自增给下一个新增的簇使用

                index_np = np.array(sent_index) #将原sentence的index
                # print("index_np:",index_np) #312
                vector_np = np.array(vector)
                # print("vector_np:",vector_np) #将原sentence对应的embedding结果
                # print("vector_np_len:",len(vector_np))
                p.add_items(vector_np, index_np) #p就是向量库的index索引,需要将向量化的结果<index,embedding>追加到向量化索引中
                # print("p:",p) #p: <hnswlib.Index(space='cosine', dim=128)> num_elements: 3036566 K_Thr: 30
                sent_index += 1
                # print("检查点:",sent_index, query)
                if sent_index == num_elements: #如果下一条数据到达预设置的样本处理总量(条目数限制)则不再进行循环聚类
                    break
                time3 = time.time()
                costtime1 += time3 - time1
                costtime2 += time3 - time2
            querys.clear() #清空querys重新统计2000数量

    # 如果querys不满足25条(最后一个不足一个批次的数据)也要进行相同的聚类处理
    if len(querys) > 0:
        print("最后不满足25条的query的处理，")
        # try:
        time1 = time.time()
        vectors = get_vector(querys)
        time2 = time.time()
        print(f"批处理{len(vectors)}条文本，共计耗时{time2 - time1}秒")
        for query, vector in zip(querys, vectors):
            # try:
            if numTopic == 0:
                Index2Topic[sent_index] = numTopic
                if numTopic not in Topic2Sent.keys():
                    Topic2Sent[numTopic] = list()
                Topic2Sent[numTopic].append(query)
                numTopic += 1
            else:
                k_thr = min(p.element_count, num_elements, K_Thr)  # <2000条
                labels, distances = p.knn_query(vector, k=k_thr)
                p_index = 0
                maxIndex = -1
                maxValue = 0
                for label in labels[0]:
                    score = 1 - distances[0][p_index]
                    if score > maxValue:
                        maxValue = score
                        maxIndex = label
                    p_index += 1

                # 以第一条文本为种子，建立一个簇，将给定新闻文本分配到现有的、最相似的簇中
                if maxValue > theta2:  # 0.875
                    Index2Topic[sent_index] = Index2Topic.get(maxIndex)  # 获取对应的topic文本
                    # print("maxIndex:", maxIndex)
                    # print("Index2Topic[sent_index]:", Index2Topic[sent_index])  # {sentence_index:簇代表文本}
                    if Index2Topic.get(maxIndex) not in Topic2Sent.keys():
                        Topic2Sent[Index2Topic.get(maxIndex)] = list()
                    Topic2Sent[Index2Topic.get(maxIndex)].append(query)
                # 创建一个新的簇
                else:
                    Index2Topic[sent_index] = numTopic
                    # print("numTopic:", numTopic)
                    # print("Index2Topic:", Index2Topic[sent_index])
                    if numTopic not in Topic2Sent.keys():
                        Topic2Sent[numTopic] = list()
                    Topic2Sent[numTopic].append(query)
                    # print("Topic2Sent[numTopic]:", Topic2Sent[numTopic])
                    numTopic += 1
            index_np = np.array(sent_index)
            vector_np = np.array(vector)
            p.add_items(vector_np, index_np)

            sent_index += 1  # 下一条样本的编号
            print(sent_index, query)
            if sent_index == num_elements:  # 如果下一条数据到达预设置的样本处理总量(条目数限制)则不再进行聚类
                break

            time3 = time.time()
            costtime1 += time3 - time1
            costtime2 += time3 - time2
        querys.clear()

    print("共计处理：" + str(sent_index) + "条", "共计耗时：" + str(costtime1) + "秒")
    if sent_index > 0:
        print("平均耗时：" + str(round(costtime1 / sent_index * 1000, 2)) + "毫秒",
              "共计耗时（不含转换向量）：" + str(costtime2) + "秒",
              "平均耗时（不含转换向量）：" + str(round(costtime2 / sent_index * 1000, 2)) + "毫秒")
    else:
        print("警告：未成功处理任何文本，请检查API Key配置")
    return Topic2Sent,Index2Topic,sent_index,costtime1,costtime2

Topic2Sent,Index2Topic,sent_index,costtime1,costtime2=text_clustering_main_process()
#得到的聚类结果包括:
#p:所有语料向量化索引，可以持久化保存到本地
print("Topic2Sent:",Topic2Sent)
print("Index2Topic:",Index2Topic)
print("向量索引中的向量数据条数:", p.get_current_count()) #查看向量索引中有多少条向量数据
#%%
#测试一下创建的p流式聚类的簇索引的检索效果（仅当有数据时执行）
if p.get_current_count() > 0:
    querys="资产的主要特征说一下"
    vectors = get_vector([querys])
    if vectors and len(vectors) > 0:
        labels, distances = p.knn_query(vectors[0], k=3) #取top3的结果
        print("簇中心相似文本在原始索引:",labels,"相似距离:",distances)
        # [[ 1 82  0]] [[0.05954397 0.27444965    0.30374885]]
        #距离越近说明相似度越高，因此与原序列中index=1所在的文本簇即:资产的主要特征是什么？相似
    else:
        print("警告：测试查询向量化失败，跳过检索测试")
else:
    print("警告：向量索引为空，跳过检索测试")
#%%
# 可以将索引本地持久化到文件中进行保存
# index_path='./vector_embedding_store.bin'
# print("Saving index to '%s'" % index_path)
# p.save_index(index_path)
#从持久化文件中加载index到内存
# p = hnswlib.Index(space='l2', dim=dim)
# index_path='./vector_embedding_store.bin'
# print("\nLoading index from '*.bin'\n")
# # Increase the total capacity (max_elements), so that it will handle the new data
# p.load_index(index_path, max_elements = 50000)
# print(p) #然后后续可以继续在索引中添加element
#%%

#%%
#将Topic2Sent和Index2Topic结果保存
# {0: ['什么是资产？', '如何定义资产？'], 1: ['资产的主要特征是什么？'], 2: ['请举例说明两种流动资产。'], 3: ['固定资产和无形资产有什么区别？'], 4: ['会计上如何确认一项资源为资产？'], 5: ['什么是负债？', '负债的基本定义是什么？'], 6: ['流动负债和非流动负债如何区分？'], 7: ['应付账款和应付票据有什么不同？'], 8: ['负债总是坏事吗？为什么？'], 9: ['什么是净资产？', '净资产代表了什么？'], 10: ['净资产另一个常用的名称是什么？'], 11: ['如果一家公司资产为100万，负债为70万，其净资产是多少？'], 12: ['净资产为负意味着什么？'], 13: ['解释一下复利的概念。'], 14: ['是什么使得复利如此强大？'], 15: ['复利和单利的关键区别是什么？'], 16: ['“72法则”是什么？它有什么用？'], 17: ['什么是现金流？', '现金流衡量的是什么？'], 18: ['为什么现金流对企业至关重要？'], 19: ['经营活动现金流主要关注什么？'], 20: ['净利润和现金流是一回事吗？'], 21: ['资产负债表是什么？', '资产负债表展示了什么？', ' 资产负债表的作用是什么？'], 22: ['资产负债表的基本会计恒等式是什么？'], 23: ['资产负债表为什么是“静态”的报表？'], 24: ['分析资产负债表可以帮助我们了解什么？'], 25: ['利润表是什么？', '利润表是反映哪个会计维度的报表？', '利润表的核心计算公式是什么？', '利润表的核心内容是什么？', '利润表的作用是什么？'], 26: ['毛利润和净利润有什么区别？'], 27: ['什么是市盈率（P/E Ratio）？', '如何计算市盈率？', '市盈率指标的含义是什么？', '投资者如何使用市盈率？', ' 什么是市盈率（P/E Ratio）？', ' 如何解读较高的市盈率数值？'], 28: ['一般来说，高市盈率可能表明什么？', ' 高市盈率可能反映哪些市场看法？', ' 看到高市盈率时，投资者应该思考什么？'], 29: ['市盈率的局限性是什么？'], 30: ['什么是分散投资？', '分散投资的核心名言是什么？', '分散投资的目的是什么？', ' 分散投资的主要目的是什么？'], 31: ['为什么分散投资可以降低风险？'], 32: ['什么是非系统性风险？分散投资能消除它吗？'], 33: ['什么是通货膨胀？', '通货膨胀的现象是什么？', ' 什么是通货膨胀？'], 34: ['通货膨胀的主要衡量指标是什么？'], 35: ['通货膨胀对储蓄者有什么影响？'], 36: ['温和的通货膨胀通常被认为是有益的还是有害的？'], 37: ['中央银行的主要职能是什么？', '中央银行的核心职责包括哪些？'], 38: ['中央银行常用的货币政策工具有哪些？'], 39: ['中央银行“最后贷款人”的角色是什么意思？'], 40: ['中国的中央银行是哪家银行？'], 41: ['什么是期货合约？', '期货合约的主要用途是什么？', '期货合约的基本特征是什么？', '期货合约的核心特点包括哪些？'], 42: ['期货交易是在哪里进行的？'], 43: ['期货合约和远期合约的关键区别是什么？'], 44: ['什么是期权？', '期权费是什么？'], 45: ['看涨期权和看跌期权的区别是什么？'], 46: ['美式期权和欧式期权的行权时间有何不同？'], 47: ['解释一下资本充足率。'], 48: ['监管银行资本充足率的国际协议叫什么？'], 49: ['资本充足率计算公式中的分母是什么？'], 50: ['资本充足率高的银行意味着什么？', '为什么资本充足率对银行很重要？', '银行为何需要重视资本充足率？'], 51: ['什么是首次公开募股（IPO）？'], 52: ['公司进行IPO的主要动机是什么？'], 53: ['IPO过程中的“承销商”扮演什么角色？'], 54: ['“一级市场”和“二级市场”在IPO中分别指什么？'], 55: ['什么是做空？'], 56: ['做空交易的主要风险是什么？', '做空交易蕴含着什么风险？', '做空交易可能存在哪些风险？'], 57: ['与“做空”相反的交易策略是什么？'], 58: ['什么是“轧空”？'], 59: ['什么是信用评级？', '信用评级评估的是什么？', '信用评级主要是评价什么内容？'], 60: ['全球三大信用评级机构是哪三家？'], 61: ['投资级和投机级（垃圾级）的评级分界线通常在哪里？'], 62: ['信用评级下调会对一个国家或公司产生什么影响？'], 63: ['解释一下净现值（NPV）。'], 64: ['NPV投资决策法则是什么？', 'NPV在投资决策中如何应用？', 'NPV如何用于投资项目的决策？'], 65: ['计算NPV时为什么要贴现未来现金流？'], 66: ['贴现率通常代表什么？'], 67: ['什么是抵押贷款？', '抵押贷款是如何运作的？', '抵押贷款的基本机制是什么？'], 68: ['抵押贷款中“首付款”是什么意思？'], 69: ['如果借款人连续多月无法偿还抵押贷款，贷款机构可以采取什么行动？'], 70: ['什么是LTV（贷款价值比）？'], 71: ['盈亏平衡点是什么?', '如何计算盈亏平衡点？'], 72: ['计算盈亏平衡点需要知道哪三个关键变量？'], 73: ['盈亏平衡点销量是如何计算的？'], 74: ['盈亏平衡点分析对企业经营有什么实际指导意义？', '了解盈亏平衡点对企业有何用处？'], 75: ['资产的核心要素包括哪些？'], 76: ['企业资产负债表上的资产项代表什么？'], 77: ['负债通常如何分类？'], 78: ['所有者权益和净资产是同一个概念吗？'], 79: ['复利效应是怎么回事？', ' 复利效应是什么意思？'], 80: ['为什么说复利是世界第八大奇迹？'], 81: ['为什么现金流被称为企业的“血液”？'], 82: ['为什么资产负债表是平衡的？'], 83: ['如何实现有效的分散投资？'], 84: ['通货膨胀对普通人有什么影响？'], 85: ['中央银行如何影响经济？', '中央银行通过哪些方式对经济施加影响？'], 86: ['期货市场的主要参与者有哪些？', '哪些人是期货市场中的活跃参与者？'], 87: ['期权合约赋予持有者什么？', '期权合约给予持有者的权利是什么？'], 88: ['期权有两种基本类型，它们是什么？', '期权的主要分类是哪两种？'], 89: ['监管机构对资本充足率有何要求？', '监管层面对银行的资本充足率有什么规定？'], 90: ['IPO过程指的是什么？', 'IPO的具体流程是怎样的？'], 91: ['企业上市能带来哪些好处？', '公司进行上市有哪些优势？'], 92: ['描述一下做空的操作过程。'], 93: ['高的信用等级意味着什么？', '高信用评级代表什么？'], 94: ['计算NPV的关键输入变量有哪些？', '计算NPV需要哪些重要参数？'], 95: ['抵押贷款中，抵押物起到什么作用？'], 96: ['请解释做空交易的具体步骤。'], 97: [' 什么是资产配置？'], 98: [' 什么是年化收益率？'], 99: [' 股票和债券的主要区别是什么？'], 100: [' 什么是共同基金？'], 101: [' ETF（交易所交易基金）有什么特点？'], 102: [' 现金流量表的重要性体现在哪里？'], 103: [' 什么是内部收益率（IRR）？'], 104: [' 投资中的“风险偏好”是指什么？'], 105: [' 什么是夏普比率？'], 106: [' 蓝筹股通常具备哪些特征？'], 107: [' 什么是股息率？'], 108: [' 债券的到期收益率（YTM）是什么？'], 109: [' 什么是信用利差？'], 110: [' 美联储加息通常会产生什么影响？'], 111: [' CPI和PPI这两个指数有何不同？'], 112: [' 什么是财政政策？'], 113: [' 货币政策与财政政策有何主要区别？'], 114: [' 什么是量化宽松（QE）？'], 115: [' 去杠杆化的含义是什么？'], 116: [' 什么是次级抵押贷款？'], 117: [' 投资银行与商业银行的主要业务有何不同？'], 118: [' 什么是对冲基金？'], 119: [' 私募股权（PE）投资主要做什么？'], 120: [' 风险投资（VC）专注于投资哪个阶段的企业？'], 121: [' 什么是天使投资？'], 122: [' M&A（兼并与收购）的主要动机有哪些？'], 123: [' 什么是杠杆收购（LBO）？'], 124: [' 首次公开发行（IPO）的定价机制通常是怎样的？'], 125: [' 什么是“绿鞋机制”？'], 126: [' 场外交易市场（OTC）有什么特点？'], 127: [' 什么是做市商？'], 128: [' 高频交易（HFT）的特点是什么？'], 129: [' 什么是算法交易？'], 130: [' 保险中的“免赔额”是什么意思？'], 131: [' 什么是再保险？'], 132: [' 什么是精算师？'], 133: [' 信托的基本结构涉及哪些方？'], 134: [' 什么是家族信托？'], 135: [' 离岸金融中心通常提供什么服务？'], 136: [' 什么是洗钱？'], 137: [' KYC（了解你的客户）原则为什么重要？'], 138: [' 什么是金融科技（FinTech）？'], 139: [' 区块链技术对金融业的主要价值是什么？'], 140: [' 什么是加密货币？'], 141: [' 中央银行数字货币（CBDC）与加密货币有何不同？'], 142: [' 什么是智能合约？'], 143: [' P2P借贷的模式是什么？'], 144: [' 众筹有哪些主要类型？'], 145: [' 什么是行为金融学？'], 146: [' “羊群效应”在投资中指什么？'], 147: [' 什么是处置效应？'], 148: [' 过度自信对投资有何危害？', ' 过度自信的投资心态会引发哪些负面后果？', ' 为什么说过度自信是投资中的一大陷阱？', ' 投资者过度自信的主要表现和危害是什么？', ' 过度自信如何导致投资者的实际回报降低？', ' 过度自信为什么常常与不佳的投资效果相关联？', ' 过度自信的投资者通常会在哪些方面犯错？', ' 克服过度自信对投资成功有何重要意义？'], 149: [' 什么是外汇市场？'], 150: [' 汇率是如何决定的？'], 151: [' 什么是套汇交易（Carry Trade）？'], 152: [' 国际货币基金组织（IMF）的主要职能是什么？'], 153: [' 世界银行集团的主要目标是什么？'], 154: [' 巴塞尔协议的主要目的是什么？'], 155: [' 什么是系统性风险？'], 156: [' 什么是道德风险？'], 157: [' 什么是次贷危机？'], 158: [' 什么是主权债务危机？'], 159: [' 什么是黑天鹅事件？'], 160: [' 什么是灰犀牛事件？'], 161: [' 压力测试在银行业中如何应用？'], 162: [' 什么是VAR（在险价值）模型？'], 163: [' 什么是另类投资？'], 164: [' 投资大宗商品的主要方式有哪些？'], 165: [' 什么是REITs（房地产投资信托基金）？'], 166: [' 什么是风险溢价？'], 167: [' 无风险收益率通常参考什么？'], 168: [' 什么是资本资产定价模型（CAPM）？'], 169: [' 有效市场假说（EMH）的核心观点是什么？'], 170: [' 什么是阿尔法（α）和贝塔（β）？'], 171: [' 什么是股息再投资计划（DRIP）？'], 172: [' 什么是技术分析？'], 173: [' 基本面分析主要关注哪些因素？'], 174: [' 什么是价值投资？'], 175: [' 成长型投资与价值型投资有何不同？'], 176: [' 什么是杠杆？'], 177: [' 保证金交易如何运作？'], 178: [' 什么是止损订单？'], 179: [' 什么是限价订单？'], 180: [' 市价订单的特点是什么？'], 181: [' 什么是投资组合的再平衡？'], 182: [' 什么是税收优化投资策略？'], 183: [' 什么是ESG投资？'], 184: [' 什么是绿色债券？'], 185: [' 什么是微观金融？'], 186: [' 什么是对冲（Hedging）？'], 187: [' 什么是套利？'], 188: [' 什么是金融衍生品？'], 189: [' 互换合约的主要类型有哪些？'], 190: [' 什么是“有毒资产”？'], 191: [' 什么是金融全球化？'], 192: [' 从行为金融学看，过度自信如何影响投资业绩？'], 193: [' 过度自信在交易行为上会造成什么具体问题？'], 194: [' 低估风险和高估信息准确性会带来什么投资结果？'], 195: [' 一只股票的市盈率很高，通常意味着什么？'], 196: [' 投资者为什么愿意接受高市盈率？'], 197: [' 在什么情况下高市盈率是合理的？', ' 高市盈率在何种背景下可以被视为是正当的？'], 198: [' 高市盈率同时伴随着什么风险？', ' 接受高市盈率意味着投资者必须承担哪些潜在风险？'], 199: [' 市盈率偏高可能由哪些因素导致？'], 200: [' 高市盈率是买入还是卖出的信号？'], 201: [' 如何避免高市盈率带来的投资陷阱？'], 202: [' 影响一国货币汇率长期走势的关键因素有哪些？'], 203: [' 是什么力量在主导汇率的日常波动？'], 204: [' 如何理解购买力平价对汇率的影响？'], 205: [' 利率水平如何影响汇率变化？'], 206: [' 汇率决定机制在长短期内有何不同？'], 207: [' 哪些因素能支撑一个较高的市盈率估值？'], 208: [' 投资高市盈率股票的核心逻辑是什么？'], 209: [' 为何对高市盈率股票需要保持警惕？']}
#将文本聚类的中间态数据保存
import json
with open('./Topic2Sent.json', 'w') as f:
    json.dump(Topic2Sent, f, indent=4)
# Index2Topic={0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16, 17: 17, 18: 18, 19: 19, 20: 20, 21: 21, 22: 22, 23: 23, 24: 24, 25: 25, 26: 26, 27: 27, 28: 28, 29: 29, 30: 30, 31: 31, 32: 32, 33: 33, 34: 34, 35: 35, 36: 36, 37: 37, 38: 38, 39: 39, 40: 40, 41: 41, 42: 42, 43: 43, 44: 44, 45: 45, 46: 46, 47: 47, 48: 28, 49: 48, 50: 49, 51: 28, 52: 50, 53: 46, 54: 51, 55: 52, 56: 53, 57: 54, 58: 55, 59: 56, 60: 57, 61: 58, 62: 59, 63: 60, 64: 61, 65: 62, 66: 63, 67: 64, 68: 65, 69: 66, 70: 67, 71: 61, 72: 68, 73: 69, 74: 70, 75: 71, 76: 72, 77: 72, 78: 73, 79: 74, 80: 62, 81: 75, 82: 76, 83: 77, 84: 78, 85: 79, 86: 80, 87: 81, 88: 82, 89: 71, 90: 83, 91: 84, 92: 85, 93: 86, 94: 70, 95: 87, 96: 88, 97: 89, 98: 90, 99: 91, 100: 92, 101: 93, 102: 94, 103: 95, 104: 96, 105: 97, 106: 98, 107: 99, 108: 100, 109: 101, 110: 96, 111: 102, 112: 103, 113: 104, 114: 105, 115: 64, 116: 106, 117: 107, 118: 1, 119: 108, 120: 109, 121: 110, 122: 111, 123: 62, 124: 112, 125: 113, 126: 114, 127: 115, 128: 116, 129: 117, 130: 118, 131: 119, 132: 120, 133: 44, 134: 121, 135: 122, 136: 123, 137: 124, 138: 125, 139: 126, 140: 127, 141: 128, 142: 129, 143: 130, 144: 131, 145: 132, 146: 133, 147: 134, 148: 135, 149: 136, 150: 64, 151: 137, 152: 138, 153: 139, 154: 140, 155: 141, 156: 142, 157: 143, 158: 144, 159: 7, 160: 145, 161: 146, 162: 109, 163: 109, 164: 58, 165: 147, 166: 148, 167: 149, 168: 7, 169: 150, 170: 151, 171: 152, 172: 149, 173: 153, 174: 154, 175: 155, 176: 156, 177: 157, 178: 158, 179: 159, 180: 160, 181: 161, 182: 162, 183: 163, 184: 133, 185: 164, 186: 165, 187: 166, 188: 167, 189: 168, 190: 169, 191: 170, 192: 171, 193: 172, 194: 173, 195: 174, 196: 175, 197: 176, 198: 177, 199: 178, 200: 179, 201: 180, 202: 181, 203: 182, 204: 183, 205: 184, 206: 185, 207: 186, 208: 187, 209: 188, 210: 189, 211: 190, 212: 191, 213: 192, 214: 193, 215: 194, 216: 195, 217: 196, 218: 197, 219: 198, 220: 199, 221: 200, 222: 201, 223: 202, 224: 203, 225: 204, 226: 84, 227: 205, 228: 206, 229: 62, 230: 207, 231: 208, 232: 68, 233: 209, 234: 210, 235: 211, 236: 212, 237: 213, 238: 214, 239: 215, 240: 216, 241: 69, 242: 217, 243: 218, 244: 219, 245: 220, 246: 221, 247: 222, 248: 223, 249: 224, 250: 225, 251: 113, 252: 226, 253: 227, 254: 228, 255: 229, 256: 230, 257: 231, 258: 232, 259: 233, 260: 234, 261: 235, 262: 1, 263: 236, 264: 237, 265: 238, 266: 239, 267: 240, 268: 241, 269: 242, 270: 243, 271: 244, 272: 245, 273: 246, 274: 247, 275: 248, 276: 249, 277: 64, 278: 250, 279: 251, 280: 52, 281: 252, 282: 253, 283: 254, 284: 255, 285: 256, 286: 257, 287: 258, 288: 259, 289: 260, 290: 261, 291: 262, 292: 263, 293: 264, 294: 265, 295: 266, 296: 267, 297: 268, 298: 269, 299: 22, 300: 270, 301: 271, 302: 272, 303: 273, 304: 274, 305: 275, 306: 276, 307: 277, 308: 278, 309: 279, 310: 280, 311: 281, 312: 282, 313: 283}
with open('./Index2Topic.json', 'w') as f1:
    json.dump(Index2Topic, f1)

# 从JSON文件加载数据
with open('./Topic2Sent.json', 'r') as json_file:
    loaded_data = json.load(json_file)
with open('./Index2Topic.json', 'r') as json_file1:
    Index2Topic = json.load(json_file1)
# 输出加载的字典
#将每个簇、簇内相似文本数(热度)、相似的条目数根据热度从高到低进行排序，然后写入text_clustering.txt文件中保存
List_Topic2Sent = sorted(Topic2Sent.items(), key=lambda x: len(x[1]), reverse=True) # 将字典转换为json串, json是字符串 #{0: ['还没'], 1: ['可以的', '可以', '可以了吗\n'], 2: ['有'],}
print("List_Topic2Sent:",List_Topic2Sent)
#%%
# 将按照热度排序的数据，利用簇的几个问题和答案，调用LLM总结出簇的代表性问题和答案，并通过LLM将口语化的表述做规范
import concurrent.futures
txt_file = open('text_clustering.txt', 'w', encoding='utf-8')
txt_file.write('[\n')
#处理单个簇文本的函数，通过LLM，将同一个文本簇内的多个问题和答案，概括成一个簇代表问题和一个簇代表答案
def process_cluster_item(items, sumcnt, temp_answer_list):
    """处理单个簇项目的辅助函数"""
    DictData = dict.fromkeys(('簇编号', '相似数', '相似概况', "簇代表问题", "簇代表答案"))
    DictData['簇编号'] = items[0]
    print(f"处理簇 {items[0]}: {items[1]}")
    # 调用LLM进行总结（这是耗时的操作）
    topic_question, topic_answer = LLM_summary_topic_and_answer(
        question_list=items[1],
        answer_list=temp_answer_list
    )
    print(f"簇 {items[0]} 总结完成: Q-{topic_question}, A-{topic_answer}")
    # 填充数据
    DictData['相似数'] = sumcnt
    DictData['相似概况'] = items[1]
    DictData['簇代表问题'] = topic_question
    DictData['簇代表答案'] = topic_answer
    return DictData, topic_question, topic_answer

#通过线程池批量并行多个簇内数据概括出对应的簇代表问题-簇代表答案的过程
def get_cluster_question_and_answer(List_Topic2Sent):
    total_data_dict = []
    # 检查是否有聚类结果
    if len(List_Topic2Sent) == 0:
        print("警告：没有聚类结果，跳过LLM摘要步骤")
        return total_data_dict
    # 准备线程池，最大线程数不超过20，并行处理获取每个簇的代表问题和答案
    max_workers = min(20, len(List_Topic2Sent))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 准备所有需要处理的任务
        future_to_item = {}
        for items in List_Topic2Sent:
            sumcnt = 0
            temp_answer_list = []
            for question in items[1]:
                # 每个子串的频率加到簇内总文本数量中
                sumcnt += query2fre.get(question, 0)
                # 将该question对应的answer找到
                answer = qa_dict.get(question)
                temp_answer_list.append(answer)
            # 提交任务到线程池
            future = executor.submit(
                process_cluster_item,
                items, sumcnt, temp_answer_list)
            future_to_item[future] = (items, sumcnt, temp_answer_list)
        # 收集处理结果
        for future in concurrent.futures.as_completed(future_to_item):
            try:
                result = future.result()
                if result:
                    DictData, topic_question, topic_answer = result
                    total_data_dict.append(DictData)
                    # 写入文件
                    JsonData = json.dumps(DictData, ensure_ascii=False)
                    txt_file.write('    ' + JsonData + ',\n')
                    print(f"处理完成: 簇编号 {DictData['簇编号']}")
            except Exception as e:
                items, sumcnt, temp_answer_list = future_to_item[future]
                print(f"处理簇 {items[0]} 时发生错误: {e}")
                # 错误处理：可以记录日志或使用默认值
    txt_file.write(']')
    txt_file.close()
    return total_data_dict

total_data_dict=get_cluster_question_and_answer(List_Topic2Sent=List_Topic2Sent)
print(len(total_data_dict))
print("聚类好的数据集:")
for elem in total_data_dict:
    print(elem)
