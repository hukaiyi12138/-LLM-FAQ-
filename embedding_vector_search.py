"""
知识库相似度检索实现：构建向量数据库，然后基于索引查询。构建的语料来源于文本聚类得到的高质量QA
"""

import os
import requests
import chromadb
from dotenv import load_dotenv
load_dotenv()
api_key=os.getenv("QWEN_API_KEY")
print(api_key)


from typing import List, Optional, Dict, Any
#定义向量化类，通过调用百炼text-embedding-v3模型完成创建知识库时的文本向量化
class QwenEmbeddingFunction:
    def __init__(self, api_key: str =api_key, model_name: str = "text-embedding-v3"):
        """
        初始化通义千问Embedding函数

        参数:
            api_key: 阿里云API密钥
            model_name: 使用的embedding模型名称，默认为"text-embedding-v1"
        """
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding"

    def __call__(self, input: List[str]) -> List[List[float]]:
        """
        调用通义千问API获取文本的embedding

        参数:
            input: 需要embedding的文本列表 (参数名必须为input以匹配ChromaDB要求)

        返回:
            文本对应的embedding列表
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model_name,
            "input": {
                "texts": input
            }
        }

        response = requests.post(self.base_url, json=payload, headers=headers)
        if response.status_code != 200:
            raise Exception(f"API请求失败，状态码: {response.status_code}, 错误信息: {response.text}")

        result = response.json()

        if "output" not in result or "embeddings" not in result["output"]:
            raise Exception(f"API返回格式异常: {result}")

        # 提取embedding数据并按原始文本顺序返回
        embeddings = [None] * len(input)
        for item in result["output"]["embeddings"]:
            index = item["text_index"]
            embeddings[index] = item["embedding"]

        return embeddings

import uuid
#
class ChromaDBWithQwen:
    def __init__(self, collection_name: str = "qwen_collection1", persist_directory: str = "./chroma_db",api_key=api_key):
        """
        初始化ChromaDB与通义千问Embedding的集成

        参数:
            collection_name: 集合名称
            persist_directory: 数据库持久化目录
        """
        # 从环境变量获取API密钥
        if not api_key:
            raise ValueError("请设置ALIYUN_API_KEY环境变量")

        # 创建自定义的embedding函数
        self.embedding_function = QwenEmbeddingFunction(api_key)

        # 初始化Chroma客户端
        self.client = chromadb.PersistentClient(path=persist_directory)

        # 获取或创建集合
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}
        )

    def add_documents(self, documents: List[str], ids: Optional[List[str]] = None,
                      metadatas: Optional[List[Dict[str, Any]]] = None):
        """
        添加文档到集合中

        参数:
            documents: 文档文本列表
            ids: 可选的文档ID列表
            metadatas: 可选的元数据列表
        """
        if ids is None:
            ids = [str(uuid.uuid4()) for i in range(len(documents))]

        self.collection.add(
            documents=documents,
            ids=ids,
            metadatas=metadatas
        )

    def query(self, query_texts: List[str], n_results: int = 5) -> Dict[str, Any]:
        """
        查询相似的文档

        参数:
            query_texts: 查询文本列表
            n_results: 返回的结果数量

        返回:
            查询结果，包含文档、距离、ID等信息
        """
        return self.collection.query(
            query_texts=query_texts,
            n_results=n_results,

        )

    def get_collection_info(self) -> Dict[str, Any]:
        """
        获取集合信息

        返回:
            集合的基本信息
        """
        return {
            "name": self.collection.name,
            "count": self.collection.count()
        }





