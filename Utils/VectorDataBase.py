import pickle
import numpy as np
import logging
from typing import List

logger = logging.getLogger(__name__)


class VectorDataBase():
    def __init__(self, vector: np.ndarray, sens: List[str] = None):
        """
        初始化VectorDataBase
        :param vector:ndarray 句向量
        :param sens:每一个向量所属的句子
        """
        self.sens = sens
        self.vector = vector
        self.faiss_index = None

    def build_faiss_index(self, index_type="IP"):
        """
        build faiss index
        :param index_type: IP or L2
        :return:
        """
        try:
            import faiss
        except:
            logger.info("no faiss in system")
            return
        logger.info("build faiss index...")
        if index_type == "IP":
            self.faiss_index = faiss.IndexFlatIP(self.vector.shape[1])
            self.faiss_index.add(self.vector)
        elif index_type == "L2":
            self.faiss_index = faiss.IndexFlatL2(self.vector.shape[1])
            self.faiss_index.add(self.vector)
        logger.info("finish building faiss index")

    def clear(self, data_type: str = "all"):
        """
        清理数据
        :param data_type: 要被清理的数据类型，
            all:清理所有数据
            sens:清理句子
            vector:清理句向量
            faiss:清理faiss的index
        :return:
        """
        if data_type in ["all", "sens"]:
            self.sens = None
            del self.sens
        if data_type in ["all", "vector"]:
            self.vector = None
            del self.vector
        if data_type in ["all", "faiss"]:
            self.faiss_index = None
            del self.faiss_index

    def save(self, save_path: str):
        """
        存储到本地，注意不存储faiss_index
        :param save_path: 存储路径
        :return:
        """
        with open(save_path, "wb") as fw:
            pickle.dump((self.sens, self.vector), fw)

    @classmethod
    def load(cls, read_path):
        """
        加载向量数据库
        :param read_path:读取路径
        :return:
        """
        with open(read_path, "rb") as fr:
            sens, vector = pickle.load(fr)
        return cls(vector=vector, sens=sens)


if __name__ == "__main__":
    vdb = VectorDataBase(np.array([[1., 2.3, 3], [4, 5, 6]]), ["12321312", "年后"])
    vdb.save("vdb.pkl")
    vdb = VectorDataBase.load("vdb.pkl")
    print(vdb.vector)
    print(vdb.sens)
