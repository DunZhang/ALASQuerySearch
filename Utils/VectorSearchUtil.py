import scipy.spatial
from sklearn.preprocessing import normalize
import logging
from typing import List, Dict
import numpy as np
from Utils.VectorDataBase import VectorDataBase

logger = logging.getLogger(__name__)


def find_topk_by_vecs(source_vecs: np.ndarray, vec_db: VectorDataBase, topk: int, metric="cosine", use_faiss="auto"):
    """
    find topk vecotr
    :param source_vecs:源句向量
    :param vec_db: 目标向量库
    :param topk: 对于每一个源句向量从目标向量库选取topk最相似的向量
    :param metric: cosine 或者 euclidean
    :param use_faiss: bool,str, True or False or "Auto"
    :return: res_index and res_distance
        topk's index and instance
    """
    if isinstance(use_faiss, str) and use_faiss.lower() == "auto":
        use_faiss = True
        try:
            import faiss
        except:
            print("faiss is not availble")
            use_faiss = False
    res_distance, res_index = None, None
    if use_faiss:
        print("use faiss to search")
        if metric == "cosine":
            source_vecs = normalize(source_vecs, axis=1)
            if vec_db.faiss_index is None:
                vec_db.build_faiss_index("IP")
            if vec_db.faiss_index is not None:
                res_distance, res_index = vec_db.faiss_index.search(source_vecs, topk)
        elif metric == "euclidean":
            if vec_db.faiss_index is None:
                vec_db.build_faiss_index("L2")
            if vec_db.faiss_index is not None:
                res_distance, res_index = vec_db.faiss_index.search(source_vecs, topk)
    if res_index is None:
        print("use scipy to search")
        sims = scipy.spatial.distance.cdist(source_vecs, vec_db.vector, metric)
        res_index = np.argsort(sims, axis=1)[:, 0: topk]
        res_distance = np.ones(shape=res_index.shape, dtype=np.float)
        for i in range(res_index.shape[0]):
            for j in range(res_index.shape[1]):
                if metric == "cosine":
                    res_distance[i, j] = 1 - sims[i, res_index[i, j]]
                else:
                    res_distance[i, j] = sims[i, res_index[i, j]]
    return res_index, res_distance


def find_topk_by_sens(sen_encoder, source_sens: List[str], target_sens: List[str], topk: int,
                      src_kwargs: Dict, tar_kwargs: Dict,
                      metric="cosine", use_faiss="auto"):
    """
    :param sen_encoder: ISentenceEncoder，句向量编码器，如果src_sen_encoder和tar_sen_encoder有一个为空，
        那么源句子和目标句子都用该编码器编码。如果这个两个编码器都不为空就分别编码。
    :param source_sens: 源句子List-like [sen1,sen2,....]
    :param target_sens: 目标句子List-like [sen1,sen2,....]
    :param topk:拿着源句子从目标句子库里选取topk最相似的
    :param metric: cosine 或者 euclidean
    :param use_faiss: True or False or auto
    :param src_sen_encoder: 专门编码源句子的编码器
    :param tar_sen_encoder: 专门编码目标句子的编码器
    :return: [[sen,topk sens,topk sens's similarity],..]
    """
    print("get sens vec...")
    if metric == "cosine":
        source_vecs = normalize(sen_encoder.get_sens_vec(source_sens, **src_kwargs), axis=1)
        vec_db = VectorDataBase(normalize(sen_encoder.get_sens_vec(target_sens, **tar_kwargs), axis=1))
    else:
        source_vecs = sen_encoder.get_sens_vec(source_sens, **src_kwargs)
        vec_db = VectorDataBase(sen_encoder.get_sens_vec(target_sens, **tar_kwargs))
    print("get search res...")
    res_index, res_distance = find_topk_by_vecs(source_vecs, vec_db, topk, metric, use_faiss)
    ### format data
    data = []
    for i in range(res_index.shape[0]):
        t = [source_sens[i], [], []]
        for j in range(res_index.shape[1]):
            t[1].append(target_sens[res_index[i, j]])
            t[2].append(res_distance[i, j])
        data.append(t)
    return data
