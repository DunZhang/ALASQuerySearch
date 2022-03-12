import random
import torch
from torch.utils.data import Dataset
from Utils.DataUtil import DataUtil
from Utils.LoggerUtil import LoggerUtil
from transformers import PreTrainedTokenizer
from Config.TrainConfig import TrainConfig
from typing import List, Dict

logger = LoggerUtil.get_logger()


def collect_fn(batch):
    """

    :param batch:List[data_set[i]]
    :return:
    """
    num_hard, num_easy = batch[0][5:]
    all_neg_ids = []
    all_q_ids, all_label_ids, all_hard_ids, all_easy_ids, all_title_weight = [], [], [], [], []
    for item in batch: all_q_ids.append(item[0])  # query
    for item in batch: all_label_ids.append(item[1])  # label
    for item in batch: all_hard_ids.extend(item[2])  # hard neg
    for item in batch: all_easy_ids.extend(item[3])  # easy neg
    for item in batch: all_title_weight.append(item[4])  # title_weight
    all_label_hash = set(["-".join([str(j) for j in i]) for i in all_label_ids])
    new_all_hard_ids = []
    for i in all_hard_ids:
        id_hash = "-".join([str(j) for j in i])
        if id_hash not in all_label_hash: new_all_hard_ids.append(i)
    new_all_easy_ids = []
    for i in all_easy_ids:
        id_hash = "-".join([str(j) for j in i])
        if id_hash not in all_label_hash: new_all_easy_ids.append(i)
    # 选取困难负样本
    if len(new_all_hard_ids) > num_hard:
        all_neg_ids.extend(random.sample(new_all_hard_ids, num_hard))
    else:
        all_neg_ids.extend(new_all_hard_ids)
    # 选取简单负样本
    all_neg_ids.extend(random.sample(new_all_easy_ids, (num_hard + num_easy - len(all_neg_ids))))
    random.shuffle(all_neg_ids)
    all_t_ids = all_label_ids + all_neg_ids
    title_ipt = DataUtil.get_bert_ipt(all_t_ids)
    max_len = title_ipt["input_ids"].shape[1]
    all_title_weight = [i + [0.0] * (max_len - len(i)) for i in all_title_weight]
    return {"query_ipt": DataUtil.get_bert_ipt(all_q_ids),
            "title_ipt": DataUtil.get_bert_ipt(all_t_ids),
            "title_word_weight": torch.tensor(all_title_weight)
            }


class VectorDataSet(Dataset):
    def __init__(self, conf: TrainConfig, data_path: str, tokenizer: PreTrainedTokenizer, data_type: str,
                 all_corpus_sens: List[str], query2topk: Dict, **kwargs):
        """

        :param conf:
        :param data_path:
        :param tokenizer:
        :param data_type:
        :param kwargs:
        """
        self.conf = conf
        # 参数初始化
        self.tokenizer = tokenizer
        self.data_type = data_type
        self.data_path = data_path
        self.all_corpus_sens = all_corpus_sens
        self.query2topk = query2topk
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
        self.init_data_model()

    def init_data_model(self):
        """ 初始化要用到的模型数据 """
        self.data = []
        with open(self.data_path, "r", encoding="utf8") as fr:
            for line in fr:
                ss = line.strip().split("\t")  # (query,label)
                assert len(ss) == 2
                self.data.append(ss)
        if self.data_type == "train": random.shuffle(self.data)
        logger.info("总数据{}条".format(len(self.data)))
        query_lens = [len(i[0]) for i in self.data]
        logger.info("query 平均长度：{}".format(sum(query_lens) / len(query_lens)))
        # 做一些数量上的计算
        self.num_hard = self.conf.batch_size * self.conf.num_hard_neg
        self.num_easy = self.conf.num_easy_neg  # 简单负例是随机选取的可以公用

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        """
        item 为数据索引，迭代取第item条数据
        """
        # 获取目标数据
        query, label = self.data[item]
        # 获取neg
        hard_neg = random.sample(self.query2topk[query], self.conf.num_hard_neg + 1)  # 多选一些避免重复
        easy_neg = random.sample(self.all_corpus_sens, (1 + self.conf.num_easy_neg + self.num_hard))  # 多选确保可以够
        if label in easy_neg: easy_neg.remove(label)
        easy_neg = easy_neg[:self.conf.num_easy_neg]
        ############################################################################################################
        # query += ("你" * 100)
        # label += ("你" * 100)
        # hard_neg = [i + ("你" * 100) for i in hard_neg]
        # easy_neg = [i + ("你" * 100) for i in easy_neg]
        ############################################################################################################
        if len(query) > self.conf.query_max_len: query = query[:self.conf.query_max_len]
        if len(label) > self.conf.title_max_len: label = label[:self.conf.title_max_len]
        hard_neg = [i if len(i) < self.conf.title_max_len else i[:self.conf.title_max_len] for i in hard_neg]
        easy_neg = [i if len(i) < self.conf.title_max_len else i[:self.conf.title_max_len] for i in easy_neg]
        ipt_id_list = []
        query_ids = self.tokenizer.encode(text=query, add_special_tokens=True, padding=False, truncation=False,
                                          return_tensors=None, max_length=None)
        query_ids_set = set(query_ids)
        ipt_id_list.append(query_ids)
        label_title_ids = self.tokenizer.encode(text=label, add_special_tokens=True, padding=False, truncation=False,
                                                return_tensors=None, max_length=None)
        title_weight_label = [1.0 if i in query_ids_set else 0.0 for i in label_title_ids]
        ipt_id_list.append(label_title_ids)
        hard_ids = [
            self.tokenizer.encode(text=i, add_special_tokens=True, padding=False, truncation=False,
                                  return_tensors=None, max_length=None)
            for i in hard_neg]
        easy_ids = [
            self.tokenizer.encode(text=i, add_special_tokens=True, padding=False, truncation=False,
                                  return_tensors=None, max_length=None)
            for i in easy_neg]
        ipt_id_list.append(hard_ids)
        ipt_id_list.append(easy_ids)
        ipt_id_list.append(title_weight_label)
        return ipt_id_list + [self.num_hard, self.num_easy]
