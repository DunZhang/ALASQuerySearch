from Utils.LoggerUtil import LoggerUtil
import re
from typing import List
import torch

logger = LoggerUtil.get_logger()


class DataUtil():
    @staticmethod
    def clean_line(line: str):
        line = line.strip()
        idx = line.find("\t")
        sen_id = line[:idx].strip()
        sen = line[idx + 1:]
        return [sen_id, re.sub("\s", "", sen)]

    @staticmethod
    def get_bert_ipt(input_ids_list: List[List[int]]):
        max_len = max([len(i) for i in input_ids_list])
        attention_mask = [[1] * len(i) + [0] * (max_len - len(i)) for i in input_ids_list]
        input_ids = [i + [0] * (max_len - len(i)) for i in input_ids_list]
        token_type_ids = [[0] * max_len for _ in range(len(input_ids))]
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        res = {"input_ids": input_ids, "token_type_ids": token_type_ids, "attention_mask": attention_mask}
        return res
