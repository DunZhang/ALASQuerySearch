import numpy as np
import torch
import torch.nn as nn
from Model.AModel import AModel
from Config.TrainConfig import TrainConfig
from transformers import BertConfig, BertTokenizer, BertModel
from transformers import ConvBertConfig, ConvBertTokenizer, ConvBertModel
from transformers import RoFormerConfig, RoFormerTokenizer, RoFormerModel
from Model.modeling_nezha import NeZhaModel, NeZhaConfig
from typing import Dict
from typing import List, Union
from shutil import copy
from os.path import join
from transformers import PreTrainedTokenizer
import torch.nn.functional as F
from Utils.LoggerUtil import LoggerUtil

logger = LoggerUtil.get_logger()
_MODEL_MAPPING = {
    "bert": (BertConfig, BertTokenizer, BertModel, True),
    "nezha": (NeZhaConfig, BertTokenizer, NeZhaModel, True),
    "roformer": (RoFormerConfig, RoFormerTokenizer, RoFormerModel, False),
    "conv": (ConvBertConfig, ConvBertTokenizer, ConvBertModel, False),
}


class _FCModel(torch.nn.Module):
    """ backone后的层，用于生成最终句向量 """

    def __init__(self, conf: TrainConfig, hidden_size, encoder_type):
        super().__init__()
        self.conf = conf
        self.encoder_type = encoder_type
        if self.encoder_type == "title":
            self.word_weight_fc = nn.Linear(in_features=hidden_size, out_features=1, bias=False)

    def forward(self, token_embeddings, pooler_output, attention_mask):
        if self.encoder_type == "title":
            word_weight = self.word_weight_fc(token_embeddings)  # bsz,seq_len,1
            weighed_token_embeddings = torch.mul(token_embeddings, word_weight)  # bsz,seq_len,hidden_size
            vecs = torch.mean(weighed_token_embeddings, dim=1, keepdim=False)
            vecs = F.normalize(vecs, 2.0, dim=1)
            return vecs, word_weight.squeeze(-1)
        else:
            vecs = pooler_output
            vecs = F.normalize(vecs, 2.0, dim=1)
            return vecs, None


class _VectorModel(AModel):
    def __init__(self, conf_or_model_dir: Union[str, TrainConfig], encoder_type: str):
        """
        如果conf_or_model_dir为配置类则代表从预训练模型开始加载，用于训练
        如果为conf_or_model_dir为目录路径，则从该路径进行加载，该路径下必须有之前存好的模型及相关配置文件，用于继续训练或预测
        """
        super().__init__()
        self.encoder_type = encoder_type
        # 加载config
        if isinstance(conf_or_model_dir, TrainConfig):
            self.conf = conf_or_model_dir
        else:
            self.conf = TrainConfig()
            self.conf.load(conf_path=join(conf_or_model_dir, "model_conf.json"))
        # 确定模型目录
        if encoder_type == "query":
            self.pretrained_model_dir = self.conf.query_pretrained_model_dir
            self.model_type = self.conf.query_model_type
            self.max_len = self.conf.query_max_len
        elif encoder_type == "title":
            self.pretrained_model_dir = self.conf.title_pretrained_model_dir
            self.model_type = self.conf.title_model_type
            self.max_len = self.conf.title_max_len
        else:
            raise Exception("encoder_type有误:{}".format(encoder_type))
        # 加载权重
        if isinstance(conf_or_model_dir, TrainConfig):
            # 加载预训练
            CONFIG, TOKENIZER, MODEL, self.has_pooler = _MODEL_MAPPING[self.model_type]
            self.model = MODEL.from_pretrained(self.pretrained_model_dir)
            self.tokenizer = TOKENIZER.from_pretrained(self.pretrained_model_dir)
            self.backbone_model_config = CONFIG.from_pretrained(self.pretrained_model_dir)
            self.fc_model = _FCModel(conf=self.conf, hidden_size=self.backbone_model_config.hidden_size,
                                     encoder_type=encoder_type)
        else:
            # 加载训练好的
            CONFIG, TOKENIZER, MODEL, self.has_pooler = _MODEL_MAPPING[self.model_type]
            if encoder_type == "query":
                self.tokenizer = TOKENIZER.from_pretrained(join(conf_or_model_dir, "query_backbone"))
                self.backbone_model_config = CONFIG.from_pretrained(join(conf_or_model_dir, "query_backbone"))
            else:
                self.tokenizer = TOKENIZER.from_pretrained(join(conf_or_model_dir, "title_backbone"))
                self.backbone_model_config = CONFIG.from_pretrained(join(conf_or_model_dir, "title_backbone"))

            self.model = MODEL(config=self.backbone_model_config)

            self.fc_model = _FCModel(conf=self.conf, hidden_size=self.backbone_model_config.hidden_size,
                                     encoder_type=encoder_type)

    def forward(self, ipt: Dict, **kwargs):

        # 变量进显存
        input_ids = ipt["input_ids"].to(self.get_device())
        token_type_ids = ipt["token_type_ids"].to(self.get_device())
        attention_mask = ipt["attention_mask"].to(self.get_device())

        # Step1 获取wordembed和poolerout
        if self.has_pooler:
            token_embeddings, pooler_output = self.model(input_ids=input_ids, token_type_ids=token_type_ids,
                                                         attention_mask=attention_mask)[0:2]
        else:
            token_embeddings = self.model(input_ids=input_ids, token_type_ids=token_type_ids,
                                          attention_mask=attention_mask)[0]  # bsz*seq_len*h
            pooler_output = token_embeddings[:, 0, :]  # bsz*h
        vecs, weight = self.fc_model(token_embeddings, pooler_output, attention_mask)

        return {"vecs": vecs, "weight": weight}

    def save_backbone(self, save_dir):
        """ 保存骨架网络，其实就是预训练模型 """
        save_dir = join(save_dir, "backbone")
        self.model.save_pretrained(save_dir)
        copy(join(self.pretrained_model_dir, "vocab.txt"), join(save_dir, "vocab.txt"))
        copy(join(self.pretrained_model_dir, "config.json"), join(save_dir, "config.json"))

    def save(self, save_dir):
        self.conf.save(join(save_dir, "model_conf.json"))
        torch.save(self.state_dict(), join(save_dir, "model_weight.bin"))
        self.backbone_model_config.save_pretrained(save_dir)
        self.tokenizer.save_vocabulary(save_dir)
        # self.save_backbone(save_dir)


class VectorModel(AModel):
    def __init__(self, conf_or_model_dir: Union[str, TrainConfig],
                 query_tokenizer: PreTrainedTokenizer, title_tokenizer: PreTrainedTokenizer):
        """
        如果conf_or_model_dir为配置类则代表从预训练模型开始加载，用于训练
        如果为conf_or_model_dir为目录路径，则从该路径进行加载，该路径下必须有之前存好的模型及相关配置文件，用于继续训练或预测
        """
        super().__init__()
        self.query_model = _VectorModel(conf_or_model_dir=conf_or_model_dir, encoder_type="query")
        self.title_model = _VectorModel(conf_or_model_dir=conf_or_model_dir, encoder_type="title")
        self.query_tokenizer = query_tokenizer
        self.title_tokenizer = title_tokenizer  # 暂时公用一个吧
        if isinstance(conf_or_model_dir, str):
            # 加载训练好的模型
            self.load_state_dict(torch.load(join(conf_or_model_dir, "model_weight.bin"), map_location="cpu"))

    def forward(self, ipt: Dict, **kwargs):
        query_vecs, title_vecs, title_word_weight = [], [], []
        # 获取query vector
        start = 0
        sub_ipt = ipt.get("query_ipt")
        while sub_ipt is not None and start < sub_ipt["input_ids"].shape[0]:
            input_ids = sub_ipt["input_ids"][start:start + self.query_model.conf.model_batch_size]
            token_type_ids = sub_ipt["token_type_ids"][start:start + self.query_model.conf.model_batch_size]
            attention_mask = sub_ipt["attention_mask"][start:start + self.query_model.conf.model_batch_size]
            query_vecs.append(
                self.query_model(
                    {"input_ids": input_ids, "token_type_ids": token_type_ids, "attention_mask": attention_mask}
                )["vecs"]
            )
            start += self.query_model.conf.model_batch_size
        if len(query_vecs) > 0:
            query_vecs = torch.cat(query_vecs, dim=0)
        # 获取title vector
        start = 0
        sub_ipt = ipt.get("title_ipt")
        while sub_ipt is not None and start < sub_ipt["input_ids"].shape[0]:
            input_ids = sub_ipt["input_ids"][start:start + self.query_model.conf.model_batch_size]
            token_type_ids = sub_ipt["token_type_ids"][start:start + self.query_model.conf.model_batch_size]
            attention_mask = sub_ipt["attention_mask"][start:start + self.query_model.conf.model_batch_size]
            output = self.title_model(
                {"input_ids": input_ids, "token_type_ids": token_type_ids, "attention_mask": attention_mask})
            title_vecs.append(output["vecs"])
            title_word_weight.append(output["weight"])
            start += self.query_model.conf.model_batch_size
        if len(title_vecs) > 0:
            title_vecs = torch.cat(title_vecs, dim=0)
            title_word_weight = torch.cat(title_word_weight, dim=0)
        return {"query_vecs": query_vecs, "title_vecs": title_vecs, "title_word_weight": title_word_weight}

    def get_sens_vec(self, sens: List[str], sen_type: str = Union["query", "title"]):
        self.eval()
        all_vecs = []
        start = 0
        if sen_type == "query":
            tokenizer = self.query_tokenizer
            max_len = self.query_model.conf.query_max_len
        else:
            tokenizer = self.title_tokenizer
            max_len = self.query_model.conf.title_max_len
        with torch.no_grad():
            while start < len(sens):
                logger.info("get sentences vector: {}/{},\t{}".format(start, len(sens), start / len(sens)))
                print("get sentences vector: {}/{},\t{}".format(start, len(sens), start / len(sens)))
                ipt = tokenizer.batch_encode_plus(sens[start:start + 256], add_special_tokens=True,
                                                  padding="longest",
                                                  truncation="longest_first",
                                                  max_length=max_len,
                                                  return_tensors="pt",
                                                  return_token_type_ids=True, return_attention_mask=True)
                if sen_type == "query":
                    vecs = self({"query_ipt": ipt})["query_vecs"].to("cpu").numpy()
                else:
                    vecs = self({"title_ipt": ipt})["title_vecs"].to("cpu").numpy()
                all_vecs.append(vecs)
                start += 256
        self.train()
        return np.vstack(all_vecs)

    def save(self, save_dir):
        self.query_model.conf.save(join(save_dir, "model_conf.json"))
        torch.save(self.state_dict(), join(save_dir, "model_weight.bin"))
        self.query_tokenizer.save_vocabulary(join(save_dir, "query_backbone"))
        self.title_tokenizer.save_vocabulary(join(save_dir, "title_backbone"))
        self.query_model.backbone_model_config.save_pretrained(join(save_dir, "query_backbone"))
        self.title_model.backbone_model_config.save_pretrained(join(save_dir, "title_backbone"))
