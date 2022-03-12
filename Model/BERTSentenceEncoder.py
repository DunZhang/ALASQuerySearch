import numpy as np
from transformers import BertConfig, BertTokenizer, BertModel, PreTrainedTokenizer
from transformers import ConvBertConfig, ConvBertTokenizer, ConvBertModel
from transformers import RoFormerConfig, RoFormerTokenizer, RoFormerModel
from Model.modeling_nezha import NeZhaModel, NeZhaConfig
import torch
import logging
from typing import Iterable, List
from os.path import join

_MODEL_MAPPING = {
    "bert": (BertConfig, BertTokenizer, BertModel, True),
    "nezha": (NeZhaConfig, BertTokenizer, NeZhaModel, True),
    "roformer": (RoFormerConfig, RoFormerTokenizer, RoFormerModel, False),
    "conv": (ConvBertConfig, ConvBertTokenizer, ConvBertModel, False),
}
try:
    import onnxruntime as ort
except:
    logging.warning("not find onnxruntime")
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)


class ONNXBertModel():
    def __init__(self, model_dir):
        self.ort_session = ort.InferenceSession(join(model_dir, "model.onnx"), providers=["CUDAExecutionProvider"])
        self.config = self.ONNXBertModelConfig(hidden_size=int(self.ort_session.get_outputs()[0].shape[-1]))

    def __call__(self, input_ids, attention_mask, token_type_ids, *args, **kwargs):
        return self.forward(input_ids, attention_mask, token_type_ids)

    def forward(self, input_ids, attention_mask, token_type_ids, *args, **kwargs):
        input_feed = {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids}
        res = self.ort_session.run(input_feed=input_feed, output_names=["sequence_output", "cls"])
        return res

    def train(self):
        # 为了和BertModel保持一致
        pass

    def eval(self):
        # 为了和BertModel保持一致
        pass

    class ONNXBertModelConfig():
        def __init__(self, hidden_size):
            self.hidden_size = hidden_size


class BERTSentenceEncoder():
    def __init__(self, pretrained_model_dir, device, pooling_modes: List[str] = None, batch_size=128,
                 max_length=128, silence=True, tokenizer: PreTrainedTokenizer = None,
                 use_onnx: bool = False, model_type: str = "bert"):
        """
        :param pretrained_model_dir:
            bert模型或者torch-bert模型目录或者onnx-bert模型目录，
            如果为torch-bert模型目录，需要有pytorch_model.bin, config.json和vocab.txt
            如果为onnx-bert模型目录，需要有model.onnx和vocab.txt,
            onnx模型的输入节点必须是["input_ids","attention_mask","token_type_ids"],
            输出节点必须是["sequence_output","cls"]
        :param device:
            torch.device, 中文模型才有用
        :param pooling_modes:
            List-like, the way of getting vector, support 'mean' 'cls' 'max', 中文模型才有用
        :param batch_size:
            batch_size
        :param max_length:
            max length of sentence, 中文模型才有用
        :param silence:
            whether output other info
        :param tokenizer:
            Bert tokenzier, 中文模型才有用
        :param use_onnx:
            是否使用onnx模型，如果是True，给定的模型路径需包含onnx模型, 中文模型才有用
        """
        if pooling_modes is None:
            pooling_modes = ["cls"]
        if use_onnx:
            self.model = ONNXBertModel(model_dir=pretrained_model_dir)
            self.tokenizer = tokenizer
            self.has_pooler = True
        else:
            CONFIG, TOKENIZER, MODEL, self.has_pooler = _MODEL_MAPPING[model_type]
            self.model = MODEL.from_pretrained(pretrained_model_dir).to(device)
            if tokenizer is None:
                self.tokenizer = TOKENIZER.from_pretrained(pretrained_model_dir)
            else:
                self.tokenizer = tokenizer

        self.use_onnx = use_onnx
        self.device = device
        self.pooling_modes = [i.strip().lower() for i in pooling_modes]
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_dim = self.model.config.hidden_size * len(pooling_modes)
        self.silence = silence
        self.model_type = model_type
        # logger.info("vec dim:{}".format(self.bert.config.hidden_size * len(pooling_modes)))

    def get_sens_vec(self, sens: List[str], *args, **kwargs):
        """
        get sentences vector according to pooling mode
        :param sens: List-like [sen1,sen2,sen3,...]，如果是英文词与词之间用空格隔开
        :return: ndarray:len(sens) * (hidden_size*len(pooling_modes))
        """
        if self.use_onnx:
            return self.get_sens_vec_onnx(sens)
        else:
            return self.get_sens_vec_pt(sens)

    def get_sens_vec_pt(self, sens: List[str]):
        self.model.eval()
        ### get sen vec
        all_sen_vec = []
        start = 0
        with torch.no_grad():
            while start < len(sens):
                if not self.silence:
                    logger.info("get sentences vector: {}/{},\t{}".format(start, len(sens), start / len(sens)))
                batch_data = self.tokenizer.batch_encode_plus(
                    batch_text_or_text_pairs=sens[start:start + self.batch_size], padding="longest",
                    return_tensors="pt", max_length=self.max_length,
                    truncation=True)
                batch_data = {k: v.to(self.device) for k, v in batch_data.items()}
                if self.model_type == "bert":
                    token_embeddings, pooler_output = self.model(**batch_data)[0:2]
                else:
                    token_embeddings = self.model(**batch_data)[0]
                    pooler_output = token_embeddings[:, 0, :].squeeze(1)  # bsz*h
                sen_vecs = []
                for pooling_mode in self.pooling_modes:
                    attention_mask = batch_data["attention_mask"]

                    if pooling_mode == "cls":
                        sen_vec = pooler_output
                    elif pooling_mode == "mean":
                        # get mean token sen vec
                        sen_vec = torch.mean(token_embeddings, dim=1, keepdim=False)
                        # input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                        # sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                        # sum_mask = input_mask_expanded.sum(1)
                        # sum_mask = torch.clamp(sum_mask, min=1e-9)
                        # sen_vec = sum_embeddings / sum_mask
                    elif pooling_mode == "max":
                        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                        token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
                        sen_vec = torch.max(token_embeddings, 1)[0]
                    sen_vecs.append(sen_vec)
                sen_vec = torch.cat(sen_vecs, 1)
                all_sen_vec.append(sen_vec.to("cpu").numpy())
                start += self.batch_size
        self.model.train()
        return np.vstack(all_sen_vec)

    def get_sens_vec_onnx(self, sens: List[str]):
        all_sen_vec = []
        start = 0
        while start < len(sens):
            if not self.silence:
                logger.info("get sentences vector: {}/{},\t{}".format(start, len(sens), start / len(sens)))
            batch_data = self.tokenizer.batch_encode_plus(
                batch_text_or_text_pairs=sens[start:start + self.batch_size], padding="longest",
                return_tensors="np", max_length=self.max_length,
                truncation=True)
            token_embeddings, pooler_output = self.model(**batch_data)[0:2]
            sen_vecs = []
            for pooling_mode in self.pooling_modes:
                attention_mask = batch_data["attention_mask"]
                if pooling_mode == "cls":
                    sen_vec = pooler_output
                elif pooling_mode == "mean":
                    # get mean token sen vec
                    input_mask_expanded = np.expand_dims(attention_mask, -1)
                    sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1,
                                            keepdims=False)  # bsz, hidden_size
                    sum_mask = np.sum(attention_mask, axis=1, keepdims=True)
                    sen_vec = sum_embeddings / sum_mask  # bsz, hidden_size
                elif pooling_mode == "max":
                    input_mask_expanded = np.expand_dims(attention_mask, -1)  # bsz, seq_len, 1
                    t_token_embeddings = token_embeddings * input_mask_expanded
                    t_token_embeddings[t_token_embeddings == 0] = -1e9  # Set padding tokens to large negative value
                    sen_vec = np.max(t_token_embeddings, axis=1, keepdims=False)  # bsz*h
                sen_vecs.append(sen_vec)
            sen_vec = np.hstack(sen_vecs)
            all_sen_vec.append(sen_vec)
            start += self.batch_size
        self.model.train()
        return np.vstack(all_sen_vec)


class MultiBERTSentenceEncoder():
    def __init__(self, path_list, device, pooling_modes: List[str] = None, batch_size=128, max_length=128,
                 silence=True, tokenizer: PreTrainedTokenizer = None, use_onnx: bool = False,
                 model_type_list: List[str] = None):
        """
        :param pretrained_model_or_path:
            bert模型或者torch-bert模型目录或者onnx-bert模型目录，
            如果为torch-bert模型目录，需要有pytorch_model.bin, config.json和vocab.txt
            如果为onnx-bert模型目录，需要有model.onnx和vocab.txt,
            onnx模型的输入节点必须是["input_ids","attention_mask","token_type_ids"],
            输出节点必须是["sequence_output","cls"]
        :param device:
            torch.device, 中文模型才有用
        :param pooling_modes:
            List-like, the way of getting vector, support 'mean' 'cls' 'max', 中文模型才有用
        :param batch_size:
            batch_size
        :param max_length:
            max length of sentence, 中文模型才有用
        :param silence:
            whether output other info
        :param tokenizer:
            Bert tokenzier, 中文模型才有用
        :param use_onnx:
            是否使用onnx模型，如果是True，给定的模型路径需包含onnx模型, 中文模型才有用
        """
        if pooling_modes is None:
            pooling_modes = ["cls"]
        self.models = []
        for model_path, model_type in zip(path_list, model_type_list):
            m = BERTSentenceEncoder(pretrained_model_dir=model_path, device=device, pooling_modes=pooling_modes,
                                    batch_size=batch_size, max_length=max_length,
                                    silence=silence, tokenizer=tokenizer, use_onnx=use_onnx,
                                    model_type=model_type)
            self.models.append(m)

    def get_sens_vec(self, sens: List[str]):
        """
        get sentences vector according to pooling mode
        :param sens: List-like [sen1,sen2,sen3,...]，如果是英文词与词之间用空格隔开
        :return: ndarray:len(sens) * (hidden_size*len(pooling_modes))
        """
        vecs = [m.get_sens_vec(sens=sens) for m in self.models]
        vec = vecs[0]
        for i in vecs[1:]: vec += i
        return vec / len(vecs)
