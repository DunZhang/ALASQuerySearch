import torch
import torch.nn.functional as F
from LossModel.ALossModel import ALossModel
from Config.TrainConfig import TrainConfig
from typing import Dict
from Utils.LoggerUtil import LoggerUtil

logger = LoggerUtil.get_logger()


class VectorLossModel(ALossModel):
    def __init__(self, conf: TrainConfig):
        super().__init__(conf)
        self.conf = conf
        self.label = torch.LongTensor(list(range(self.conf.batch_size)))

    def forward(self, model_output: Dict, ipt: Dict):
        if self.conf.loss_type == "cosine":
            Q_vecs = model_output["query_vecs"]  # bsz * h
            T_vecs = model_output["title_vecs"]  # X * h
            T_label_weight = ipt["title_word_weight"].to(Q_vecs.device)  # bsz * h
            T_weight = model_output["title_word_weight"][:Q_vecs.shape[0], :]  # bsz * h
            logits = torch.mm(Q_vecs, T_vecs.t())
            label_loss = F.cross_entropy(logits * self.conf.score_ratio, self.label.to(Q_vecs.device))
            weight_loss = 10 * F.mse_loss(T_label_weight, T_weight)
            logger.info("weight_loss:{}".format(weight_loss.cpu().data))
            return label_loss + weight_loss
        elif self.conf.loss_type == "triple_loss":
            Q_vecs = model_output["query_vecs"]  # bsz * h
            T_pos_vecs = model_output["title_vecs"][:Q_vecs.shape[0], :]  # bsz * h
            T_neg_vecs = model_output["title_vecs"][Q_vecs.shape[0]:, :]  # X * h
            pos_scores = torch.sum(torch.mul(Q_vecs, T_pos_vecs), dim=1, keepdim=True)  # bsz * 1
            neg_scores = torch.mm(Q_vecs, T_neg_vecs.t())  # bsz * X
            return F.relu(neg_scores + 0.1 - pos_scores).mean()


if __name__ == "__main__":
    logits = torch.tensor([[1.0, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]])
    # print(logits.shape)
    pos_idxs, neg_idxs = [0, 1, 2, 4, 6], [3, 5, 7]
    pos_value = torch.index_select(logits, dim=0, index=torch.LongTensor(pos_idxs))[:, 1:2]
    neg_value = torch.index_select(logits, dim=0, index=torch.LongTensor(neg_idxs))[:, 1:2].t()
    #
    # print(pos)
    # print(neg)
    # print(pos - neg)
    res = torch.log(torch.sum(torch.exp(2 * (neg_value - pos_value))) + 1)
    print(res)
    res = torch.logsumexp(2 * (neg_value - pos_value), dim=0).mean()
    print(res)
