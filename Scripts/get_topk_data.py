import json
import sys

sys.path.append("..")
from Model.BERTSentenceEncoder import BERTSentenceEncoder
from Utils.VectorSearchUtil import find_topk_by_sens
from Utils.DataUtil import DataUtil
import pandas as pd
import torch
import logging

logging.basicConfig(level=logging.INFO)


def get_topk_data_v2(model: BERTSentenceEncoder, data_path: str, corpus_path: str):
    data = []
    with open(data_path, "r", encoding="utf8") as fr:
        for line in fr:
            ss = line.strip().split("\t")  # query,title
            data.append(ss)
    with open(corpus_path, "r", encoding="utf8") as fr:
        all_corpus_sens = [DataUtil.clean_line(line)[1] for line in fr]
    res = find_topk_by_sens(sen_encoder=model, source_sens=[i[0] for i in data], target_sens=all_corpus_sens,
                            topk=10, src_kwargs={}, tar_kwargs={})
    top10acc, mrr_10 = 0, 0
    for idx, (query, pred_topk, _) in enumerate(res):
        label_title = data[idx][1]
        if label_title in pred_topk:
            mrr_10 += (1 / (pred_topk.index(label_title) + 1))
            top10acc += 1
    return {"mrr10": mrr_10 / len(data), "top10acc": top10acc / len(data)}


def get_topk_data_by_vec(model: BERTSentenceEncoder, query_path: str, corpus_path: str, train_label_path: str,
                         xls_save_path: str, json_save_path: str):
    # 读取训练集
    query_data = []  # (query_id,label)
    with open(query_path, "r", encoding="utf8") as fr:
        for line in fr:
            ss = DataUtil.clean_line(line)
            query_data.append(ss)
    qid2sen, arrid2sen, arrid2qid, q2qid = {}, {}, {}, {}
    for idx, (q_id, sen) in enumerate(query_data):
        qid2sen[q_id] = sen
        arrid2sen[idx] = sen
        arrid2qid[idx] = q_id
        q2qid[sen] = q_id
    qid2docid = {}
    with open(train_label_path, "r", encoding="utf8") as fr:
        for line in fr:
            ss = DataUtil.clean_line(line)
            qid2docid[ss[0]] = ss[1]

    # 读取测试集
    doc_data, docid2sen = [], {}
    with open(corpus_path, "r", encoding="utf8") as fr:
        for line in fr:
            ss = DataUtil.clean_line(line)
            doc_data.append(ss)
            docid2sen[ss[0]] = ss[1]
    res = find_topk_by_sens(sen_encoder=model, t_source_sens=[i[1] for i in query_data],
                            t_target_sens=[i[1] for i in doc_data], topk=520)

    # [sen,topk sens,topk sens's similarity]
    save_data, json_data = [], {}
    for sen, topk_sens, topk_score in res:
        row_data = [sen, "\n".join(topk_sens[:10]), "\n".join([str(i) for i in topk_score[:10]])]  # 存10个吧那不然excel绷不住
        label_sen = docid2sen[qid2docid[q2qid[sen]]]
        row_data.append(label_sen)
        row_data.append(label_sen == topk_sens[0])
        row_data.append(label_sen in topk_sens[:10])
        row_data.append(label_sen in topk_sens[:512])
        save_data.append(row_data)
        if label_sen in topk_sens: topk_sens.remove(label_sen)
        json_data[sen] = [label_sen] + topk_sens[:512]

    df = pd.DataFrame(save_data, columns=["用户query", "TopK标题", "TopK标题得分", "标注问题", "是否等于第一名", "是否在Top10", "是否在Top512"])
    df.to_excel(xls_save_path, index=False)
    with open(json_save_path, "w", encoding="utf8") as fw:
        json.dump(json_data, fw, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    model_dir = "../model_data/public_model/roformer_chinese_sim_char_base"
    query_path = "../model_data/ori_data/train.query.txt"
    train_label_path = "../model_data/ori_data/qrels.train.tsv"
    corpus_path = "../model_data/mini_corpus.tsv"
    xls_save_path = "TopKData_base.xlsx"
    json_save_path = "../model_data/topk_vec_base.json"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BERTSentenceEncoder(pretrained_model_dir=model_dir, device=device, pooling_modes=["cls"],
                                batch_size=512, max_length=128, model_type="roformer", silence=False)
    # get_topk_data_by_vec(model=model, query_path=query_path, corpus_path=corpus_path, train_label_path=train_label_path,
    #                      xls_save_path=xls_save_path, json_save_path=json_save_path)
    res = get_topk_data_v2(model=model, data_path="../model_data/hold_out/dev.txt", corpus_path=corpus_path, )
    print(res)
