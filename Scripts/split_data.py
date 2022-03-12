import os.path
import random

random.seed(1)
from Utils.DataUtil import DataUtil


def hodl_out(query_path: str, corpus_path: str, train_label_path: str, save_dir):
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

    all_data = [(sen, docid2sen[qid2docid[q_id]]) for q_id, sen in query_data]
    all_data = ["{}\t{}\n".format(*i) for i in all_data]
    with open(os.path.join(save_dir, "train.txt"), "w", encoding="utf8") as fw:
        fw.writelines(all_data[10000:])
    with open(os.path.join(save_dir, "dev.txt"), "w", encoding="utf8") as fw:
        fw.writelines(all_data[:10000])


if __name__ == "__main__":
    model_dir = "../model_data/public_model/roformer_chinese_sim_char_base"
    query_path = "../model_data/ori_data/train.query.txt"
    train_label_path = "../model_data/ori_data/qrels.train.tsv"
    corpus_path = "../model_data/ori_data/corpus.tsv"
    save_dir = "../model_data/hold_out"
    hodl_out(query_path=query_path, corpus_path=corpus_path, train_label_path=train_label_path,
             save_dir=save_dir)
