import torch
from Model.BERTSentenceEncoder import BERTSentenceEncoder
from typing import Dict
import logging
from Utils.VectorSearchUtil import find_topk_by_sens
from Utils.DataUtil import DataUtil
from transformers import RoFormerTokenizer

logging.basicConfig(level=logging.INFO)

import pandas as pd


def get_src_tar_sens(dev_path, corpus_path):
    data = []
    with open(dev_path, "r", encoding="utf8") as fr:
        for line in fr:
            ss = line.strip().split("\t")  # query,title
            data.append(ss)
    source_sens = [i[0] for i in data]
    label_titles = [i[1] for i in data]
    with open(corpus_path, "r", encoding="utf8") as fr:
        if "mini" in corpus_path:
            all_corpus_sens = [line.strip() for line in fr]
        else:
            all_corpus_sens = [DataUtil.clean_line(line)[1] for line in fr]
    return source_sens, label_titles, all_corpus_sens


def evaluate(model: BERTSentenceEncoder, dev_path: str, corpus_path: str, save_path: str, metric: str) -> Dict:
    print("评测指标:{}".format(metric))
    print("evaluate model...")
    source_sens, label_titles, all_corpus_sens = get_src_tar_sens(dev_path=dev_path, corpus_path=corpus_path)
    res = find_topk_by_sens(sen_encoder=model, source_sens=source_sens, target_sens=all_corpus_sens,
                            topk=10, src_kwargs={}, tar_kwargs={},
                            metric=metric)
    top10acc, mrr_10 = 0, 0
    df_data = []
    for idx, (query, pred_topk, _) in enumerate(res):
        label_title = label_titles[idx]
        if label_title in pred_topk:
            mrr_10 += (1 / (pred_topk.index(label_title) + 1))
            top10acc += 1
        df_data.append([query, label_title, "\n".join(pred_topk), label_title in pred_topk])
    if save_path:
        df_data.append(["mrr10", str(mrr_10 / len(source_sens)), None, None])
        df_data.append(["top10acc", str(top10acc / len(source_sens)), None, None])
        pd.DataFrame(df_data, columns=["Query", "LabelTitle", "TopK", "是否在Top10"]).to_excel(save_path, index=False)
    return {"mrr10": mrr_10 / len(source_sens), "top10acc": top10acc / len(source_sens)}


if __name__ == "__main__":
    model_dir = "./model_data/public_model/roformer_chinese_sim_char_base"
    data_path = "./model_data/hold_out/dev.txt"
    corpus_path = "./model_data/mini_corpus.tsv"
    save_path = "roformer_base_res.xlsx"
    metric = "cosine"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = RoFormerTokenizer.from_pretrained(model_dir)
    model = BERTSentenceEncoder(model_dir, model_type="roformer", silence=False, tokenizer=tokenizer, device=device)
    res = evaluate(model=model, dev_path=data_path, corpus_path=corpus_path, save_path=save_path, metric=metric)
    print(res)
