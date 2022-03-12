from typing import Dict, Union

import pandas as pd
from Model.VectorModel import VectorModel
from Config.TrainConfig import TrainConfig
from Evaluator.AEvaluator import AEvaluator
from Utils.LoggerUtil import LoggerUtil
from Utils.VectorSearchUtil import find_topk_by_sens
from Utils.DataUtil import DataUtil
import jieba

logger = LoggerUtil.get_logger()


class VectorEvaluator(AEvaluator):
    def __init__(self, conf: TrainConfig):
        super().__init__(conf=conf)
        self.conf = conf

    def get_src_tar_sens(self, dev_path, corpus_path):
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

    def evaluate(self, model: VectorModel, dev_path: str, corpus_path: str, *args, **kwargs) -> Dict:
        save_path = kwargs.get("save_path", None)
        logger.info("evaluate model...")
        source_sens, label_titles, all_corpus_sens = self.get_src_tar_sens(dev_path=dev_path, corpus_path=corpus_path)
        res = find_topk_by_sens(sen_encoder=model, source_sens=source_sens, target_sens=all_corpus_sens,
                                topk=10, src_kwargs={"sen_type": "query"}, tar_kwargs={"sen_type": "title"})
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

    def evaluate_set(self, dev_path: str, corpus_path: str, *args, **kwargs) -> Dict:
        """ 基于集合相似度 """
        save_path = kwargs.get("save_path", None)
        use_seg = kwargs.get("use_seg", False)
        logger.info("evaluate model...")
        source_sens, label_titles, all_corpus_sens = self.get_src_tar_sens(dev_path=dev_path, corpus_path=corpus_path)
        if use_seg:
            all_corpus_set = [set(list(jieba.cut(i))) for i in all_corpus_sens]
        else:
            all_corpus_set = [set(i) for i in all_corpus_sens]
        mrr_10, top10acc = 0.0, 0.0
        df_data = []
        for idx, (query, label_title) in enumerate(zip(source_sens, label_titles)):
            if idx % 100 == 0:
                print("进度", idx / len(source_sens))
            query_set = set(list(jieba.cut(query))) if use_seg else set(query)
            scores = [(title, len(query_set.intersection(i)) / len(query_set)) for title, i in
                      zip(all_corpus_sens, all_corpus_set)]
            scores.sort(key=lambda x: x[1], reverse=True)
            pred_topk = [i[0] for i in scores[:10]]
            if label_title in pred_topk:
                mrr_10 += (1 / (pred_topk.index(label_title) + 1))
                top10acc += 1
            df_data.append([query, label_title, "\n".join(pred_topk), label_title in pred_topk])
        if save_path:
            pd.DataFrame(df_data, columns=["Query", "LabelTitle", "TopK", "是否在Top10"]).to_excel(save_path, index=False)
        return {"mrr10": mrr_10 / len(source_sens), "top10acc": top10acc / len(source_sens)}

    def try_evaluate(self, model: VectorModel, data_path: str, corpus_path: str, step: int, global_step: int,
                     epoch_steps: int, num_epochs: int, *args, **kwargs) -> Union[Dict, None]:
        if self.conf.eval_step > 1 and global_step % self.conf.eval_step == 0 and data_path is not None:
            return self.evaluate(model, data_path, corpus_path, *args, **kwargs)
        else:
            return None
