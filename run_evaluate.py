from Model.VectorModel import VectorModel
from Config.TrainConfig import TrainConfig
from Evaluator.VectorEvaluator import VectorEvaluator
from transformers import BertTokenizer, RoFormerTokenizer
from os.path import join
import torch

if __name__ == "__main__":
    # ------------------------------------------------------------------------------------------------------------
    # data_path = "./model_data/hold_out/dev.txt"
    # corpus_path = "./model_data/mini_corpus.tsv"
    # save_path = "char_jacc.xlsx"
    # evaluator = VectorEvaluator(conf=None)
    # res = evaluator.evaluate_set(dev_path=data_path, corpus_path=corpus_path,
    #                              save_path=save_path, use_seg=False)
    # print(res)
    # # ------------------------------------------------------------------------------------------------------------
    # data_path = "./model_data/hold_out/dev.txt"
    # corpus_path = "./model_data/ori_data/corpus.tsv"
    # save_path = "jieba_jacc.xlsx"
    # evaluator = VectorEvaluator(conf=None)
    # res = evaluator.evaluate_set(data_path=data_path, corpus_path=corpus_path, use_all_corpus=False,
    #                              save_path=save_path, use_seg=True)
    # # ------------------------------------------------------------------------------------------------------------
    model_dir = "./output/res_v1/best-top10acc"
    data_path = "./model_data/hold_out/dev.txt"
    corpus_path = "./model_data/mini_corpus.tsv"
    save_path = "v1_res.xlsx"
    metric = "euclidean"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conf = TrainConfig()
    conf.load(join(model_dir, "model_conf.json"))
    if conf.query_model_type == "roformer":
        query_tokenizer = RoFormerTokenizer(vocab_file=join(model_dir, "query_backbone/vocab.txt"))
    else:
        query_tokenizer = BertTokenizer(vocab_file=join(model_dir, "query_backbone/vocab.txt"))
    if conf.title_model_type == "roformer":
        title_tokenizer = RoFormerTokenizer(vocab_file=join(model_dir, "title_backbone/vocab.txt"))
    else:
        title_tokenizer = BertTokenizer(vocab_file=join(model_dir, "title_backbone/vocab.txt"))

    model = VectorModel(model_dir, query_tokenizer=query_tokenizer, title_tokenizer=title_tokenizer).to(device)
    evaluator = VectorEvaluator(conf=conf)
    res = evaluator.evaluate(model=model, dev_path=data_path, corpus_path=corpus_path,
                             save_path=save_path, metric=metric)
    print(model_dir, res)
