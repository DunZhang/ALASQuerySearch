import random
from Utils.DataUtil import DataUtil

if __name__ == "__main__":
    train_path = "../model_data/hold_out/dev.txt"
    dev_path = "../model_data/hold_out/train.txt"
    corpus_path = "../model_data/ori_data/corpus.tsv"
    save_path = "mini_corpus.tsv"
    data = []
    with open(train_path, "r", encoding="utf8") as fr:
        for line in fr:
            ss = line.strip().split("\t")  # query,title
            data.append(ss)
    with open(dev_path, "r", encoding="utf8") as fr:
        for line in fr:
            ss = line.strip().split("\t")  # query,title
            data.append(ss)
    label_sens = [i[1] for i in data]
    with open(corpus_path, "r", encoding="utf8") as fr:
        all_corpus_sens = [DataUtil.clean_line(line)[1] for line in fr]
    all_corpus_sens = random.sample(all_corpus_sens, 100000) + label_sens
    all_corpus_sens = list(set(all_corpus_sens))
    with open(save_path, "w", encoding="utf8") as fw:
        fw.writelines([i + "\n" for i in all_corpus_sens])
