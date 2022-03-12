from Config.FAQSimConfig import FAQSimConfig
from Model.FAQSimModel import FAQSimModel
from torch.utils.data import DataLoader, SequentialSampler
from DataLoader.FAQSimDataSet import FAQSimDataSet, collect_fn
import numpy as np
import os
import torch
import pandas as pd
from transformers import BertTokenizer, RoFormerTokenizer
from os.path import join
import random


class Predictor():
    def __init__(self, model_dir):
        self.conf = FAQSimConfig()
        self.conf.load(join(model_dir, "model_conf.json"))
        self.model_dir = model_dir

    def get_pred_res(self, file_path: str, device: str):
        # device
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # data
        if self.conf.model_type == "roformer":
            tokenizer = RoFormerTokenizer.from_pretrained(self.conf.pretrained_model_dir)
        else:
            tokenizer = BertTokenizer.from_pretrained(self.conf.pretrained_model_dir)
        # tokenizer = BertTokenizer.from_pretrained(self.conf.pretrained_model_dir)
        tokenizer.add_tokens(["[unused{}]".format(i) for i in range(1, 100)], special_tokens=True)

        role_dataset = FAQSimDataSet(conf=self.conf, data_dir=file_path, tokenizer=tokenizer, data_type="test")
        sampler = SequentialSampler(data_source=role_dataset)
        data_loader = DataLoader(dataset=role_dataset, batch_size=self.conf.batch_size, sampler=sampler,
                                 num_workers=1, collate_fn=collect_fn, pin_memory=False, drop_last=False)
        # model
        model = FAQSimModel(conf=self.conf)
        model.load(self.model_dir)
        model = model.to(device)
        model.eval()
        # predict
        logits = []
        with torch.no_grad():
            for idx, ipt in enumerate(data_loader):
                batch_logits = model(ipt)["logits"].to("cpu").numpy()
                logits.append(batch_logits)

        logits = np.vstack(logits)  # n * 3
        return logits

    def predict(self, file_path: str, device: str, save_path: str):
        logits = self.get_pred_res(file_path=file_path, device=device)  # n * 3
        pred = np.argmax(logits, axis=1).tolist()
        with open(save_path, "w", encoding="utf8") as fw:
            fw.writelines([str(int(i)) + "\n" for i in pred])
