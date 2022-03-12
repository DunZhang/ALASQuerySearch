import json
from typing import Tuple, Dict
from torch.optim import Optimizer
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler

from Config.TrainConfig import TrainConfig
from Evaluator.AEvaluator import AEvaluator
from InfoLogger.AInfoLogger import AInfoLogger
from InfoLogger.VectorInfoLogger import VectorInfoLogger
from LossModel.ALossModel import ALossModel
from LossModel.VectorLossModel import VectorLossModel
from Model.AModel import AModel
from ModelSaver.AModelSaver import AModelSaver
from ModelSaver.GeneralModelSaver import GeneralModelSaver
from Trainer.ATrainer import ATrainer
from DataLoader.VectorDataSet import collect_fn, VectorDataSet
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
import os
from Model.VectorModel import VectorModel
from Evaluator.VectorEvaluator import VectorEvaluator
from Utils.LoggerUtil import LoggerUtil
from Utils.Adversarial import FGM, PGD
from Utils.DataUtil import DataUtil
from transformers import RoFormerTokenizer
import gc

logger = LoggerUtil.get_logger()


class VectorTrainer(ATrainer):
    def __init__(self, conf: TrainConfig):
        super().__init__(conf)
        self.conf = conf
        self.seed_everything(seed=conf.seed)

    def get_data_loader(self) -> DataLoader:
        if self.conf.query_model_type == "roformer":
            tokenizer = RoFormerTokenizer.from_pretrained(self.conf.query_pretrained_model_dir)
        else:
            tokenizer = BertTokenizer.from_pretrained(self.conf.query_pretrained_model_dir)
        self.query_tokenizer = tokenizer
        if self.conf.title_model_type == "roformer":
            tokenizer = RoFormerTokenizer.from_pretrained(self.conf.title_pretrained_model_dir)
        else:
            tokenizer = BertTokenizer.from_pretrained(self.conf.title_pretrained_model_dir)
        self.title_tokenizer = tokenizer
        # 读取公用数据
        with open(self.conf.corpus_path, "r", encoding="utf8") as fr:
            self.all_corpus_sens = [DataUtil.clean_line(line)[1] for line in fr]
        lens = [len(i) for i in self.all_corpus_sens]
        logger.info("Title 平均长度：{}".format(sum(lens) / len(lens)))
        with open(self.conf.topk_path, "r", encoding="utf8") as fr:
            self.topk_data = {k: [str(i) for i in v[1:self.conf.use_topk + 1]] for k, v in json.load(fr).items()}
        # train data
        train_dataset = VectorDataSet(conf=self.conf, data_path=os.path.join(self.conf.data_dir, "dev.txt"),
                                      data_type="train", tokenizer=tokenizer, all_corpus_sens=self.all_corpus_sens,
                                      query2topk=self.topk_data)
        sampler = RandomSampler(data_source=train_dataset)
        train_data_loader = DataLoader(dataset=train_dataset, batch_size=self.conf.batch_size, sampler=sampler,
                                       num_workers=2, collate_fn=collect_fn, pin_memory=True, drop_last=True)

        return train_data_loader

    def get_model(self) -> AModel:
        model = VectorModel(conf_or_model_dir=self.conf, query_tokenizer=self.query_tokenizer,
                            title_tokenizer=self.title_tokenizer)
        if self.conf.share_weight:
            del model.title_model.model
            model.title_model.model = model.query_model.model
        return model

    def get_evaluator(self) -> AEvaluator:
        return VectorEvaluator(conf=self.conf)

    def get_loss_model(self) -> ALossModel:
        return VectorLossModel(conf=self.conf)

    def get_model_saver(self) -> AModelSaver:
        return GeneralModelSaver(conf=self.conf)

    def get_optimizer(self, model: AModel) -> Optimizer:
        """
        因为是BERT，就默认用adam了
        :param model:
        :return:
        """
        no_decay = ["bias", "LayerNorm.weight"]
        paras = dict(model.named_parameters())
        optimizer_grouped_parameters = [{
            "params": [p for n, p in paras.items() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
            {"params": [p for n, p in paras.items() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.conf.lr)
        return optimizer

    def get_step_info(self) -> Tuple[int, int]:
        return self.conf.num_epoch, len(self.train_data_loader.dataset) // self.conf.batch_size

    def get_lr_scheduler(self, optimizer: Optimizer) -> torch.optim.lr_scheduler._LRScheduler:
        total_steps = self.num_epoch * self.epoch_steps
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=int(self.conf.warmup_proportion * total_steps),
                                                    num_training_steps=total_steps)
        return scheduler

    def get_info_logger(self) -> AInfoLogger:
        return VectorInfoLogger(conf=self.conf)

    def get_device(self) -> torch.device:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = self.conf.device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return device

    def train(self):
        logger.info("模型架构信息：\n{}".format(self.model))
        self.model = self.model.to(self.device)
        ####################################################
        for name, weight in self.model.named_parameters():
            if any([i in name for i in ["layer.10.", "layer.11.", "fc_model"]]):
                logger.info("train parameter: {}".format(name))
                weight.requires_grad = True
            else:
                weight.requires_grad = False
        ###################################################
        self.loss_model = self.loss_model.to(self.device)
        if self.conf.adva_type.startswith("fgm"):
            logger.info("使用fgm对抗训练")
            fgm = FGM(self.model)
            fgm_eps = float(self.conf.adva_type.split("_")[1])
        elif self.conf.adva_type.startswith("pgd"):
            logger.info("使用pgd对抗训练")
            K = int(self.conf.adva_type.split("_")[1])
            pgd = PGD(self.model)
        # train
        global_step = 1
        logger.info("start train")
        for epoch in range(self.num_epoch):
            for step, ipt in enumerate(self.train_data_loader):
                global_step += 1
                model_output = self.model(ipt)
                loss = self.loss_model(model_output, ipt)
                # 对抗训练
                if self.conf.adva_type == "fgm":
                    loss.backward()
                    fgm.attack(epsilon=fgm_eps)  # 在embedding上添加对抗扰动
                    model_output = self.model(ipt)
                    loss_adv = self.loss_model(model_output, ipt)
                    loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                    fgm.restore()  # 恢复embedding参数
                elif self.conf.adva_type.startswith("pgd"):
                    loss.backward()
                    pgd.backup_grad()
                    for t in range(K):
                        pgd.attack(is_first_attack=(t == 0))  # 在embedding上添加对抗扰动, first attack时备份param.data
                        if t != K - 1:
                            self.model.zero_grad()
                        else:
                            pgd.restore_grad()
                        model_output = self.model(ipt)
                        loss_adv = self.loss_model(model_output, ipt)
                        loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                    pgd.restore()  # 恢复embedding参数
                else:
                    loss.backward()
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                # 对抗训练
                # 梯度累计
                # 梯度下降，更新参数
                self.optimizer.step()
                self.lr_scheduler.step()
                # 把梯度置0
                self.model.zero_grad()
                self.optimizer.zero_grad()

                # 如果符合条件则会进行模型评估
                eval_result = self.evaluator.try_evaluate(model=self.model,
                                                          data_path=os.path.join(self.conf.data_dir, "dev.txt"),
                                                          corpus_path=self.conf.corpus_path_for_test,
                                                          step=step,
                                                          global_step=global_step,
                                                          epoch_steps=self.epoch_steps,
                                                          num_epochs=self.num_epoch)
                # 如果符合条件则会保存模型
                self.model_saver.try_save_model(model=self.model, step=step, global_step=global_step,
                                                epoch_steps=self.epoch_steps,
                                                num_epochs=self.num_epoch, eval_result=eval_result)
                # 如果符合条件则会输出相关信息
                self.info_logger.try_print_log(loss=loss, eval_result=eval_result, step=step, global_step=global_step,
                                               epoch_steps=self.epoch_steps, num_epochs=self.num_epoch, epoch=epoch + 1,
                                               ipt=ipt, tokenizer=self.query_tokenizer)
