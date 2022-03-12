import random
import os
import numpy as np
import torch
from abc import ABCMeta, abstractmethod
from torch.utils.data import DataLoader
from typing import Tuple
from Model.AModel import AModel
from Config.AConfig import AConfig
from Evaluator.AEvaluator import AEvaluator
from LossModel.ALossModel import ALossModel
from ModelSaver.AModelSaver import AModelSaver
from Utils.LoggerUtil import LoggerUtil
from torch.optim import Optimizer
from InfoLogger.AInfoLogger import AInfoLogger

logger = LoggerUtil.get_logger()


class ATrainer(metaclass=ABCMeta):
    def __init__(self, conf: AConfig):
        self.conf = conf

        # data loader
        self.train_data_loader = self.get_data_loader()
        # model
        self.model = self.get_model()
        # evaluator
        self.evaluator = self.get_evaluator()

        # loss model
        self.loss_model = self.get_loss_model()

        # model saver
        self.model_saver = self.get_model_saver()

        # device
        self.device = self.get_device()

        # optimizer
        self.optimizer = self.get_optimizer(model=self.model)

        # step info
        self.num_epoch, self.epoch_steps = self.get_step_info()

        #  scheuler
        self.lr_scheduler = self.get_lr_scheduler(optimizer=self.optimizer)

        # info logger
        self.info_logger = self.get_info_logger()

    @abstractmethod
    def get_data_loader(self) -> DataLoader:
        pass

    @abstractmethod
    def get_model(self, ) -> AModel:
        pass

    @abstractmethod
    def get_evaluator(self) -> AEvaluator:
        pass

    @abstractmethod
    def get_loss_model(self) -> ALossModel:
        pass

    @abstractmethod
    def get_model_saver(self) -> AModelSaver:
        pass

    @abstractmethod
    def get_device(self) -> torch.device:
        pass

    @abstractmethod
    def get_optimizer(self, model: AModel) -> Optimizer:
        pass

    @abstractmethod
    def get_step_info(self) -> Tuple[int, int]:
        """

        :return:[num_epoch, epoch_steps]
        """
        pass

    @abstractmethod
    def get_lr_scheduler(self, optimizer: Optimizer) -> torch.optim.lr_scheduler._LRScheduler:
        pass

    @abstractmethod
    def get_info_logger(self) -> AInfoLogger:
        pass

    def seed_everything(self, seed=1029):
        '''
        设置整个开发环境的seed
        :param seed:
        :param device:
        :return:
        '''
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # some cudnn methods can be random even after fixing the seed
        # unless you tell it to be deterministic
        torch.backends.cudnn.deterministic = True

    def train(self):
        self.model = self.model.to(self.device)
        self.loss_model = self.loss_model.to(self.device)
        # train
        global_step = 1
        logger.info("start train")
        for epoch in range(self.num_epoch):
            for step, ipt in enumerate(self.train_data_loader):
                global_step += 1
                model_output = self.model(ipt)
                loss = self.loss_model(model_output, ipt)
                loss.backward()
                # 梯度裁剪
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                # 对抗训练
                # 梯度累计
                # 梯度下降，更新参数
                self.optimizer.step()
                self.lr_scheduler.step()
                # 把梯度置0
                self.model.zero_grad()
                self.optimizer.zero_grad()

                # 如果符合条件则会进行模型评估
                eval_result = self.evaluator.try_evaluate(model=self.model, data_loader=self.dev_data_loader_dict,
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
                                               epoch_steps=self.epoch_steps, num_epochs=self.num_epoch, epoch=epoch + 1)
