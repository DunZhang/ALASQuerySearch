import transformers

transformers.logging.set_verbosity_error()
import sys
import os
from Config.TrainConfig import TrainConfig
from Trainer.VectorTrainer import VectorTrainer
from Utils.LoggerUtil import LoggerUtil


def main(conf_path=None):
    if len(sys.argv) < 2:
        conf_path = conf_path
    else:
        conf_path = sys.argv[1]
    conf = TrainConfig()
    conf.load(conf_path=conf_path)
    LoggerUtil.init_logger("Vector", os.path.join(conf.output_dir, "logs.txt"))
    trainer = VectorTrainer(conf=conf)
    trainer.train()



if __name__ == "__main__":
    main("train_configs/test.yaml")
