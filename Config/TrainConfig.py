from Config.AConfig import AConfig


class TrainConfig(AConfig):
    def __init__(self):
        # 训练相关
        self.num_epoch = 10
        self.batch_size = 32  #
        self.model_batch_size = 512  #
        self.lr = 5e-5  #
        self.warmup_proportion = 0.1  #
        self.log_step = 20  #
        self.adva_type = "none"  # None or fgm
        self.loss_type = "normal"  # batch_neg,hard_neg
        self.score_ratio = 20.0  # 余弦得分过小，需要放大
        self.pool = "cls"  # 句向量方式
        # 模型相关
        self.query_pretrained_model_dir = ""
        self.query_model_type = "roformer"  # 基本确定是roformer了
        self.title_pretrained_model_dir = ""
        self.title_model_type = "roformer"  # 基本确定是roformer了
        self.device = "0"
        self.share_weight = True
        # 评估相关
        self.eval_step = 1000
        self.print_input_step = 200  # 多少步输出一次输入信息
        self.eval_metrics = ["mrr10", "top10acc"]
        # 数据相关
        self.corpus_path = "./model_data/sim_data"
        self.corpus_path_for_test = "./model_data/sim_data"
        self.topk_path = "./model_data/sim_data"
        self.data_dir = "./model_data/sim_data"
        self.title_max_len = 64  # 最大长度
        self.query_max_len = 64  # 最大长度
        self.num_hard_neg = 64  # 最大长度
        self.num_easy_neg = 64  # 最大长度
        self.use_topk = 4  # 最大长度
        # 模型保存相关
        self.output_dir = "./output/test1"
        self.seed = ""
        self.save_times_per_epoch = -1
