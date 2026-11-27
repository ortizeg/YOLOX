import os
from yolox.exp import Exp as MyExp


class Exp(MyExp):
    """Identical to yolox_s_run20_match but capped at 2 epochs for validation."""

    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.num_classes = 80
        self.max_epoch = 2
        self.warmup_epochs = 1
        self.no_aug_epochs = 0
        self.input_size = (640, 640)
        self.test_size = (640, 640)
        self.data_num_workers = 8
        self.eval_interval = 10  # no eval during 2 epochs
        self.mosaic_prob = 1.0
        self.mixup_prob = 1.0
        self.hsv_prob = 1.0
        self.flip_prob = 0.5
        self.degrees = 10.0
        self.translate = 0.1
        self.mosaic_scale = (0.1, 2)
        self.mixup_scale = (0.5, 1.5)
        self.shear = 2.0
        self.enable_mixup = True
        self.multiscale_range = 5
        self.basic_lr_per_img = 0.01 / 64.0
        self.momentum = 0.9
        self.weight_decay = 5e-4
        self.warmup_lr = 0
        self.min_lr_ratio = 0.05
        self.scheduler = "yoloxwarmcos"
        self.test_conf = 0.01
        self.nmsthre = 0.65
        self.ema = True
        self.print_interval = 10
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
