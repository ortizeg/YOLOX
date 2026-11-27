"""exp.no_aug_eval_interval: the no-aug phase sets eval_interval from it (default 1 = original YOLOX behaviour)."""
from types import SimpleNamespace

from yolox.core.trainer import Trainer
from yolox.exp import Exp


class _Loader:
    def close_mosaic(self):
        pass


def _trainer(exp):
    t = Trainer.__new__(Trainer)
    t.exp, t.epoch, t.max_epoch, t.no_aug, t.is_distributed = exp, exp.max_epoch - exp.no_aug_epochs - 1, exp.max_epoch, True, False
    t.train_loader = _Loader()
    t.model = SimpleNamespace(head=SimpleNamespace(use_l1=False))
    t.save_ckpt = lambda *a, **k: None
    return t


def test_default_keeps_original_behaviour():
    exp = Exp()
    assert exp.no_aug_eval_interval == 1
    exp.eval_interval = 1000
    Trainer.before_epoch(_trainer(exp))
    assert exp.eval_interval == 1


def test_override_keeps_eval_off():
    exp = Exp()
    exp.merge(["eval_interval", "1000", "no_aug_eval_interval", "1000"])
    Trainer.before_epoch(_trainer(exp))
    assert exp.eval_interval == 1000
