"""Tier A: async checkpoints write the same values as the original synchronous save; persistent val workers."""

import types

import torch

from yolox.core.trainer import Trainer
from yolox.models import YOLOPAFPN, YOLOX, YOLOXHead

from .conftest import flags


def fake_trainer(tmp, model, opt):
    t = types.SimpleNamespace(rank=0, use_model_ema=False, model=model, optimizer=opt, best_ap=0.5, epoch=3,
                              file_name=str(tmp), args=types.SimpleNamespace(logger="tensorboard"))
    t._join_ckpt_writer = types.MethodType(Trainer._join_ckpt_writer, t)
    t.save_ckpt = types.MethodType(Trainer.save_ckpt, t)
    return t


def flat(x, prefix=""):
    if torch.is_tensor(x):
        return {prefix: x}
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            out.update(flat(v, f"{prefix}/{k}"))
        return out
    if isinstance(x, (list, tuple)):
        out = {}
        for i, v in enumerate(x):
            out.update(flat(v, f"{prefix}/{i}"))
        return out
    return {prefix: x}


def test_async_ckpt_same_values(tmp_path):
    torch.manual_seed(0)
    model = YOLOX(YOLOPAFPN(0.33, 0.25), YOLOXHead(5, 0.25))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.rand(2, 3, 128, 128)
    model.train()
    out = model(x, torch.zeros(2, 10, 5))
    out["total_loss"].backward()
    opt.step()
    a, b = tmp_path / "sync", tmp_path / "async"
    with flags(""):
        fake_trainer(a, model, opt).save_ckpt("latest", update_best_ckpt=True, ap=0.4)
    with flags("async_ckpt"):
        t = fake_trainer(b, model, opt)
        t.save_ckpt("latest", update_best_ckpt=True, ap=0.4)
        with torch.no_grad():  # training continues while the writer runs: the snapshot must not change
            for p in model.parameters():
                p.add_(1.0)
        t._join_ckpt_writer()
        with torch.no_grad():
            for p in model.parameters():
                p.sub_(1.0)
    for name in ["latest_ckpt.pth", "best_ckpt.pth"]:
        fa = flat(torch.load(a / name, weights_only=False))
        fb = flat(torch.load(b / name, weights_only=False))
        assert fa.keys() == fb.keys()
        for k in fa:
            if torch.is_tensor(fa[k]):
                assert torch.equal(fa[k], fb[k]), k
            else:
                assert fa[k] == fb[k], k


def test_eval_loader_persistent(tmp_path, coco_dir):
    from yolox.exp import Exp

    e = Exp()
    e.data_dir = str(coco_dir)
    e.val_ann = "train.json"
    e.data_num_workers = 2

    def ds(**kw):
        from yolox.data import COCODataset, ValTransform
        return COCODataset(data_dir=str(coco_dir), json_file="train.json", name="train", img_size=(320, 320),
                           preproc=ValTransform())
    e.get_eval_dataset = ds
    with flags(""):
        assert not e.get_eval_loader(4, False).persistent_workers
    with flags("eval_persistent"):
        assert e.get_eval_loader(4, False).persistent_workers
