import pytest

from yolox.utils.lr_scheduler import LRScheduler


class TestCosineScheduler:
    def test_starts_at_lr_and_decreases(self):
        lr = 0.01
        iters_per_epoch = 100
        total_epochs = 10
        scheduler = LRScheduler("cos", lr, iters_per_epoch, total_epochs)

        first_lr = scheduler.update_lr(0)
        last_lr = scheduler.update_lr(iters_per_epoch * total_epochs - 1)
        assert first_lr == pytest.approx(lr, abs=1e-5)
        assert last_lr < first_lr
        assert last_lr == pytest.approx(0.0, abs=1e-3)

    def test_monotonically_decreasing(self):
        lr = 0.01
        scheduler = LRScheduler("cos", lr, 100, 10)
        lrs = [scheduler.update_lr(i) for i in range(0, 1000, 50)]
        for i in range(1, len(lrs)):
            assert lrs[i] <= lrs[i - 1] + 1e-7


class TestWarmCosineScheduler:
    def test_warmup_phase_increases(self):
        lr = 0.01
        scheduler = LRScheduler(
            "warmcos", lr, iters_per_epoch=100, total_epochs=50, warmup_epochs=5
        )
        warmup_iters = 5 * 100
        # LR at start of warmup should be less than LR at end of warmup
        lr_start = scheduler.update_lr(0)
        lr_end_warmup = scheduler.update_lr(warmup_iters - 1)
        assert lr_start < lr_end_warmup


class TestYoloxWarmCosineScheduler:
    def test_warmup_cosine_and_constant_tail(self):
        lr = 0.01
        total_epochs = 100
        iters_per_epoch = 100
        no_aug_epochs = 15
        warmup_epochs = 5
        min_lr_ratio = 0.05

        scheduler = LRScheduler(
            "yoloxwarmcos",
            lr,
            iters_per_epoch,
            total_epochs,
            warmup_epochs=warmup_epochs,
            no_aug_epochs=no_aug_epochs,
            min_lr_ratio=min_lr_ratio,
        )

        # Warmup: LR should increase
        lr_start = scheduler.update_lr(0)
        lr_end_warmup = scheduler.update_lr(warmup_epochs * iters_per_epoch - 1)
        assert lr_start < lr_end_warmup

        # Constant tail: last no_aug_epochs should have constant LR
        tail_start = (total_epochs - no_aug_epochs) * iters_per_epoch
        lr_tail_1 = scheduler.update_lr(tail_start)
        lr_tail_2 = scheduler.update_lr(tail_start + iters_per_epoch)
        assert lr_tail_1 == pytest.approx(lr_tail_2, abs=1e-7)


class TestMultistepScheduler:
    def test_lr_drops_at_milestones(self):
        lr = 0.1
        scheduler = LRScheduler(
            "multistep",
            lr,
            iters_per_epoch=100,
            total_epochs=300,
            milestones=[100, 200],
            gamma=0.1,
        )

        # Before first milestone
        lr_before = scheduler.update_lr(50 * 100)
        assert lr_before == pytest.approx(lr, abs=1e-5)

        # After first milestone
        lr_after_1 = scheduler.update_lr(150 * 100)
        assert lr_after_1 == pytest.approx(lr * 0.1, abs=1e-5)

        # After second milestone
        lr_after_2 = scheduler.update_lr(250 * 100)
        assert lr_after_2 == pytest.approx(lr * 0.01, abs=1e-5)


class TestInvalidScheduler:
    def test_invalid_name_raises(self):
        with pytest.raises(ValueError):
            LRScheduler("invalid_name", 0.01, 100, 10)
