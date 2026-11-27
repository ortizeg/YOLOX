import os

import pytest
import torch

pytestmark = pytest.mark.integration


def coco_data_available():
    data_dir = os.getenv("YOLOX_DATADIR", None)
    if data_dir is None:
        return False
    ann_path = os.path.join(data_dir, "COCO", "annotations", "instances_val2017.json")
    return os.path.exists(ann_path)


skip_no_coco = pytest.mark.skipif(
    not coco_data_available(), reason="COCO data not available"
)


@pytest.fixture(scope="module")
def data_dir():
    return os.environ["YOLOX_DATADIR"]


@pytest.fixture(scope="module")
def coco_dataset(data_dir):
    pytest.importorskip("pycocotools")
    from yolox.data import COCODataset, ValTransform

    preproc = ValTransform(swap=(2, 0, 1), legacy=False)
    dataset = COCODataset(
        data_dir=os.path.join(data_dir, "COCO"),
        json_file="instances_val2017.json",
        img_size=(640, 640),
        preproc=preproc,
        cache=False,
    )
    return dataset


@skip_no_coco
class TestCOCODatasetLoading:
    """Integration tests for COCO dataset loading and evaluation."""

    def test_dataset_loads_and_has_samples(self, coco_dataset):
        """COCO dataset loads successfully and has > 0 samples."""
        assert len(coco_dataset) > 0

    def test_single_sample_format(self, coco_dataset):
        """Single sample returns (image_tensor, target_array) with correct types."""
        sample = coco_dataset[0]
        assert isinstance(sample, tuple), "Sample should be a tuple"
        assert len(sample) >= 2, "Sample should have at least (image, target)"

        image, target = sample[0], sample[1]

        # Image should be a torch Tensor or numpy array with 3 channels
        if isinstance(image, torch.Tensor):
            assert image.ndim == 3, "Image tensor should have 3 dimensions (C, H, W)"
            assert image.shape[0] == 3, "Image should have 3 channels"
        else:
            # numpy array — channels may be first or last depending on preproc
            assert image.ndim == 3, "Image array should have 3 dimensions"

        # Target should be array-like (annotations for the image)
        assert target is not None, "Target should not be None"

    def test_single_batch_eval_yolox_s(self, coco_dataset):
        """Single batch evaluation with yolox-s model does not crash."""
        from yolox.exp import get_exp

        exp = get_exp(None, "yolox-s")
        model = exp.get_model()
        model.eval()

        image, target = coco_dataset[0][0], coco_dataset[0][1]
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image)
        # Add batch dimension
        image = image.unsqueeze(0).float()

        with torch.no_grad():
            outputs = model(image)

        assert outputs is not None, "Model should produce output"
