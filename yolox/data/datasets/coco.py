#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import copy
import json
import math
import os

import cv2
import numpy as np
from pycocotools.coco import COCO

from ..dataloading import get_yolox_datadir
from yolox.utils import perf
from .datasets_wrapper import CacheDataset, cache_read_img


def remove_useless_info(coco):
    """
    Remove useless info in coco dataset. COCO object is modified inplace.
    This function is mainly used for saving memory (save about 30% mem).
    """
    if isinstance(coco, COCO):
        dataset = coco.dataset
        dataset.pop("info", None)
        dataset.pop("licenses", None)
        for img in dataset["images"]:
            img.pop("license", None)
            img.pop("coco_url", None)
            img.pop("date_captured", None)
            img.pop("flickr_url", None)
        if "annotations" in coco.dataset:
            for anno in coco.dataset["annotations"]:
                anno.pop("segmentation", None)


def _anno_cache_dir(data_dir, json_file, img_size):
    """Cache folder keyed by the json's name, size and mtime and by img_size (boxes are stored pre-scaled)."""
    path = os.path.join(data_dir, "annotations", json_file)
    st = os.stat(path)
    return os.path.join(data_dir, "annotations",
                        f".annocache_{json_file}_{st.st_size}_{int(st.st_mtime)}_{img_size[0]}x{img_size[1]}")


def _write_anno_cache(d, ds):
    """Write the parsed annotations as flat arrays (atomic: temp folder, then rename)."""
    tmp = f"{d}.tmp{os.getpid()}"
    os.makedirs(tmp, exist_ok=True)
    counts = np.array([len(a[0]) for a in ds.annotations], dtype=np.int64)
    offsets = np.zeros(len(counts) + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    boxes = (np.concatenate([a[0] for a in ds.annotations], 0) if offsets[-1] else np.zeros((0, 5)))
    np.save(os.path.join(tmp, "boxes.npy"), boxes.astype(np.float64))
    np.save(os.path.join(tmp, "offsets.npy"), offsets)
    np.save(os.path.join(tmp, "img_info.npy"), np.array([a[1] for a in ds.annotations], dtype=np.int64))
    np.save(os.path.join(tmp, "resized_info.npy"), np.array([a[2] for a in ds.annotations], dtype=np.int64))
    names = [a[3].encode() for a in ds.annotations]
    np.save(os.path.join(tmp, "names.npy"), np.array(names, dtype=f"S{max(map(len, names), default=1)}"))
    json.dump({"ids": [int(i) for i in ds.ids], "class_ids": [int(c) for c in ds.class_ids],
               "cats": [{"id": c["id"], "name": c["name"]} for c in ds.cats]},
              open(os.path.join(tmp, "meta.json"), "w"))
    open(os.path.join(tmp, "complete"), "w").write("ok\n")
    try:
        os.replace(tmp, d)
    except OSError:  # another rank won the race; theirs is identical
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


class CompactAnnotations:
    """List-like view of ``COCODataset.annotations`` over memory-mapped arrays: item i is
    (boxes [n, 5] float64, (h, w), (rh, rw), file_name), the same values the parsed list holds."""

    def __init__(self, d):
        self.boxes = np.load(os.path.join(d, "boxes.npy"), mmap_mode="r")
        self.offsets = np.load(os.path.join(d, "offsets.npy"), mmap_mode="r")
        self.img_info = np.load(os.path.join(d, "img_info.npy"), mmap_mode="r")
        self.resized_info = np.load(os.path.join(d, "resized_info.npy"), mmap_mode="r")
        self.names = np.load(os.path.join(d, "names.npy"), mmap_mode="r")

    def __len__(self):
        return len(self.offsets) - 1

    def file_name(self, i):
        return self.names[i].decode()

    def __getitem__(self, i):
        s, e = int(self.offsets[i]), int(self.offsets[i + 1])
        h, w = self.img_info[i]
        rh, rw = self.resized_info[i]
        return (np.array(self.boxes[s:e]), (int(h), int(w)), (int(rh), int(rw)), self.file_name(i))


class COCODataset(CacheDataset):
    """
    COCO dataset class.
    """

    def __init__(
        self,
        data_dir=None,
        json_file="instances_train2017.json",
        name="train2017",
        img_size=(416, 416),
        preproc=None,
        cache=False,
        cache_type="ram",
    ):
        """
        COCO dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            data_dir (str): dataset root directory
            json_file (str): COCO json file name
            name (str): COCO data name (e.g. 'train2017' or 'val2017')
            img_size (int): target image size after pre-processing
            preproc: data augmentation strategy
        """
        if data_dir is None:
            data_dir = os.path.join(get_yolox_datadir(), "COCO")
        self.data_dir = data_dir
        self.json_file = json_file

        self.name = name
        self.img_size = img_size
        self.preproc = preproc
        self._coco = None
        acache = _anno_cache_dir(self.data_dir, self.json_file, img_size) if perf.on("anno_cache") else None
        if acache is not None and os.path.exists(os.path.join(acache, "complete")):
            # anno_cache hit: no JSON parse, no COCO object; arrays are memory-mapped, so every rank and forked
            # worker shares one copy through the page cache. `self.coco` still loads lazily (the evaluator uses it).
            self._init_from_anno_cache(acache, cache, cache_type)
            return
        self._coco = COCO(os.path.join(self.data_dir, "annotations", self.json_file))
        remove_useless_info(self.coco)
        self.ids = self.coco.getImgIds()
        self.num_imgs = len(self.ids)
        self.class_ids = sorted(self.coco.getCatIds())
        self._class_index = ({c: i for i, c in enumerate(self.class_ids)} if perf.on("class_map") else None)
        self.cats = self.coco.loadCats(self.coco.getCatIds())
        self._classes = tuple([c["name"] for c in self.cats])
        self.annotations = self._load_coco_annotations()
        if acache is not None:
            _write_anno_cache(acache, self)
            # use the cache from here on too: the builder must not keep the parsed COCO object (tens of GB at
            # Objects365 scale), because spawned dataloader workers each receive a pickled copy of the dataset
            self._coco = None
            del self.annotations
            self._init_from_anno_cache(acache, cache, cache_type)
            return

        path_filename = [os.path.join(name, anno[3]) for anno in self.annotations]
        super().__init__(
            input_dimension=img_size,
            num_imgs=self.num_imgs,
            data_dir=data_dir,
            cache_dir_name=f"cache_{name}",
            path_filename=path_filename,
            cache=cache,
            cache_type=cache_type
        )

    @property
    def coco(self):
        if self._coco is None:
            self._coco = COCO(os.path.join(self.data_dir, "annotations", self.json_file))
            remove_useless_info(self._coco)
        return self._coco

    @coco.setter
    def coco(self, value):
        self._coco = value

    def _init_from_anno_cache(self, d, cache, cache_type):
        meta = json.load(open(os.path.join(d, "meta.json")))
        self.ids = meta["ids"]
        self.num_imgs = len(self.ids)
        self.class_ids = meta["class_ids"]
        self._class_index = ({c: i for i, c in enumerate(self.class_ids)} if perf.on("class_map") else None)
        self.cats = meta["cats"]
        self._classes = tuple([c["name"] for c in self.cats])
        self.annotations = CompactAnnotations(d)
        path_filename = ([os.path.join(self.name, self.annotations.file_name(i)) for i in range(self.num_imgs)]
                         if cache else None)
        CacheDataset.__init__(
            self, input_dimension=self.img_size, num_imgs=self.num_imgs, data_dir=self.data_dir,
            cache_dir_name=f"cache_{self.name}", path_filename=path_filename, cache=cache, cache_type=cache_type)

    def __len__(self):
        return self.num_imgs

    def _load_coco_annotations(self):
        return [self.load_anno_from_ids(_ids) for _ids in self.ids]

    def load_anno_from_ids(self, id_):
        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]
        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
        annotations = self.coco.loadAnns(anno_ids)
        if perf.on("fast_anno"):
            return self._fast_anno(annotations, width, height, im_ann, id_)
        objs = []
        for obj in annotations:
            x1 = np.max((0, obj["bbox"][0]))
            y1 = np.max((0, obj["bbox"][1]))
            x2 = np.min((width, x1 + np.max((0, obj["bbox"][2]))))
            y2 = np.min((height, y1 + np.max((0, obj["bbox"][3]))))
            if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                obj["clean_bbox"] = [x1, y1, x2, y2]
                objs.append(obj)

        num_objs = len(objs)

        res = np.zeros((num_objs, 5))
        for ix, obj in enumerate(objs):
            cls = (self._class_index[obj["category_id"]] if self._class_index is not None
                   else self.class_ids.index(obj["category_id"]))
            res[ix, 0:4] = obj["clean_bbox"]
            res[ix, 4] = cls

        r = min(self.img_size[0] / height, self.img_size[1] / width)
        res[:, :4] *= r

        img_info = (height, width)
        resized_info = (int(height * r), int(width * r))

        file_name = (
            im_ann["file_name"]
            if "file_name" in im_ann
            else "{:012}".format(id_) + ".jpg"
        )

        return (res, img_info, resized_info, file_name)

    def _fast_anno(self, annotations, width, height, im_ann, id_):
        """``load_anno_from_ids`` with Python ``max``/``min`` on scalars instead of ``np.max``/``np.min`` on tuples
        (each of those builds an array; ~6 per object, minutes at Objects365 scale). Same IEEE double maths, same
        filter, same array. An annotation with a non-finite value takes the original numpy expression, so NaN
        handling is unchanged. ``clean_bbox`` is still written back into the COCO dicts as before."""
        isfinite = math.isfinite
        rows = []
        for obj in annotations:
            b = obj["bbox"]
            b0, b1, b2, b3 = b[0], b[1], b[2], b[3]
            # -0.0 also takes the numpy path: max(0, -0.0) keeps whichever zero numpy's version picks (numpy 2.1
            # and 2.4 differ), so only the original expression is bit-identical there
            neg0 = (b0 == 0 and math.copysign(1.0, b0) < 0) or (b1 == 0 and math.copysign(1.0, b1) < 0) or \
                (b2 == 0 and math.copysign(1.0, b2) < 0) or (b3 == 0 and math.copysign(1.0, b3) < 0)
            if not neg0 and isfinite(b0) and isfinite(b1) and isfinite(b2) and isfinite(b3):
                x1 = max(0, b0)
                y1 = max(0, b1)
                x2 = min(width, x1 + max(0, b2))
                y2 = min(height, y1 + max(0, b3))
            else:
                x1 = np.max((0, b0))
                y1 = np.max((0, b1))
                x2 = np.min((width, x1 + np.max((0, b2))))
                y2 = np.min((height, y1 + np.max((0, b3))))
            if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                obj["clean_bbox"] = [x1, y1, x2, y2]
                cls = (self._class_index[obj["category_id"]] if self._class_index is not None
                       else self.class_ids.index(obj["category_id"]))
                rows.append((x1, y1, x2, y2, cls))

        res = np.zeros((len(rows), 5))
        if rows:
            res[:] = rows
        r = min(self.img_size[0] / height, self.img_size[1] / width)
        res[:, :4] *= r

        img_info = (height, width)
        resized_info = (int(height * r), int(width * r))
        file_name = im_ann["file_name"] if "file_name" in im_ann else "{:012}".format(id_) + ".jpg"
        return (res, img_info, resized_info, file_name)

    def load_anno(self, index):
        return self.annotations[index][0]

    def load_resized_img(self, index):
        img = self.load_image(index)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        return resized_img

    def load_image(self, index):
        file_name = self.annotations[index][3]

        img_file = os.path.join(self.data_dir, self.name, file_name)

        img = cv2.imread(img_file)
        assert img is not None, f"file named {img_file} not found"

        return img

    @cache_read_img(use_cache=True)
    def read_img(self, index):
        return self.load_resized_img(index)

    def pull_item(self, index):
        id_ = self.ids[index]
        label, origin_image_size, _, _ = self.annotations[index]
        img = self.read_img(index)

        return img, copy.deepcopy(label), origin_image_size, np.array([id_])

    @CacheDataset.mosaic_getitem
    def __getitem__(self, index):
        """
        One image / label pair for the given index is picked up and pre-processed.

        Args:
            index (int): data index

        Returns:
            img (numpy.ndarray): pre-processed image
            padded_labels (torch.Tensor): pre-processed label data.
                The shape is :math:`[max_labels, 5]`.
                each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
            info_img : tuple of h, w.
                h, w (int): original shape of the image
            img_id (int): same as the input index. Used for evaluation.
        """
        img, target, img_info, img_id = self.pull_item(index)

        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)
        return img, target, img_info, img_id
