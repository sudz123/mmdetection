# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp

import mmcv
import numpy as np
import pytest
import torch
from mmcv.utils import build_from_cfg

from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from mmdet.datasets.builder import PIPELINES
from .utils import create_random_bboxes, create_random_masks, create_boxes_from_masks, get_random_idx,\
    bbox_flip, rescale_boxes, is_box_occluded, get_updated_mask


def test_resize():
    # test assertion if img_scale is a list
    with pytest.raises(AssertionError):
        transform = dict(type='Resize', img_scale=[1333, 800], keep_ratio=True)
        build_from_cfg(transform, PIPELINES)

    # test assertion if len(img_scale) while ratio_range is not None
    with pytest.raises(AssertionError):
        transform = dict(
            type='Resize',
            img_scale=[(1333, 800), (1333, 600)],
            ratio_range=(0.9, 1.1),
            keep_ratio=True)
        build_from_cfg(transform, PIPELINES)

    # test assertion for invalid multiscale_mode
    with pytest.raises(AssertionError):
        transform = dict(
            type='Resize',
            img_scale=[(1333, 800), (1333, 600)],
            keep_ratio=True,
            multiscale_mode='2333')
        build_from_cfg(transform, PIPELINES)

    # test assertion if both scale and scale_factor are setted
    with pytest.raises(AssertionError):
        results = dict(
            img_prefix=osp.join(osp.dirname(__file__), '../../../data'),
            img_info=dict(filename='color.jpg'))
        load = dict(type='LoadImageFromFile')
        load = build_from_cfg(load, PIPELINES)
        transform = dict(type='Resize', img_scale=(1333, 800), keep_ratio=True)
        transform = build_from_cfg(transform, PIPELINES)
        results = load(results)
        results['scale'] = (1333, 800)
        results['scale_factor'] = 1.0
        results = transform(results)

    transform = dict(type='Resize', img_scale=(1333, 800), keep_ratio=True)
    resize_module = build_from_cfg(transform, PIPELINES)

    results = dict()
    img = mmcv.imread(
        osp.join(osp.dirname(__file__), '../../../data/color.jpg'), 'color')
    results['img'] = img
    results['img2'] = copy.deepcopy(img)
    results['img_shape'] = img.shape
    results['ori_shape'] = img.shape
    # Set initial values for default meta_keys
    results['pad_shape'] = img.shape
    results['img_fields'] = ['img', 'img2']

    results = resize_module(results)
    assert np.equal(results['img'], results['img2']).all()

    results.pop('scale')
    results.pop('scale_factor')
    transform = dict(
        type='Resize',
        img_scale=(1280, 800),
        multiscale_mode='value',
        keep_ratio=False)
    resize_module = build_from_cfg(transform, PIPELINES)
    results = resize_module(results)
    assert np.equal(results['img'], results['img2']).all()
    assert results['img_shape'] == (800, 1280, 3)
    assert results['img'].dtype == results['img'].dtype == np.uint8

    results_seg = {
        'img': img,
        'img_shape': img.shape,
        'ori_shape': img.shape,
        'gt_semantic_seg': copy.deepcopy(img),
        'gt_seg': copy.deepcopy(img),
        'seg_fields': ['gt_semantic_seg', 'gt_seg']
    }
    transform = dict(
        type='Resize',
        img_scale=(640, 400),
        multiscale_mode='value',
        keep_ratio=False)
    resize_module = build_from_cfg(transform, PIPELINES)
    results_seg = resize_module(results_seg)
    assert results_seg['gt_semantic_seg'].shape == results_seg['gt_seg'].shape
    assert results_seg['img_shape'] == (400, 640, 3)
    assert results_seg['img_shape'] != results_seg['ori_shape']
    assert results_seg['gt_semantic_seg'].shape == results_seg['img_shape']
    assert np.equal(results_seg['gt_semantic_seg'],
                    results_seg['gt_seg']).all()


def test_flip():
    # test assertion for invalid flip_ratio
    with pytest.raises(AssertionError):
        transform = dict(type='RandomFlip', flip_ratio=1.5)
        build_from_cfg(transform, PIPELINES)
    # test assertion for 0 <= sum(flip_ratio) <= 1
    with pytest.raises(AssertionError):
        transform = dict(
            type='RandomFlip',
            flip_ratio=[0.7, 0.8],
            direction=['horizontal', 'vertical'])
        build_from_cfg(transform, PIPELINES)

    # test assertion for mismatch between number of flip_ratio and direction
    with pytest.raises(AssertionError):
        transform = dict(type='RandomFlip', flip_ratio=[0.4, 0.5])
        build_from_cfg(transform, PIPELINES)

    # test assertion for invalid direction
    with pytest.raises(AssertionError):
        transform = dict(
            type='RandomFlip', flip_ratio=1., direction='horizonta')
        build_from_cfg(transform, PIPELINES)

    transform = dict(type='RandomFlip', flip_ratio=1.)
    flip_module = build_from_cfg(transform, PIPELINES)

    results = dict()
    img = mmcv.imread(
        osp.join(osp.dirname(__file__), '../../../data/color.jpg'), 'color')
    original_img = copy.deepcopy(img)
    results['img'] = img
    results['img2'] = copy.deepcopy(img)
    results['img_shape'] = img.shape
    results['ori_shape'] = img.shape
    # Set initial values for default meta_keys
    results['pad_shape'] = img.shape
    results['scale_factor'] = 1.0
    results['img_fields'] = ['img', 'img2']

    results = flip_module(results)
    assert np.equal(results['img'], results['img2']).all()

    flip_module = build_from_cfg(transform, PIPELINES)
    results = flip_module(results)
    assert np.equal(results['img'], results['img2']).all()
    assert np.equal(original_img, results['img']).all()

    # test flip_ratio is float, direction is list
    transform = dict(
        type='RandomFlip',
        flip_ratio=0.9,
        direction=['horizontal', 'vertical', 'diagonal'])
    flip_module = build_from_cfg(transform, PIPELINES)

    results = dict()
    img = mmcv.imread(
        osp.join(osp.dirname(__file__), '../../../data/color.jpg'), 'color')
    original_img = copy.deepcopy(img)
    results['img'] = img
    results['img_shape'] = img.shape
    results['ori_shape'] = img.shape
    # Set initial values for default meta_keys
    results['pad_shape'] = img.shape
    results['scale_factor'] = 1.0
    results['img_fields'] = ['img']
    results = flip_module(results)
    if results['flip']:
        assert np.array_equal(
            mmcv.imflip(original_img, results['flip_direction']),
            results['img'])
    else:
        assert np.array_equal(original_img, results['img'])

    # test flip_ratio is list, direction is list
    transform = dict(
        type='RandomFlip',
        flip_ratio=[0.3, 0.3, 0.2],
        direction=['horizontal', 'vertical', 'diagonal'])
    flip_module = build_from_cfg(transform, PIPELINES)

    results = dict()
    img = mmcv.imread(
        osp.join(osp.dirname(__file__), '../../../data/color.jpg'), 'color')
    original_img = copy.deepcopy(img)
    results['img'] = img
    results['img_shape'] = img.shape
    results['ori_shape'] = img.shape
    # Set initial values for default meta_keys
    results['pad_shape'] = img.shape
    results['scale_factor'] = 1.0
    results['img_fields'] = ['img']
    results = flip_module(results)
    if results['flip']:
        assert np.array_equal(
            mmcv.imflip(original_img, results['flip_direction']),
            results['img'])
    else:
        assert np.array_equal(original_img, results['img'])


def test_random_crop():
    # test assertion for invalid random crop
    with pytest.raises(AssertionError):
        transform = dict(type='RandomCrop', crop_size=(-1, 0))
        build_from_cfg(transform, PIPELINES)

    results = dict()
    img = mmcv.imread(
        osp.join(osp.dirname(__file__), '../../../data/color.jpg'), 'color')
    results['img'] = img

    results['img_shape'] = img.shape
    results['ori_shape'] = img.shape
    # TODO: add img_fields test
    results['bbox_fields'] = ['gt_bboxes', 'gt_bboxes_ignore']
    # Set initial values for default meta_keys
    results['pad_shape'] = img.shape
    results['scale_factor'] = 1.0

    h, w, _ = img.shape
    gt_bboxes = create_random_bboxes(8, w, h)
    gt_bboxes_ignore = create_random_bboxes(2, w, h)
    results['gt_labels'] = np.ones(gt_bboxes.shape[0], dtype=np.int64)
    results['gt_bboxes'] = gt_bboxes
    results['gt_bboxes_ignore'] = gt_bboxes_ignore
    transform = dict(type='RandomCrop', crop_size=(h - 20, w - 20))
    crop_module = build_from_cfg(transform, PIPELINES)
    results = crop_module(results)
    assert results['img'].shape[:2] == (h - 20, w - 20)
    # All bboxes should be reserved after crop
    assert results['img_shape'][:2] == (h - 20, w - 20)
    assert results['gt_labels'].shape[0] == results['gt_bboxes'].shape[0]
    assert results['gt_labels'].dtype == np.int64
    assert results['gt_bboxes'].dtype == np.float32
    assert results['gt_bboxes'].shape[0] == 8
    assert results['gt_bboxes_ignore'].shape[0] == 2

    def area(bboxes):
        return np.prod(bboxes[:, 2:4] - bboxes[:, 0:2], axis=1)

    assert (area(results['gt_bboxes']) <= area(gt_bboxes)).all()
    assert (area(results['gt_bboxes_ignore']) <= area(gt_bboxes_ignore)).all()
    assert results['gt_bboxes'].dtype == np.float32
    assert results['gt_bboxes_ignore'].dtype == np.float32

    # test assertion for invalid crop_type
    with pytest.raises(ValueError):
        transform = dict(
            type='RandomCrop', crop_size=(1, 1), crop_type='unknown')
        build_from_cfg(transform, PIPELINES)

    # test assertion for invalid crop_size
    with pytest.raises(AssertionError):
        transform = dict(
            type='RandomCrop', crop_type='relative', crop_size=(0, 0))
        build_from_cfg(transform, PIPELINES)

    def _construct_toy_data():
        img = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.uint8)
        img = np.stack([img, img, img], axis=-1)
        results = dict()
        # image
        results['img'] = img
        results['img_shape'] = img.shape
        results['img_fields'] = ['img']
        # bboxes
        results['bbox_fields'] = ['gt_bboxes', 'gt_bboxes_ignore']
        results['gt_bboxes'] = np.array([[0., 0., 2., 1.]], dtype=np.float32)
        results['gt_bboxes_ignore'] = np.array([[2., 0., 3., 1.]],
                                               dtype=np.float32)
        # labels
        results['gt_labels'] = np.array([1], dtype=np.int64)
        return results

    # test crop_type "relative_range"
    results = _construct_toy_data()
    transform = dict(
        type='RandomCrop',
        crop_type='relative_range',
        crop_size=(0.3, 0.7),
        allow_negative_crop=True)
    transform_module = build_from_cfg(transform, PIPELINES)
    results_transformed = transform_module(copy.deepcopy(results))
    h, w = results_transformed['img_shape'][:2]
    assert int(2 * 0.3 + 0.5) <= h <= int(2 * 1 + 0.5)
    assert int(4 * 0.7 + 0.5) <= w <= int(4 * 1 + 0.5)
    assert results_transformed['gt_bboxes'].dtype == np.float32
    assert results_transformed['gt_bboxes_ignore'].dtype == np.float32

    # test crop_type "relative"
    transform = dict(
        type='RandomCrop',
        crop_type='relative',
        crop_size=(0.3, 0.7),
        allow_negative_crop=True)
    transform_module = build_from_cfg(transform, PIPELINES)
    results_transformed = transform_module(copy.deepcopy(results))
    h, w = results_transformed['img_shape'][:2]
    assert h == int(2 * 0.3 + 0.5) and w == int(4 * 0.7 + 0.5)
    assert results_transformed['gt_bboxes'].dtype == np.float32
    assert results_transformed['gt_bboxes_ignore'].dtype == np.float32

    # test crop_type "absolute"
    transform = dict(
        type='RandomCrop',
        crop_type='absolute',
        crop_size=(1, 2),
        allow_negative_crop=True)
    transform_module = build_from_cfg(transform, PIPELINES)
    results_transformed = transform_module(copy.deepcopy(results))
    h, w = results_transformed['img_shape'][:2]
    assert h == 1 and w == 2
    assert results_transformed['gt_bboxes'].dtype == np.float32
    assert results_transformed['gt_bboxes_ignore'].dtype == np.float32

    # test crop_type "absolute_range"
    transform = dict(
        type='RandomCrop',
        crop_type='absolute_range',
        crop_size=(1, 20),
        allow_negative_crop=True)
    transform_module = build_from_cfg(transform, PIPELINES)
    results_transformed = transform_module(copy.deepcopy(results))
    h, w = results_transformed['img_shape'][:2]
    assert 1 <= h <= 2 and 1 <= w <= 4
    assert results_transformed['gt_bboxes'].dtype == np.float32
    assert results_transformed['gt_bboxes_ignore'].dtype == np.float32


def test_min_iou_random_crop():
    results = dict()
    img = mmcv.imread(
        osp.join(osp.dirname(__file__), '../../../data/color.jpg'), 'color')
    results['img'] = img

    results['img_shape'] = img.shape
    results['ori_shape'] = img.shape
    results['bbox_fields'] = ['gt_bboxes', 'gt_bboxes_ignore']
    # Set initial values for default meta_keys
    results['pad_shape'] = img.shape
    results['scale_factor'] = 1.0
    h, w, _ = img.shape
    gt_bboxes = create_random_bboxes(1, w, h)
    gt_bboxes_ignore = create_random_bboxes(1, w, h)
    results['gt_labels'] = np.ones(gt_bboxes.shape[0], dtype=np.int64)
    results['gt_bboxes'] = gt_bboxes
    results['gt_bboxes_ignore'] = gt_bboxes_ignore
    transform = dict(type='MinIoURandomCrop')
    crop_module = build_from_cfg(transform, PIPELINES)

    # Test for img_fields
    results_test = copy.deepcopy(results)
    results_test['img1'] = results_test['img']
    results_test['img_fields'] = ['img', 'img1']
    with pytest.raises(AssertionError):
        crop_module(results_test)
    results = crop_module(results)
    assert results['gt_labels'].shape[0] == results['gt_bboxes'].shape[0]
    assert results['gt_labels'].dtype == np.int64
    assert results['gt_bboxes'].dtype == np.float32
    assert results['gt_bboxes_ignore'].dtype == np.float32

    patch = np.array([0, 0, results['img_shape'][1], results['img_shape'][0]])
    ious = bbox_overlaps(patch.reshape(-1, 4),
                         results['gt_bboxes']).reshape(-1)
    ious_ignore = bbox_overlaps(
        patch.reshape(-1, 4), results['gt_bboxes_ignore']).reshape(-1)
    mode = crop_module.mode
    if mode == 1:
        assert np.equal(results['gt_bboxes'], gt_bboxes).all()
        assert np.equal(results['gt_bboxes_ignore'], gt_bboxes_ignore).all()
    else:
        assert (ious >= mode).all()
        assert (ious_ignore >= mode).all()


def test_pad():
    # test assertion if both size_divisor and size is None
    with pytest.raises(AssertionError):
        transform = dict(type='Pad')
        build_from_cfg(transform, PIPELINES)

    transform = dict(type='Pad', size_divisor=32)
    transform = build_from_cfg(transform, PIPELINES)
    results = dict()
    img = mmcv.imread(
        osp.join(osp.dirname(__file__), '../../../data/color.jpg'), 'color')
    original_img = copy.deepcopy(img)
    results['img'] = img
    results['img2'] = copy.deepcopy(img)
    results['img_shape'] = img.shape
    results['ori_shape'] = img.shape
    # Set initial values for default meta_keys
    results['pad_shape'] = img.shape
    results['scale_factor'] = 1.0
    results['img_fields'] = ['img', 'img2']

    results = transform(results)
    assert np.equal(results['img'], results['img2']).all()
    # original img already divisible by 32
    assert np.equal(results['img'], original_img).all()
    img_shape = results['img'].shape
    assert img_shape[0] % 32 == 0
    assert img_shape[1] % 32 == 0

    resize_transform = dict(
        type='Resize', img_scale=(1333, 800), keep_ratio=True)
    resize_module = build_from_cfg(resize_transform, PIPELINES)
    results = resize_module(results)
    results = transform(results)
    img_shape = results['img'].shape
    assert np.equal(results['img'], results['img2']).all()
    assert img_shape[0] % 32 == 0
    assert img_shape[1] % 32 == 0

    # test the size and size_divisor must be None when pad2square is True
    with pytest.raises(AssertionError):
        transform = dict(type='Pad', size_divisor=32, pad_to_square=True)
        build_from_cfg(transform, PIPELINES)

    transform = dict(type='Pad', pad_to_square=True)
    transform = build_from_cfg(transform, PIPELINES)
    results['img'] = img
    results = transform(results)
    assert results['img'].shape[0] == results['img'].shape[1]

    # test the pad_val is converted to a dict
    transform = dict(type='Pad', size_divisor=32, pad_val=0)
    with pytest.deprecated_call():
        transform = build_from_cfg(transform, PIPELINES)

    assert isinstance(transform.pad_val, dict)
    results = transform(results)
    img_shape = results['img'].shape
    assert img_shape[0] % 32 == 0
    assert img_shape[1] % 32 == 0


def test_normalize():
    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True)
    transform = dict(type='Normalize', **img_norm_cfg)
    transform = build_from_cfg(transform, PIPELINES)
    results = dict()
    img = mmcv.imread(
        osp.join(osp.dirname(__file__), '../../../data/color.jpg'), 'color')
    original_img = copy.deepcopy(img)
    results['img'] = img
    results['img2'] = copy.deepcopy(img)
    results['img_shape'] = img.shape
    results['ori_shape'] = img.shape
    # Set initial values for default meta_keys
    results['pad_shape'] = img.shape
    results['scale_factor'] = 1.0
    results['img_fields'] = ['img', 'img2']

    results = transform(results)
    assert np.equal(results['img'], results['img2']).all()

    mean = np.array(img_norm_cfg['mean'])
    std = np.array(img_norm_cfg['std'])
    converted_img = (original_img[..., ::-1] - mean) / std
    assert np.allclose(results['img'], converted_img)


def test_albu_transform():
    results = dict(
        img_prefix=osp.join(osp.dirname(__file__), '../../../data'),
        img_info=dict(filename='color.jpg'))

    # Define simple pipeline
    load = dict(type='LoadImageFromFile')
    load = build_from_cfg(load, PIPELINES)

    albu_transform = dict(
        type='Albu', transforms=[dict(type='ChannelShuffle', p=1)])
    albu_transform = build_from_cfg(albu_transform, PIPELINES)

    normalize = dict(type='Normalize', mean=[0] * 3, std=[0] * 3, to_rgb=True)
    normalize = build_from_cfg(normalize, PIPELINES)

    # Execute transforms
    results = load(results)
    results = albu_transform(results)
    results = normalize(results)

    assert results['img'].dtype == np.float32


def test_random_center_crop_pad():
    # test assertion for invalid crop_size while test_mode=False
    with pytest.raises(AssertionError):
        transform = dict(
            type='RandomCenterCropPad',
            crop_size=(-1, 0),
            test_mode=False,
            test_pad_mode=None)
        build_from_cfg(transform, PIPELINES)

    # test assertion for invalid ratios while test_mode=False
    with pytest.raises(AssertionError):
        transform = dict(
            type='RandomCenterCropPad',
            crop_size=(511, 511),
            ratios=(1.0),
            test_mode=False,
            test_pad_mode=None)
        build_from_cfg(transform, PIPELINES)

    # test assertion for invalid mean, std and to_rgb
    with pytest.raises(AssertionError):
        transform = dict(
            type='RandomCenterCropPad',
            crop_size=(511, 511),
            mean=None,
            std=None,
            to_rgb=None,
            test_mode=False,
            test_pad_mode=None)
        build_from_cfg(transform, PIPELINES)

    # test assertion for invalid crop_size while test_mode=True
    with pytest.raises(AssertionError):
        transform = dict(
            type='RandomCenterCropPad',
            crop_size=(511, 511),
            ratios=None,
            border=None,
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True,
            test_mode=True,
            test_pad_mode=('logical_or', 127))
        build_from_cfg(transform, PIPELINES)

    # test assertion for invalid ratios while test_mode=True
    with pytest.raises(AssertionError):
        transform = dict(
            type='RandomCenterCropPad',
            crop_size=None,
            ratios=(0.9, 1.0, 1.1),
            border=None,
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True,
            test_mode=True,
            test_pad_mode=('logical_or', 127))
        build_from_cfg(transform, PIPELINES)

    # test assertion for invalid border while test_mode=True
    with pytest.raises(AssertionError):
        transform = dict(
            type='RandomCenterCropPad',
            crop_size=None,
            ratios=None,
            border=128,
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True,
            test_mode=True,
            test_pad_mode=('logical_or', 127))
        build_from_cfg(transform, PIPELINES)

    # test assertion for invalid test_pad_mode while test_mode=True
    with pytest.raises(AssertionError):
        transform = dict(
            type='RandomCenterCropPad',
            crop_size=None,
            ratios=None,
            border=None,
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True,
            test_mode=True,
            test_pad_mode=('do_nothing', 100))
        build_from_cfg(transform, PIPELINES)

    results = dict(
        img_prefix=osp.join(osp.dirname(__file__), '../../../data'),
        img_info=dict(filename='color.jpg'))

    load = dict(type='LoadImageFromFile', to_float32=True)
    load = build_from_cfg(load, PIPELINES)
    results = load(results)
    test_results = copy.deepcopy(results)

    h, w, _ = results['img_shape']
    gt_bboxes = create_random_bboxes(8, w, h)
    gt_bboxes_ignore = create_random_bboxes(2, w, h)
    results['gt_bboxes'] = gt_bboxes
    results['gt_bboxes_ignore'] = gt_bboxes_ignore
    train_transform = dict(
        type='RandomCenterCropPad',
        crop_size=(h - 20, w - 20),
        ratios=(1.0, ),
        border=128,
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True,
        test_mode=False,
        test_pad_mode=None)
    crop_module = build_from_cfg(train_transform, PIPELINES)
    train_results = crop_module(results)
    assert train_results['img'].shape[:2] == (h - 20, w - 20)
    # All bboxes should be reserved after crop
    assert train_results['pad_shape'][:2] == (h - 20, w - 20)
    assert train_results['gt_bboxes'].shape[0] == 8
    assert train_results['gt_bboxes_ignore'].shape[0] == 2
    assert train_results['gt_bboxes'].dtype == np.float32
    assert train_results['gt_bboxes_ignore'].dtype == np.float32

    test_transform = dict(
        type='RandomCenterCropPad',
        crop_size=None,
        ratios=None,
        border=None,
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True,
        test_mode=True,
        test_pad_mode=('logical_or', 127))
    crop_module = build_from_cfg(test_transform, PIPELINES)

    test_results = crop_module(test_results)
    assert test_results['img'].shape[:2] == (h | 127, w | 127)
    assert test_results['pad_shape'][:2] == (h | 127, w | 127)
    assert 'border' in test_results


def test_multi_scale_flip_aug():
    # test assertion if give both scale_factor and img_scale
    with pytest.raises(AssertionError):
        transform = dict(
            type='MultiScaleFlipAug',
            scale_factor=1.0,
            img_scale=[(1333, 800)],
            transforms=[dict(type='Resize')])
        build_from_cfg(transform, PIPELINES)

    # test assertion if both scale_factor and img_scale are None
    with pytest.raises(AssertionError):
        transform = dict(
            type='MultiScaleFlipAug',
            scale_factor=None,
            img_scale=None,
            transforms=[dict(type='Resize')])
        build_from_cfg(transform, PIPELINES)

    # test assertion if img_scale is not tuple or list of tuple
    with pytest.raises(AssertionError):
        transform = dict(
            type='MultiScaleFlipAug',
            img_scale=[1333, 800],
            transforms=[dict(type='Resize')])
        build_from_cfg(transform, PIPELINES)

    # test assertion if flip_direction is not str or list of str
    with pytest.raises(AssertionError):
        transform = dict(
            type='MultiScaleFlipAug',
            img_scale=[(1333, 800)],
            flip_direction=1,
            transforms=[dict(type='Resize')])
        build_from_cfg(transform, PIPELINES)

    scale_transform = dict(
        type='MultiScaleFlipAug',
        img_scale=[(1333, 800), (1333, 640)],
        transforms=[dict(type='Resize', keep_ratio=True)])
    transform = build_from_cfg(scale_transform, PIPELINES)

    results = dict()
    img = mmcv.imread(
        osp.join(osp.dirname(__file__), '../../../data/color.jpg'), 'color')
    results['img'] = img
    results['img_shape'] = img.shape
    results['ori_shape'] = img.shape
    # Set initial values for default meta_keys
    results['pad_shape'] = img.shape
    results['img_fields'] = ['img']

    scale_results = transform(copy.deepcopy(results))
    assert len(scale_results['img']) == 2
    assert scale_results['img'][0].shape == (750, 1333, 3)
    assert scale_results['img_shape'][0] == (750, 1333, 3)
    assert scale_results['img'][1].shape == (640, 1138, 3)
    assert scale_results['img_shape'][1] == (640, 1138, 3)

    scale_factor_transform = dict(
        type='MultiScaleFlipAug',
        scale_factor=[0.8, 1.0, 1.2],
        transforms=[dict(type='Resize', keep_ratio=False)])
    transform = build_from_cfg(scale_factor_transform, PIPELINES)
    scale_factor_results = transform(copy.deepcopy(results))
    assert len(scale_factor_results['img']) == 3
    assert scale_factor_results['img'][0].shape == (230, 409, 3)
    assert scale_factor_results['img_shape'][0] == (230, 409, 3)
    assert scale_factor_results['img'][1].shape == (288, 512, 3)
    assert scale_factor_results['img_shape'][1] == (288, 512, 3)
    assert scale_factor_results['img'][2].shape == (345, 614, 3)
    assert scale_factor_results['img_shape'][2] == (345, 614, 3)

    # test pipeline of coco_detection
    results = dict(
        img_prefix=osp.join(osp.dirname(__file__), '../../../data'),
        img_info=dict(filename='color.jpg'))
    load_cfg, multi_scale_cfg = mmcv.Config.fromfile(
        'configs/_base_/datasets/coco_detection.py').test_pipeline
    load = build_from_cfg(load_cfg, PIPELINES)
    transform = build_from_cfg(multi_scale_cfg, PIPELINES)
    results = transform(load(results))
    assert len(results['img']) == 1
    assert len(results['img_metas']) == 1
    assert isinstance(results['img'][0], torch.Tensor)
    assert isinstance(results['img_metas'][0], mmcv.parallel.DataContainer)
    assert results['img_metas'][0].data['ori_shape'] == (288, 512, 3)
    assert results['img_metas'][0].data['img_shape'] == (750, 1333, 3)
    assert results['img_metas'][0].data['pad_shape'] == (768, 1344, 3)
    assert results['img_metas'][0].data['scale_factor'].tolist() == [
        2.603515625, 2.6041667461395264, 2.603515625, 2.6041667461395264
    ]


def test_cutout():
    # test n_holes
    with pytest.raises(AssertionError):
        transform = dict(type='CutOut', n_holes=(5, 3), cutout_shape=(8, 8))
        build_from_cfg(transform, PIPELINES)
    with pytest.raises(AssertionError):
        transform = dict(type='CutOut', n_holes=(3, 4, 5), cutout_shape=(8, 8))
        build_from_cfg(transform, PIPELINES)
    # test cutout_shape and cutout_ratio
    with pytest.raises(AssertionError):
        transform = dict(type='CutOut', n_holes=1, cutout_shape=8)
        build_from_cfg(transform, PIPELINES)
    with pytest.raises(AssertionError):
        transform = dict(type='CutOut', n_holes=1, cutout_ratio=0.2)
        build_from_cfg(transform, PIPELINES)
    # either of cutout_shape and cutout_ratio should be given
    with pytest.raises(AssertionError):
        transform = dict(type='CutOut', n_holes=1)
        build_from_cfg(transform, PIPELINES)
    with pytest.raises(AssertionError):
        transform = dict(
            type='CutOut',
            n_holes=1,
            cutout_shape=(2, 2),
            cutout_ratio=(0.4, 0.4))
        build_from_cfg(transform, PIPELINES)

    results = dict()
    img = mmcv.imread(
        osp.join(osp.dirname(__file__), '../../../data/color.jpg'), 'color')

    results['img'] = img
    results['img_shape'] = img.shape
    results['ori_shape'] = img.shape
    results['pad_shape'] = img.shape
    results['img_fields'] = ['img']

    transform = dict(type='CutOut', n_holes=1, cutout_shape=(10, 10))
    cutout_module = build_from_cfg(transform, PIPELINES)
    cutout_result = cutout_module(copy.deepcopy(results))
    assert cutout_result['img'].sum() < img.sum()

    transform = dict(type='CutOut', n_holes=1, cutout_ratio=(0.8, 0.8))
    cutout_module = build_from_cfg(transform, PIPELINES)
    cutout_result = cutout_module(copy.deepcopy(results))
    assert cutout_result['img'].sum() < img.sum()

    transform = dict(
        type='CutOut',
        n_holes=(2, 4),
        cutout_shape=[(10, 10), (15, 15)],
        fill_in=(255, 255, 255))
    cutout_module = build_from_cfg(transform, PIPELINES)
    cutout_result = cutout_module(copy.deepcopy(results))
    assert cutout_result['img'].sum() > img.sum()

    transform = dict(
        type='CutOut',
        n_holes=1,
        cutout_ratio=(0.8, 0.8),
        fill_in=(255, 255, 255))
    cutout_module = build_from_cfg(transform, PIPELINES)
    cutout_result = cutout_module(copy.deepcopy(results))
    assert cutout_result['img'].sum() > img.sum()


def test_random_shift():
    # test assertion for invalid shift_ratio
    with pytest.raises(AssertionError):
        transform = dict(type='RandomShift', shift_ratio=1.5)
        build_from_cfg(transform, PIPELINES)

    # test assertion for invalid max_shift_px
    with pytest.raises(AssertionError):
        transform = dict(type='RandomShift', max_shift_px=-1)
        build_from_cfg(transform, PIPELINES)

    results = dict()
    img = mmcv.imread(
        osp.join(osp.dirname(__file__), '../../../data/color.jpg'), 'color')
    results['img'] = img
    # TODO: add img_fields test
    results['bbox_fields'] = ['gt_bboxes', 'gt_bboxes_ignore']

    h, w, _ = img.shape
    gt_bboxes = create_random_bboxes(8, w, h)
    gt_bboxes_ignore = create_random_bboxes(2, w, h)
    results['gt_labels'] = np.ones(gt_bboxes.shape[0], dtype=np.int64)
    results['gt_bboxes'] = gt_bboxes
    results['gt_bboxes_ignore'] = gt_bboxes_ignore
    transform = dict(type='RandomShift', shift_ratio=1.0)
    random_shift_module = build_from_cfg(transform, PIPELINES)
    results = random_shift_module(results)

    assert results['img'].shape[:2] == (h, w)
    assert results['gt_labels'].shape[0] == results['gt_bboxes'].shape[0]
    assert results['gt_labels'].dtype == np.int64
    assert results['gt_bboxes'].dtype == np.float32
    assert results['gt_bboxes_ignore'].dtype == np.float32


def test_random_affine():
    # test assertion for invalid translate_ratio
    with pytest.raises(AssertionError):
        transform = dict(type='RandomAffine', max_translate_ratio=1.5)
        build_from_cfg(transform, PIPELINES)

    # test assertion for invalid scaling_ratio_range
    with pytest.raises(AssertionError):
        transform = dict(type='RandomAffine', scaling_ratio_range=(1.5, 0.5))
        build_from_cfg(transform, PIPELINES)

    with pytest.raises(AssertionError):
        transform = dict(type='RandomAffine', scaling_ratio_range=(0, 0.5))
        build_from_cfg(transform, PIPELINES)

    results = dict()
    img = mmcv.imread(
        osp.join(osp.dirname(__file__), '../../../data/color.jpg'), 'color')
    results['img'] = img
    results['bbox_fields'] = ['gt_bboxes', 'gt_bboxes_ignore']

    h, w, _ = img.shape
    gt_bboxes = create_random_bboxes(8, w, h)
    gt_bboxes_ignore = create_random_bboxes(2, w, h)
    results['gt_labels'] = np.ones(gt_bboxes.shape[0], dtype=np.int64)
    results['gt_bboxes'] = gt_bboxes
    results['gt_bboxes_ignore'] = gt_bboxes_ignore
    transform = dict(type='RandomAffine')
    random_affine_module = build_from_cfg(transform, PIPELINES)
    results = random_affine_module(results)

    assert results['img'].shape[:2] == (h, w)
    assert results['gt_labels'].shape[0] == results['gt_bboxes'].shape[0]
    assert results['gt_labels'].dtype == np.int64
    assert results['gt_bboxes'].dtype == np.float32
    assert results['gt_bboxes_ignore'].dtype == np.float32

    # test filter bbox
    gt_bboxes = np.array([[0, 0, 1, 1], [0, 0, 3, 100]], dtype=np.float32)
    results['gt_labels'] = np.ones(gt_bboxes.shape[0], dtype=np.int64)
    results['gt_bboxes'] = gt_bboxes
    transform = dict(
        type='RandomAffine',
        max_rotate_degree=0.,
        max_translate_ratio=0.,
        scaling_ratio_range=(1., 1.),
        max_shear_degree=0.,
        border=(0, 0),
        min_bbox_size=2,
        max_aspect_ratio=20)
    random_affine_module = build_from_cfg(transform, PIPELINES)

    results = random_affine_module(results)

    assert results['gt_bboxes'].shape[0] == 0
    assert results['gt_labels'].shape[0] == 0
    assert results['gt_labels'].shape[0] == results['gt_bboxes'].shape[0]
    assert results['gt_labels'].dtype == np.int64
    assert results['gt_bboxes'].dtype == np.float32
    assert results['gt_bboxes_ignore'].dtype == np.float32


def test_mosaic():
    # test assertion for invalid img_scale
    with pytest.raises(AssertionError):
        transform = dict(type='Mosaic', img_scale=640)
        build_from_cfg(transform, PIPELINES)

    results = dict()
    img = mmcv.imread(
        osp.join(osp.dirname(__file__), '../../../data/color.jpg'), 'color')
    results['img'] = img
    # TODO: add img_fields test
    results['bbox_fields'] = ['gt_bboxes', 'gt_bboxes_ignore']

    h, w, _ = img.shape
    gt_bboxes = create_random_bboxes(8, w, h)
    gt_bboxes_ignore = create_random_bboxes(2, w, h)
    results['gt_labels'] = np.ones(gt_bboxes.shape[0], dtype=np.int64)
    results['gt_bboxes'] = gt_bboxes
    results['gt_bboxes_ignore'] = gt_bboxes_ignore
    transform = dict(type='Mosaic', img_scale=(10, 12))
    mosaic_module = build_from_cfg(transform, PIPELINES)

    # test assertion for invalid mix_results
    with pytest.raises(AssertionError):
        mosaic_module(results)

    results['mix_results'] = [copy.deepcopy(results)] * 3
    results = mosaic_module(results)
    assert results['img'].shape[:2] == (20, 24)
    assert results['gt_labels'].shape[0] == results['gt_bboxes'].shape[0]
    assert results['gt_labels'].dtype == np.int64
    assert results['gt_bboxes'].dtype == np.float32
    assert results['gt_bboxes_ignore'].dtype == np.float32


def test_mixup():
    # test assertion for invalid img_scale
    with pytest.raises(AssertionError):
        transform = dict(type='MixUp', img_scale=640)
        build_from_cfg(transform, PIPELINES)

    results = dict()
    img = mmcv.imread(
        osp.join(osp.dirname(__file__), '../../../data/color.jpg'), 'color')
    results['img'] = img
    # TODO: add img_fields test
    results['bbox_fields'] = ['gt_bboxes', 'gt_bboxes_ignore']

    h, w, _ = img.shape
    gt_bboxes = create_random_bboxes(8, w, h)
    gt_bboxes_ignore = create_random_bboxes(2, w, h)
    results['gt_labels'] = np.ones(gt_bboxes.shape[0], dtype=np.int64)
    results['gt_bboxes'] = gt_bboxes
    results['gt_bboxes_ignore'] = gt_bboxes_ignore
    transform = dict(type='MixUp', img_scale=(10, 12))
    mixup_module = build_from_cfg(transform, PIPELINES)

    # test assertion for invalid mix_results
    with pytest.raises(AssertionError):
        mixup_module(results)

    with pytest.raises(AssertionError):
        results['mix_results'] = [copy.deepcopy(results)] * 2
        mixup_module(results)

    results['mix_results'] = [copy.deepcopy(results)]
    results = mixup_module(results)
    assert results['img'].shape[:2] == (288, 512)
    assert results['gt_labels'].shape[0] == results['gt_bboxes'].shape[0]
    assert results['gt_labels'].dtype == np.int64
    assert results['gt_bboxes'].dtype == np.float32
    assert results['gt_bboxes_ignore'].dtype == np.float32


def test_simple_copy_paste():
    """
    # TODO work on masks function -> use get boxes from masks code
    # make masks first (n masks)
    # use boxes_from_masks func to get the random boxes (create_random_bboxes_from_masks)

    Start with getting random data, data which breaks the algo, (ask sudarshan for the cases) " \
    " Then check out each func with an assertion in a deterministic format "
    # TODO just remember this works on the masks ; make sure if the masks are of varied sizes and break the system
    # test out the individual functions before then we can run the whole loop on assert
    """

    # test assertion for invalid scale range
    with pytest.raises(AssertionError):
        transform = dict(type='SimpleCopyPaste', img_scale=0.5)
        build_from_cfg(transform, PIPELINES)

    results = dict()
    img = mmcv.imread(
        osp.join(osp.dirname(__file__), '../../../data/color.jpg'), 'color')
    results['img'] = img
    results['bbox_fields'] = ['gt_bboxes', 'gt_bboxes_ignore']

    max_paste_objects = 6
    box_occlusion_thresh = 10   # TODO add check for mask size check as per google code
    occluded_area_thresh = 300

    h, w, _ = img.shape
    gt_masks = create_random_masks(6, w, h)
    gt_bboxes = create_boxes_from_masks(gt_masks.masks)
    gt_bboxes_ignore = create_random_bboxes(2, w, h)

    # test assertion for random index generation
    idx = get_random_idx(max_paste_objects, gt_bboxes)
    assert len(idx) <= max_paste_objects
    for x in idx:
        assert x < gt_bboxes.shape[0]

    # test assertion for invalid box values
    with pytest.raises(AssertionError):
        create_boxes_from_masks(np.array([[1, 2, 3]]))

    # test assertion for creating box from mask function
    # TODO check by hand if the values are correct - Done
    boxes = create_boxes_from_masks(gt_masks.masks)
    assert np.array([0.,  20., 511.,  47.]).all() == gt_bboxes[0].all()   # hand code these values
    assert np.array([4.,  56., 503.,  84.]).all() == gt_bboxes[1].all()
    assert np.array([0., 112., 511., 253.]).all() == gt_bboxes[2].all()
    assert np.array([0., 140., 511., 196.]).all() == gt_bboxes[3].all()
    assert np.array([0.,  28., 511.,  39.]).all() == gt_bboxes[4].all()
    assert np.array([0.,   0., 511.,  83.]).all() == gt_bboxes[5].all()
    assert boxes.dtype == np.float32

    # TODO check if the boxes are out of the frame of the image post flipping,
    #  and if they are eliminated, if not add code for the same - Done
    # test assertion for box flip function
    flipped_bboxes = bbox_flip(gt_bboxes, img.shape)
    assert np.array([1., 269., 512., 281.]).all() == flipped_bboxes[0].all()   # hand code these values
    assert np.array([9.,  56., 508.,  84.]).all() == flipped_bboxes[1].all()   # TODO check by hand if the values are correct
    assert np.array([1., 112., 512., 253.]).all() == flipped_bboxes[2].all()
    assert np.array([1., 140., 512., 196.]).all() == flipped_bboxes[3].all()
    assert np.array([1.,  28., 512.,  39.]).all() == flipped_bboxes[4].all()
    assert np.array([1.,   0., 512.,  83.]).all() == flipped_bboxes[5].all()
    # check if the boxes are out of bounds
    assert flipped_bboxes[..., 1::2].all() < img.shape[0], "box y-axis co-ords out of image frame"
    assert flipped_bboxes[..., ::2].all() < img.shape[1], "box x-axis co-ords out of image frame"

    # test assertion for rescale boxes function
    # TODO check if the boxes are kept inside the frame of the image - Done
    rescaled_boxes = rescale_boxes(gt_bboxes, (1.4, 1.6))   # TODO check the scaling - Done
    assert np.array([0., 109.2, 817.60004, 294.]).all() == rescaled_boxes[0].all()    # hand code these values
    assert np.array([6.4, 78.4, 804.8, 117.6]).all() == rescaled_boxes[1].all()
    assert np.array([0., 156.8, 817.60004, 354.19998]).all() == rescaled_boxes[2].all()
    assert np.array([0., 196., 817.60004, 274.4]).all() == rescaled_boxes[3].all()
    assert np.array([0., 39.2, 817.60004,  54.6]).all() == rescaled_boxes[4].all()
    assert np.array([0., 0., 817.60004, 116.2]).all() == rescaled_boxes[5].all()
    # check if the boxes are out of bounds
    assert rescaled_boxes[..., 1::2].all() < img.shape[0], "box y-axis co-ords out of image frame"
    assert rescaled_boxes[..., ::2].all() < img.shape[1], "box x-axis co-ords out of image frame"
    # TODO check for the threshold of how much area is being kept, the size threshold for the mask
    # TODO check for size limits as well for mask and box size

    # TODO check this test assertion for checking occluded boxes function
    test_masks = create_random_masks(4, w, h, create_overlapping=True).masks
    assert is_box_occluded(test_masks[0], test_masks[1], box_occlusion_thresh)\
           and np.sum(test_masks[0]) <= occluded_area_thresh == True
    assert is_box_occluded(test_masks[1], test_masks[0], box_occlusion_thresh) \
           and np.sum(test_masks[0]) <= occluded_area_thresh == False
    assert is_box_occluded(test_masks[1], test_masks[0], box_occlusion_thresh) == False
    assert is_box_occluded(test_masks[0], test_masks[1], box_occlusion_thresh) == True

    # test assertion for updated mask creation function
    with pytest.raises(AssertionError):
        get_updated_mask(np.ones((3, 540, 240)), np.ones((3, 240, 540)))

    updated_mask = test_masks[1] - test_masks[0]
    updated_mask2 = get_updated_mask(test_masks[0], test_masks[1])
    assert updated_mask.all() == updated_mask2.all()

    results['gt_labels'] = np.ones(gt_bboxes.shape[0], dtype=np.int64)
    results['gt_bboxes'] = gt_bboxes
    results['gt_bboxes_ignore'] = gt_bboxes_ignore
    results['gt_masks'] = gt_masks

    transform = dict(type='SimpleCopyPaste', img_scale=(0.1, 2))
    simple_copy_paste_module = build_from_cfg(transform, PIPELINES)

    # test assertion for mix_results key not present
    with pytest.raises(AssertionError):
        simple_copy_paste_module(results)

    results['mix_results'] = [copy.deepcopy(results)] * 1
    results = simple_copy_paste_module(results)

    assert results['img'].shape[2] == 3
    assert results['img'].dtype == np.uint8

    assert results['gt_labels'].shape[0] == results['gt_bboxes'].shape[0]
    assert results['gt_labels'].dtype == np.int64

    assert results['gt_bboxes'].dtype == np.float32
    assert results['gt_bboxes_ignore'].dtype == np.float32

    assert results['gt_masks'].masks.dtype == np.uint8
    assert results['gt_masks'].masks.shape[0] == results['gt_bboxes'].shape[0]
    assert results['gt_masks'].masks.shape[0] == results['gt_labels'].shape[0]

"""
/home/grantorshadow/miniconda3/envs/openmmlab_new/bin/python /home/grantorshadow/Shaunak/pycharm-2021.2.2/plugins/python/helpers/pydev/pydevd.py --multiproc --qt-support=auto --client 127.0.0.1 --port 36301 --file /home/grantorshadow/Shaunak/MMdet/mmdetection/tools/train.py /home/grantorshadow/Shaunak/MMdet/mmdetection/configs/simplecopypaste/mask_rcnn_r50_fpn_random_simple_copy_paste_mstrain_2x_lvis_v1.py --no-validate
Connected to pydev debugger (build 212.5284.44)
2021-11-01 15:26:59,054 - mmdet - INFO - Environment info:
------------------------------------------------------------
sys.platform: linux
Python: 3.7.11 (default, Jul 27 2021, 14:32:16) [GCC 7.5.0]
CUDA available: True
GPU 0: NVIDIA GeForce GTX 1060
CUDA_HOME: /usr/local/cuda
NVCC: Build cuda_11.2.r11.2/compiler.29618528_0
GCC: gcc (Ubuntu 7.5.0-6ubuntu2) 7.5.0
PyTorch: 1.7.1
PyTorch compiling details: PyTorch built with:
  - GCC 7.3
  - C++ Version: 201402
  - Intel(R) oneAPI Math Kernel Library Version 2021.3-Product Build 20210617 for Intel(R) 64 architecture applications
  - Intel(R) MKL-DNN v1.6.0 (Git Hash 5ef631a030a6f73131c77892041042805a06064f)
  - OpenMP 201511 (a.k.a. OpenMP 4.5)
  - NNPACK is enabled
  - CPU capability usage: AVX2
  - CUDA Runtime 11.0
  - NVCC architecture flags: -gencode;arch=compute_37,code=sm_37;-gencode;arch=compute_50,code=sm_50;-gencode;arch=compute_60,code=sm_60;-gencode;arch=compute_61,code=sm_61;-gencode;arch=compute_70,code=sm_70;-gencode;arch=compute_75,code=sm_75;-gencode;arch=compute_80,code=sm_80;-gencode;arch=compute_37,code=compute_37
  - CuDNN 8.0.5
  - Magma 2.5.2
  - Build settings: BLAS=MKL, BUILD_TYPE=Release, CXX_FLAGS= -Wno-deprecated -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -fopenmp -DNDEBUG -DUSE_FBGEMM -DUSE_QNNPACK -DUSE_PYTORCH_QNNPACK -DUSE_XNNPACK -DUSE_VULKAN_WRAPPER -O2 -fPIC -Wno-narrowing -Wall -Wextra -Werror=return-type -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-sign-compare -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function -Wno-unused-result -Wno-unused-local-typedefs -Wno-strict-overflow -Wno-strict-aliasing -Wno-error=deprecated-declarations -Wno-stringop-overflow -Wno-psabi -Wno-error=pedantic -Wno-error=redundant-decls -Wno-error=old-style-cast -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Wno-stringop-overflow, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, USE_CUDA=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_MKL=ON, USE_MKLDNN=ON, USE_MPI=OFF, USE_NCCL=ON, USE_NNPACK=ON, USE_OPENMP=ON, 
TorchVision: 0.8.2
OpenCV: 4.5.3
MMCV: 1.3.13
MMCV Compiler: GCC 7.3
MMCV CUDA Compiler: 11.0
MMDetection: 2.17.0+ccf8691
------------------------------------------------------------
2021-11-01 15:27:02,125 - mmdet - INFO - Distributed training: False
2021-11-01 15:27:06,051 - mmdet - INFO - Config:
model = dict(
    type='MaskRCNN',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=1203,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0),
            cls_predictor_cfg=dict(type='NormedLinear', tempearture=20)),
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=1203,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            mask_size=28,
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.0001,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=300,
            mask_thr_binary=0.5)))
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[16, 22])
runner = dict(type='EpochBasedRunner', max_epochs=24)
checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
load_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True)
]
train_pipeline = [
    dict(type='SimpleCopyPaste'),
    dict(
        type='Resize',
        img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                   (1333, 768), (1333, 800)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
parent_dataset_type = 'MultiImageMixDataset'
dataset_type = 'LVISV1Dataset'
data_root = '/home/grantorshadow/Shaunak/mmdetection/data/lvis_v1/'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=0,
    train=dict(
        type='MultiImageMixDataset',
        dataset=dict(
            type='LVISV1Dataset',
            ann_file=
            '/home/grantorshadow/Shaunak/mmdetection/data/lvis_v1/annotations/lvis_v1_val.json',
            img_prefix='/home/grantorshadow/Shaunak/mmdetection/data/lvis_v1/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True, with_mask=True)
            ]),
        pipeline=[
            dict(type='SimpleCopyPaste'),
            dict(
                type='Resize',
                img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                           (1333, 768), (1333, 800)],
                multiscale_mode='value',
                keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
        ]),
    val=dict(
        type='LVISV1Dataset',
        ann_file=
        '/home/grantorshadow/Shaunak/mmdetection/data/lvis_v1/annotations/lvis_v1_val.json',
        img_prefix='/home/grantorshadow/Shaunak/mmdetection/data/lvis_v1/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='LVISV1Dataset',
        ann_file=
        '/home/grantorshadow/Shaunak/mmdetection/data/lvis_v1/annotations/lvis_v1_val.json',
        img_prefix='/home/grantorshadow/Shaunak/mmdetection/data/lvis_v1/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
evaluation = dict(interval=24, metric=['bbox', 'segm'])
work_dir = './work_dirs/mask_rcnn_r50_fpn_random_simple_copy_paste_mstrain_2x_lvis_v1'
gpu_ids = range(0, 1)
/home/grantorshadow/Shaunak/MMdet/mmdetection/mmdet/core/anchor/builder.py:17: UserWarning: ``build_anchor_generator`` would be deprecated soon, please use ``build_prior_generator`` 
  '``build_anchor_generator`` would be deprecated soon, please use '
2021-11-01 15:27:07,298 - mmdet - INFO - initialize ResNet with init_cfg {'type': 'Pretrained', 'checkpoint': 'torchvision://resnet50'}
2021-11-01 15:27:07,298 - mmcv - INFO - load model from: torchvision://resnet50
2021-11-01 15:27:07,299 - mmcv - INFO - Use load_from_torchvision loader
2021-11-01 15:27:07,489 - mmcv - WARNING - The model and loaded state dict do not match exactly
unexpected key in source state_dict: fc.weight, fc.bias
2021-11-01 15:27:07,528 - mmdet - INFO - initialize FPN with init_cfg {'type': 'Xavier', 'layer': 'Conv2d', 'distribution': 'uniform'}
2021-11-01 15:27:07,555 - mmdet - INFO - initialize RPNHead with init_cfg {'type': 'Normal', 'layer': 'Conv2d', 'std': 0.01}
2021-11-01 15:27:07,563 - mmdet - INFO - initialize Shared2FCBBoxHead with init_cfg [{'type': 'Normal', 'std': 0.01, 'override': {'name': 'fc_cls'}}, {'type': 'Normal', 'std': 0.001, 'override': {'name': 'fc_reg'}}, {'type': 'Xavier', 'layer': 'Linear', 'override': [{'name': 'shared_fcs'}, {'name': 'cls_fcs'}, {'name': 'reg_fcs'}]}]
2021-11-01 15:27:15,503 - mmdet - INFO - Start running, host: grantorshadow@uss-nautilus, work_dir: /home/grantorshadow/Shaunak/MMdet/mmdetection/tools/work_dirs/mask_rcnn_r50_fpn_random_simple_copy_paste_mstrain_2x_lvis_v1
[11/01 15:27:15] mmdet INFO: Start running, host: grantorshadow@uss-nautilus, work_dir: /home/grantorshadow/Shaunak/MMdet/mmdetection/tools/work_dirs/mask_rcnn_r50_fpn_random_simple_copy_paste_mstrain_2x_lvis_v1
2021-11-01 15:27:15,503 - mmdet - INFO - Hooks will be executed in the following order:
before_run:
(VERY_HIGH   ) StepLrUpdaterHook                  
(NORMAL      ) CheckpointHook                     
(VERY_LOW    ) TextLoggerHook                     
 -------------------- 
before_train_epoch:
(VERY_HIGH   ) StepLrUpdaterHook                  
(NORMAL      ) NumClassCheckHook                  
(LOW         ) IterTimerHook                      
(VERY_LOW    ) TextLoggerHook                     
 -------------------- 
before_train_iter:
(VERY_HIGH   ) StepLrUpdaterHook                  
(LOW         ) IterTimerHook                      
 -------------------- 
after_train_iter:
(ABOVE_NORMAL) OptimizerHook                      
(NORMAL      ) CheckpointHook                     
(LOW         ) IterTimerHook                      
(VERY_LOW    ) TextLoggerHook                     
 -------------------- 
after_train_epoch:
(NORMAL      ) CheckpointHook                     
(VERY_LOW    ) TextLoggerHook                     
 -------------------- 
before_val_epoch:
(NORMAL      ) NumClassCheckHook                  
(LOW         ) IterTimerHook                      
(VERY_LOW    ) TextLoggerHook                     
 -------------------- 
before_val_iter:
(LOW         ) IterTimerHook                      
 -------------------- 
after_val_iter:
(LOW         ) IterTimerHook                      
 -------------------- 
after_val_epoch:
(VERY_LOW    ) TextLoggerHook                     
 -------------------- 
[11/01 15:27:15] mmdet INFO: Hooks will be executed in the following order:
before_run:
(VERY_HIGH   ) StepLrUpdaterHook                  
(NORMAL      ) CheckpointHook                     
(VERY_LOW    ) TextLoggerHook                     
 -------------------- 
before_train_epoch:
(VERY_HIGH   ) StepLrUpdaterHook                  
(NORMAL      ) NumClassCheckHook                  
(LOW         ) IterTimerHook                      
(VERY_LOW    ) TextLoggerHook                     
 -------------------- 
before_train_iter:
(VERY_HIGH   ) StepLrUpdaterHook                  
(LOW         ) IterTimerHook                      
 -------------------- 
after_train_iter:
(ABOVE_NORMAL) OptimizerHook                      
(NORMAL      ) CheckpointHook                     
(LOW         ) IterTimerHook                      
(VERY_LOW    ) TextLoggerHook                     
 -------------------- 
after_train_epoch:
(NORMAL      ) CheckpointHook                     
(VERY_LOW    ) TextLoggerHook                     
 -------------------- 
before_val_epoch:
(NORMAL      ) NumClassCheckHook                  
(LOW         ) IterTimerHook                      
(VERY_LOW    ) TextLoggerHook                     
 -------------------- 
before_val_iter:
(LOW         ) IterTimerHook                      
 -------------------- 
after_val_iter:
(LOW         ) IterTimerHook                      
 -------------------- 
after_val_epoch:
(VERY_LOW    ) TextLoggerHook                     
 -------------------- 
2021-11-01 15:27:15,504 - mmdet - INFO - workflow: [('train', 1)], max: 24 epochs
[11/01 15:27:15] mmdet INFO: workflow: [('train', 1)], max: 24 epochs
cv2.bitwise_and(
	pastable_img_src, src_mask_3channel)
PyDev console: starting.
pastable_img_src.shape, type(pastable_img_src)
((334, 500, 3), <class 'numpy.ndarray'>)
src_mask_3channel.shape, type(src_mask_3channel)
((334, 500, 3), <class 'numpy.ndarray'>)
pastable_img_src
array([[[100, 100, 100],
        [100, 100, 100],
        [100, 100, 100],
        ...,
        [100, 100, 100],
        [100, 100, 100],
        [100, 100, 100]],
       [[100, 100, 100],
        [100, 100, 100],
        [100, 100, 100],
        ...,
        [100, 100, 100],
        [100, 100, 100],
        [100, 100, 100]],
       [[100, 100, 100],
        [100, 100, 100],
        [100, 100, 100],
        ...,
        [100, 100, 100],
        [100, 100, 100],
        [100, 100, 100]],
       ...,
       [[100, 100, 100],
        [100, 100, 100],
        [100, 100, 100],
        ...,
        [100, 100, 100],
        [100, 100, 100],
        [100, 100, 100]],
       [[100, 100, 100],
        [100, 100, 100],
        [100, 100, 100],
        ...,
        [100, 100, 100],
        [100, 100, 100],
        [100, 100, 100]],
       [[100, 100, 100],
        [100, 100, 100],
        [100, 100, 100],
        ...,
        [100, 100, 100],
        [100, 100, 100],
        [100, 100, 100]]], dtype=uint8)
src_mask_3channel
array([[[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        ...,
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]],
       [[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        ...,
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]],
       [[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        ...,
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]],
       ...,
       [[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        ...,
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]],
       [[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        ...,
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]],
       [[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        ...,
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]]], dtype=uint8)

"""