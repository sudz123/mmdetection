# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

from mmdet.core.mask import BitmapMasks, PolygonMasks


def _check_fields(results, pipeline_results, keys):
    """Check data in fields from two results are same."""
    for key in keys:
        if isinstance(results[key], (BitmapMasks, PolygonMasks)):
            assert np.equal(results[key].to_ndarray(),
                            pipeline_results[key].to_ndarray()).all()
        else:
            assert np.equal(results[key], pipeline_results[key]).all()
            assert results[key].dtype == pipeline_results[key].dtype


def check_result_same(results, pipeline_results):
    """Check whether the `pipeline_results` is the same with the predefined
    `results`.

    Args:
        results (dict): Predefined results which should be the standard output
            of the transform pipeline.
        pipeline_results (dict): Results processed by the transform pipeline.
    """
    # check image
    _check_fields(results, pipeline_results,
                  results.get('img_fields', ['img']))
    # check bboxes
    _check_fields(results, pipeline_results, results.get('bbox_fields', []))
    # check masks
    _check_fields(results, pipeline_results, results.get('mask_fields', []))
    # check segmentations
    _check_fields(results, pipeline_results, results.get('seg_fields', []))
    # check gt_labels
    if 'gt_labels' in results:
        assert np.equal(results['gt_labels'],
                        pipeline_results['gt_labels']).all()


def construct_toy_data(poly2mask=True):
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
    # masks
    results['mask_fields'] = ['gt_masks']
    if poly2mask:
        gt_masks = np.array([[0, 1, 1, 0], [0, 1, 0, 0]],
                            dtype=np.uint8)[None, :, :]
        results['gt_masks'] = BitmapMasks(gt_masks, 2, 4)
    else:
        raw_masks = [[np.array([0, 0, 2, 0, 2, 1, 0, 1], dtype=np.float)]]
        results['gt_masks'] = PolygonMasks(raw_masks, 2, 4)
    # segmentations
    results['seg_fields'] = ['gt_semantic_seg']
    results['gt_semantic_seg'] = img[..., 0]
    return results


def create_random_bboxes(num_bboxes, img_w, img_h):
    bboxes_left_top = np.random.uniform(0, 0.5, size=(num_bboxes, 2))
    bboxes_right_bottom = np.random.uniform(0.5, 1, size=(num_bboxes, 2))
    bboxes = np.concatenate((bboxes_left_top, bboxes_right_bottom), 1)
    bboxes = (bboxes * np.array([img_w, img_h, img_w, img_h])).astype(
        np.float32)
    return bboxes


def create_random_masks(num_masks, mask_w, mask_h, create_overlapping=False) -> BitmapMasks:
    mask_arr = np.zeros((num_masks, mask_w, mask_h), dtype=np.uint8)

    if create_overlapping:
        mask_arr[0, 100:150, 100:120] = 1
        mask_arr[1, 100:450, 100:260] = 1
        mask_arr[2, 70:300, 130:200] = 1
        mask_arr[3, 90:250, 100:260] = 1

    else:
        # random box
        r1, r2 = np.random.randint(0, mask_w, 2)
        if r1 > r2:
            temp, r1 = r1, r2
            r2 = temp
        # value_when_true if condition else value_when_false
        c1, c2 = np.random.randint(0, mask_h, 2)
        if c1 > c2:
            temp, c1 = c1, c2
            c2 = temp

        mask_arr[0, r1:r2, c1:c2] = 1

        # overlapping
        mask_arr[1, 100:150, 100:120] = 1
        mask_arr[2, 200:450, 100:260] = 1

        # small
        mask_arr[3, 250:350, 100:140] = 1

        # large
        mask_arr[4, 50:70, 30:190] = 1

        # edges
        mask_arr[5, 0:150, 0:50] = 1

    mask_arr = BitmapMasks(mask_arr, mask_arr.shape[2],
                           mask_arr.shape[1])

    return mask_arr


def create_boxes_from_masks(mask_list) -> np.array:
    """Convert mask Y to a bounding box, assumes 0 as background nonzero object"""
    assert mask_list.shape[1] % 4 == 0
    random_masks = []
    for mask in mask_list:
        Y_vals, X_vals = np.nonzero(mask)   # gives in a Xi , Yi co-ord system in the cols and rows

        if len(Y_vals) == 0:
            random_masks.append(np.zeros(4, dtype=np.float32))

        y1 = np.min(Y_vals)
        x1 = np.min(X_vals)
        y2 = np.max(Y_vals)
        x2 = np.max(X_vals)

        random_masks.append(np.array([x1, y1, x2, y2], dtype=np.float32))
    return np.array(random_masks)


def get_random_idx(max_paste_objects, arr) -> np.array:
    if arr.shape[0] <= max_paste_objects:
        return np.random.randint(0, arr.shape[0], size=arr.shape[0])
    return np.random.randint(0, arr.shape[0], size=max_paste_objects)


def bbox_flip(bboxes, img_shape) -> np.array:
    """Flip bboxes horizontally.

    Args:
        bboxes (numpy.ndarray): Bounding boxes, shape (..., 4*k)
        img_shape (tuple[int]): Image shape (height, width)

    Returns:
        numpy.ndarray: Flipped bounding boxes.
    """

    assert bboxes.shape[-1] % 4 == 0
    flipped = bboxes.copy()
    w = img_shape[1]
    flipped[..., 0::4] = w - bboxes[..., 2::4]
    flipped[..., 2::4] = w - bboxes[..., 0::4]
    return flipped


def rescale_boxes(bboxes, rescale_ratio, img_shape=None, clip=False) -> np.array:
    if isinstance(rescale_ratio, float):
        bboxes = bboxes * rescale_ratio
    if isinstance(rescale_ratio, tuple):
        bboxes[:, 0::2] = bboxes[:, 0::2] * rescale_ratio[1]
        bboxes[:, 1::2] = bboxes[:, 1::2] * rescale_ratio[0]
    if clip:
        bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
        bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
    return bboxes


def is_box_occluded(box1, box2, box_iou_threshold) -> bool:
    if np.any(np.abs(box1 - box2) > box_iou_threshold):
        return True
    return False


def get_updated_mask(parent_mask, child_mask) -> np.array:
    assert parent_mask.shape == child_mask.shape, 'Cannot compare two arrays of different size'
    return np.where(parent_mask, 0, child_mask)
