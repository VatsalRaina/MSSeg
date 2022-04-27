import numpy as np
from scipy import ndimage
from metrics import intersection_over_union


class OneHotEncoding:
    def __call__(self, multi_mask, *args, **kwargs):
        """
        Parameters:
            * multi_mask (numpy.ndarray) of shape [H, W, D]
        that contains labels for instant segmentation gt / predictions

        Returns:
             * float32 numpy.ndarray of shape [L, H, W, D] where L is the
             number of unique labels

        Note: zero-label is inclluded
        """
        return np.stack([
            (multi_mask == label).astype("float32") for label in np.unique(multi_mask) if label != 0.0
        ], axis=0)


def mask_encoding(lesion_masks_list, dtype="float32", mask_type='binary'):
    if mask_type == 'one_hot':
        return np.stack(lesion_masks_list, axis=0)
    elif mask_type == 'multi':
        out_mask = np.zeros_like(lesion_masks_list[0])
        for i_fn, fn_lesion in enumerate(lesion_masks_list):
            out_mask += (fn_lesion.astype(dtype) * (i_fn + 1))
        return out_mask
    elif mask_type == 'binary':
        out_mask = np.zeros_like(lesion_masks_list[0])
        for i_fn, fn_lesion in enumerate(lesion_masks_list):
            out_mask += fn_lesion.astype(dtype)
        return out_mask
    else:
        return [lesion.astype(dtype) for lesion in lesion_masks_list]


def get_FN_lesions_mask(ground_truth, predictions, IoU_threshold, mask_type='binary'):
    fn_lesions = []

    mask_multi_gt = ndimage.label(ground_truth)[0]
    mask_multi_pred = ndimage.label(predictions)[0]

    for label_gt in np.unique(mask_multi_gt):
        if label_gt != 0.0:
            mask_label_gt = (mask_multi_gt == label_gt).astype(int)
            all_iou = [0]
            for int_label_pred in np.unique(mask_multi_pred * mask_label_gt):
                if int_label_pred != 0.0:
                    mask_label_pred = (mask_multi_pred == int_label_pred).astype(int)
                    all_iou.append(intersection_over_union(mask_label_pred, mask_label_gt))
            max_iou = max(all_iou)
            if max_iou < IoU_threshold:
                fn_lesions.append(mask_label_gt)

    return mask_encoding(lesion_masks_list=fn_lesions, dtype="int", mask_type=mask_type)


def get_TP_lesions_mask(ground_truth, predictions, IoU_threshold, mask_type='binary'):
    raise NotImplementedError

    tp_lesions_gt, tp_lesions_pred = [], []

    mask_multi_gt = ndimage.label(ground_truth)[0]
    mask_multi_pred = ndimage.label(predictions)[0]

    for label_pred in np.unique(mask_multi_pred):
        if label_pred != 0.0:
            mask_label_pred = (mask_multi_pred == label_pred).astype(int)
            max_iou = 0.0
            max_label = None
            for int_label_gt in np.unique(mask_multi_gt * mask_label_pred):  # iterate only intersections
                mask_label_gt = (mask_multi_gt == int_label_gt).astype(int)
                iou = intersection_over_union(mask_label_pred, mask_label_gt)
                if iou > max_iou:
                    max_iou = iou
                    max_label = int_label_gt
            if max_iou >= IoU_threshold:
                tp_lesions_gt.append((mask_multi_gt == max_label).astype(int))
                tp_lesions_pred.append(mask_label_pred)

    return mask_encoding(lesion_masks_list=tp_lesions_gt, dtype="int", mask_type=mask_type), \
           mask_encoding(lesion_masks_list=tp_lesions_pred, dtype="int", mask_type=mask_type)
