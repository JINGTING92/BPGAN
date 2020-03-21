import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc, confusion_matrix

# noinspection PyPep8Naming
def AUC_ROC(gt_seg, pred_seg):
    """
    Area under the ROC curve with x axis flipped
    ROC: Receiver operating characteristic
    """
    # roc_auc_score: sklearn function
    AUC_ROC_ = roc_auc_score(gt_seg.flatten(), pred_seg.flatten())
    return AUC_ROC_


# noinspection PyPep8Naming
def AUC_PR(gt_seg, pred_seg):
    """
    Precision-recall curve: sklearn function
    auc: Area Under Curve, sklearn function
    """
    precision, recall, _ = precision_recall_curve(gt_seg.flatten(), pred_seg.flatten(), pos_label=1)
    AUC_prec_rec = auc(recall, precision)

    return precision, recall, AUC_prec_rec

def dice_coef_tumor(gt, pred):

    y_true_f = np.reshape(gt, [-1])
    y_pred_f = np.reshape(pred, [-1])

    y_true_tumor = np.true_divide(y_true_f, 2).astype(int)
    y_pred_tumor = np.true_divide(y_pred_f, 2).astype(int)

    assert np.max(y_pred_tumor) <= 1 or np.max(y_true_tumor) <= 1

    intersection_t = np.sum(y_true_tumor * y_pred_tumor)
    union_t = np.sum(y_true_tumor) + np.sum(y_pred_tumor)

    if np.sum(y_true_tumor) == 0 and np.sum(y_pred_tumor) == 0:
        dice_tumor = -1
    else:
        dice_tumor = (2. * intersection_t) / (union_t + 0.00001)

    return dice_tumor


def dice_coef_organ(gt, seg):

    y_true_f = np.reshape(gt, [-1])
    y_pred_f = np.reshape(seg, [-1])

    y_true_org = y_true_f.copy()
    y_pred_org = y_pred_f.copy()
    y_true_org[y_true_f > 0] = 1
    y_pred_org[y_pred_f > 0] = 1

    assert np.max(y_true_org) <= 1 or np.max(y_pred_org) <= 1

    intersection_o = np.count_nonzero(y_true_org.astype(np.bool) & y_pred_org.astype(np.bool))
    size_i1 = np.count_nonzero(y_true_org == 1)
    size_i2 = np.count_nonzero(y_pred_org == 1)

    if size_i1 == 0 and size_i2 == 0:
        dice_organ = 1.0
    else:
        dice_organ = 2. * intersection_o / float(size_i1 + size_i2)

    return dice_organ

def dice_coefficient_for_class(gt_seg, pred_seg, k):

    # dice = np.sum(seg[gt == k]) * 2.0 / (np.sum(seg) + np.sum(gt))

    y_true_f = np.reshape(gt_seg, [-1])
    y_pred_f = np.reshape(pred_seg, [-1])

    intersection = np.sum(y_pred_f[y_true_f==k])  # y_true_f * y_pred_f
    union = np.sum(y_true_f) + np.sum(y_pred_f)

    if np.sum(y_true_f) == 0 and np.sum(y_pred_f) == 0:
        dice = 1
    else:
        dice = (2. * intersection) / (union + 0.00001)

    return dice



def dice_coefficient(gt_seg, pred_seg):

    """
    Calculate the dice for pancreas-segmentation + tumor-segmentation
    :param gt_seg:
    :param pred_seg:
    :return:
    """

    y_true_f = np.reshape(gt_seg, [-1])
    y_pred_f = np.reshape(pred_seg, [-1])

    intersection = np.sum(np.logical_and(y_pred_f, y_true_f))   # y_true_f * y_pred_f
    union = np.sum(y_true_f) + np.sum(y_pred_f)

    if np.sum(y_true_f) == 0 and np.sum(y_pred_f) == 0:
        dice = 1
    else:
        dice = (2. * intersection) / (union + 0.00001)

    return dice


def pixel_accuracy(eval_segm, gt_segm):
    '''
    sum_i(n_ii) / sum_i(t_i)
    '''

    check_size(eval_segm, gt_segm)

    cl, n_cl = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    sum_n_ii = 0
    sum_t_i = 0

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        sum_n_ii += np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        sum_t_i += np.sum(curr_gt_mask)

    if (sum_t_i == 0):
        pixel_accuracy_ = 0
    else:
        pixel_accuracy_ = sum_n_ii / sum_t_i

    return pixel_accuracy_


def mean_accuracy(eval_segm, gt_segm):
    '''
    (1/n_cl) sum_i(n_ii/t_i)
    '''

    check_size(eval_segm, gt_segm)

    cl, n_cl = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    accuracy = list([0]) * n_cl

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i = np.sum(curr_gt_mask)

        if (t_i != 0):
            accuracy[i] = n_ii / t_i

    mean_accuracy_ = np.mean(accuracy)
    return mean_accuracy_


def mean_IU(eval_segm, gt_segm):
    '''
    (1/n_cl) * sum_i(n_ii / (t_i + sum_j(n_ji) - n_ii))
    '''

    check_size(eval_segm, gt_segm)

    cl, n_cl = union_classes(eval_segm, gt_segm)
    _, n_cl_gt = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    IU = list([0]) * n_cl

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        if (np.sum(curr_eval_mask) == 0) or (np.sum(curr_gt_mask) == 0):
            continue

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i = np.sum(curr_gt_mask)
        n_ij = np.sum(curr_eval_mask)

        IU[i] = n_ii / (t_i + n_ij - n_ii)

    mean_IU_ = np.sum(IU) / n_cl_gt
    return mean_IU_


def frequency_weighted_IU(eval_segm, gt_segm):
    '''
    sum_k(t_k)^(-1) * sum_i((t_i*n_ii)/(t_i + sum_j(n_ji) - n_ii))
    '''

    check_size(eval_segm, gt_segm)

    cl, n_cl = union_classes(eval_segm, gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    frequency_weighted_IU_ = list([0]) * n_cl

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        if (np.sum(curr_eval_mask) == 0) or (np.sum(curr_gt_mask) == 0):
            continue

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i = np.sum(curr_gt_mask)
        n_ij = np.sum(curr_eval_mask)

        frequency_weighted_IU_[i] = (t_i * n_ii) / (t_i + n_ij - n_ii)

    sum_k_t_k = get_pixel_area(eval_segm)

    frequency_weighted_IU_ = np.sum(frequency_weighted_IU_) / sum_k_t_k
    return frequency_weighted_IU_


'''
Auxiliary functions used during evaluation.
'''


def get_pixel_area(segm):
    return segm.shape[0] * segm.shape[1]


def extract_both_masks(eval_segm, gt_segm, cl, n_cl):

    eval_mask = extract_masks(eval_segm, cl, n_cl)
    gt_mask = extract_masks(gt_segm, cl, n_cl)

    return eval_mask, gt_mask


def extract_classes(segm):
    cl = np.unique(segm)
    n_cl = len(cl)

    return cl, n_cl


def union_classes(eval_segm, gt_segm):
    eval_cl, _ = extract_classes(eval_segm)
    gt_cl, _ = extract_classes(gt_segm)

    cl = np.union1d(eval_cl, gt_cl)
    n_cl = len(cl)

    return cl, n_cl


def extract_masks(segm, cl, n_cl):

    h, w = segm_size(segm)
    masks = np.zeros((n_cl, h, w))

    for i, c in enumerate(cl):
        masks[:, :, i] = segm == c

    return masks


def segm_size(segm):
    try:
        height = segm.shape[0]
        width = segm.shape[1]
    except IndexError:
        raise

    return height, width


def check_size(eval_segm, gt_segm):
    h_e, w_e = segm_size(eval_segm)
    h_g, w_g = segm_size(gt_segm)

    if (h_e != h_g) or (w_e != w_g):
        print("false dimension!")