from math import log10
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
import logging


def PSNR(mse, peak=1.):
    return 10 * log10((peak ** 2) / mse)


class SegMetric:
    def __init__(self, values=0.):
        assert isinstance(values, dict)
        self.miou = values.miou
        self.oa = values.get('oa', None) 
        self.miou = values.miou
        self.miou = values.miou


    def better_than(self, other):
        if self.acc > other.acc:
            return True
        else:
            return False

    def state_dict(self):
        _dict = dict()
        _dict['acc'] = self.acc
        return _dict


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ConfusionMatrix:
    """Accumulate a confusion matrix for a classification task.
    ignore_index only supports index <0, or > num_classes 
    """

    def __init__(self, num_classes, ignore_index=None):
        self.value = 0
        self.num_classes = num_classes
        self.virtual_num_classes = num_classes + 1 if ignore_index is not None else num_classes
        self.ignore_index = ignore_index

    @torch.no_grad()
    def update(self, pred, true): 
        """Update the confusion matrix with the given predictions."""
        true = true.flatten()
        pred = pred.flatten()
        if self.ignore_index is not None:
            if (true == self.ignore_index).sum() > 0:
                pred[true == self.ignore_index] = self.virtual_num_classes -1
                true[true == self.ignore_index] = self.virtual_num_classes -1
        unique_mapping = true.flatten() * self.virtual_num_classes + pred.flatten()
        bins = torch.bincount(unique_mapping, minlength=self.virtual_num_classes**2)
        self.value += bins.view(self.virtual_num_classes, self.virtual_num_classes)[:self.num_classes, :self.num_classes]

    def reset(self):
        """Reset all accumulated values."""
        self.value = 0

    @property
    def tp(self):
        """Get the true positive samples per-class."""
        return self.value.diag()
    
    @property
    def actual(self):
        """Get the false negative samples per-class."""
        return self.value.sum(dim=1)

    @property
    def predicted(self):
        """Get the false negative samples per-class."""
        return self.value.sum(dim=0)
    
    @property
    def fn(self):
        """Get the false negative samples per-class."""
        return self.actual - self.tp

    @property
    def fp(self):
        """Get the false positive samples per-class."""
        return self.predicted - self.tp

    @property
    def tn(self):
        """Get the true negative samples per-class."""
        actual = self.actual
        predicted = self.predicted
        return actual.sum() + self.tp - (actual + predicted)

    @property
    def count(self):  # a.k.a. actual positive class
        """Get the number of samples per-class."""
        # return self.tp + self.fn
        return self.value.sum(dim=1)

    @property
    def frequency(self):
        """Get the per-class frequency."""
        # we avoid dividing by zero using: max(denomenator, 1)
        # return self.count / self.total.clamp(min=1)
        count = self.value.sum(dim=1)
        return count / count.sum().clamp(min=1)

    @property
    def total(self):
        """Get the total number of samples."""
        return self.value.sum()

    @property
    def overall_accuray(self):
        return self.tp.sum() / self.total

    @property
    def union(self):
        return self.value.sum(dim=0) + self.value.sum(dim=1) - self.value.diag()

    def all_acc(self):
        return self.cal_acc(self.tp, self.count)

    @staticmethod
    def cal_acc(tp, count):
        acc_per_cls = tp / count.clamp(min=1) * 100
        over_all_acc = tp.sum() / count.sum() * 100
        macc = torch.mean(acc_per_cls)  # class accuracy
        return macc.item(), over_all_acc.item(), acc_per_cls.cpu().numpy()

    @staticmethod
    def print_acc(accs):
        out = '\n    Class  ' + '   Acc  '
        for i, values in enumerate(accs):
            out += '\n' + str(i).rjust(8) + f'{values.item():.2f}'.rjust(8)
        out += '\n' + '-' * 20
        out += '\n' + '   Mean  ' + f'{torch.mean(accs).item():.2f}'.rjust(8)
        logging.info(out)

    def all_metrics(self):
        tp, fp, fn = self.tp, self.fp, self.fn,  
  
        iou_per_cls = tp / (tp + fp + fn).clamp(min=1) * 100
        acc_per_cls = tp / self.count.clamp(min=1) * 100
        over_all_acc = tp.sum() / self.total * 100

        miou = torch.mean(iou_per_cls)
        macc = torch.mean(acc_per_cls)  # class accuracy
        return miou.item(), macc.item(), over_all_acc.item(), iou_per_cls.cpu().numpy(), acc_per_cls.cpu().numpy()


def get_mious(tp, union, count):
    iou_per_cls = (tp + 1e-10) / (union + 1e-10) * 100
    acc_per_cls = (tp + 1e-10) / (count + 1e-10) * 100 
    over_all_acc = tp.sum() / count.sum() * 100

    miou = torch.mean(iou_per_cls)
    macc = torch.mean(acc_per_cls)  # class accuracy
    return miou.item(), macc.item(), over_all_acc.item(), iou_per_cls.cpu().numpy(), acc_per_cls.cpu().numpy()


def partnet_metrics(num_classes, num_parts, objects, preds, targets):
    """

    Args:
        num_classes:
        num_parts:
        objects: [int]
        preds:[(num_parts,num_points)]
        targets: [(num_points)]

    Returns:

    """
    shape_iou_tot = [0.0] * num_classes
    shape_iou_cnt = [0] * num_classes
    part_intersect = [np.zeros((num_parts[o_l]), dtype=np.float32) for o_l in range(num_classes)]
    part_union = [np.zeros((num_parts[o_l]), dtype=np.float32) + 1e-6 for o_l in range(num_classes)]

    for obj, cur_pred, cur_gt in zip(objects, preds, targets):
        cur_num_parts = num_parts[obj]
        cur_pred = np.argmax(cur_pred[1:, :], axis=0) + 1
        cur_pred[cur_gt == 0] = 0
        cur_shape_iou_tot = 0.0
        cur_shape_iou_cnt = 0
        for j in range(1, cur_num_parts):
            cur_gt_mask = (cur_gt == j)
            cur_pred_mask = (cur_pred == j)

            has_gt = (np.sum(cur_gt_mask) > 0)
            has_pred = (np.sum(cur_pred_mask) > 0)

            if has_gt or has_pred:
                intersect = np.sum(cur_gt_mask & cur_pred_mask)
                union = np.sum(cur_gt_mask | cur_pred_mask)
                iou = intersect / union

                cur_shape_iou_tot += iou
                cur_shape_iou_cnt += 1

                part_intersect[obj][j] += intersect
                part_union[obj][j] += union
        if cur_shape_iou_cnt > 0:
            cur_shape_miou = cur_shape_iou_tot / cur_shape_iou_cnt
            shape_iou_tot[obj] += cur_shape_miou
            shape_iou_cnt[obj] += 1

    msIoU = [shape_iou_tot[o_l] / shape_iou_cnt[o_l] for o_l in range(num_classes)]
    part_iou = [np.divide(part_intersect[o_l][1:], part_union[o_l][1:]) for o_l in range(num_classes)]
    mpIoU = [np.mean(part_iou[o_l]) for o_l in range(num_classes)]

    # Print instance mean
    mmsIoU = np.mean(np.array(msIoU))
    mmpIoU = np.mean(mpIoU)

    return msIoU, mpIoU, mmsIoU, mmpIoU


def IoU_from_confusions(confusions):
    """
    Computes IoU from confusion matrices.
    :param confusions: ([..., n_c, n_c] np.int32). Can be any dimension, the confusion matrices should be described by
    the last axes. n_c = number of classes
    :param ignore_unclassified: (bool). True if the the first class should be ignored in the results
    :return: ([..., n_c] np.float32) IoU score
    """

    # Compute TP, FP, FN. This assume that the second to last axis counts the truths (like the first axis of a
    # confusion matrix), and that the last axis counts the predictions (like the second axis of a confusion matrix)
    TP = np.diagonal(confusions, axis1=-2, axis2=-1)
    TP_plus_FN = np.sum(confusions, axis=-1)
    TP_plus_FP = np.sum(confusions, axis=-2)

    # Compute IoU
    IoU = TP / (TP_plus_FP + TP_plus_FN - TP + 1e-6)

    # Compute miou with only the actual classes
    mask = TP_plus_FN < 1e-3
    counts = np.sum(1 - mask, axis=-1, keepdims=True)
    miou = np.sum(IoU, axis=-1, keepdims=True) / (counts + 1e-6)

    # If class is absent, place miou in place of 0 IoU to get the actual mean later
    IoU += mask * miou

    return IoU


def shapenetpart_metrics(num_classes, num_parts, objects, preds, targets, masks):
    """
    Args:
        num_classes:
        num_parts:
        objects: [int]
        preds:[(num_parts,num_points)]
        targets: [(num_points)]
        masks: [(num_points)]
    """
    total_correct = 0.0
    total_seen = 0.0
    Confs = []
    for obj, cur_pred, cur_gt, cur_mask in zip(objects, preds, targets, masks):
        obj = int(obj)
        cur_num_parts = num_parts[obj]
        cur_pred = np.argmax(cur_pred, axis=0)
        cur_pred = cur_pred[cur_mask]
        cur_gt = cur_gt[cur_mask]
        correct = np.sum(cur_pred == cur_gt)
        total_correct += correct
        total_seen += cur_pred.shape[0]
        parts = [j for j in range(cur_num_parts)]
        Confs += [confusion_matrix(cur_gt, cur_pred, labels=parts)]

    Confs = np.array(Confs)
    obj_mious = []
    objects = np.asarray(objects)
    for l in range(num_classes):
        obj_inds = np.where(objects == l)[0]
        obj_confs = np.stack(Confs[obj_inds])
        obj_IoUs = IoU_from_confusions(obj_confs)
        obj_mious += [np.mean(obj_IoUs, axis=-1)]

    objs_average = [np.mean(mious) for mious in obj_mious]
    instance_average = np.mean(np.hstack(obj_mious))
    class_average = np.mean(objs_average)
    acc = total_correct / total_seen

    print('Objs | Inst | Air  Bag  Cap  Car  Cha  Ear  Gui  Kni  Lam  Lap  Mot  Mug  Pis  Roc  Ska  Tab')
    print('-----|------|--------------------------------------------------------------------------------')

    s = '{:4.1f} | {:4.1f} | '.format(100 * class_average, 100 * instance_average)
    for Amiou in objs_average:
        s += '{:4.1f} '.format(100 * Amiou)
    print(s + '\n')
    return acc, objs_average, class_average, instance_average
