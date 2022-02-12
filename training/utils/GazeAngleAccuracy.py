import torch


class GazeAngleAccuracy(object):

    def __call__(self, batch_y_pred, batch_y_true):
        y_pred = batch_y_pred.cpu().detach()
        y_true = batch_y_true.cpu().detach()

        pred_x = -1 * torch.cos(y_pred[:, 0]) * torch.sin(y_pred[:, 1])
        pred_y = -1 * torch.sin(y_pred[:, 0])
        pred_z = -1 * torch.cos(y_pred[:, 0]) * torch.cos(y_pred[:, 1])
        pred = torch.vstack([pred_x, pred_y, pred_z]).T
        pred_norm = (pred.T / torch.norm(pred, dim=1)).T

        true_x = -1 * torch.cos(y_true[:, 0]) * torch.sin(y_true[:, 1])
        true_y = -1 * torch.sin(y_true[:, 0])
        true_z = -1 * torch.cos(y_true[:, 0]) * torch.cos(y_true[:, 1])
        gt = torch.vstack([true_x, true_y, true_z]).T
        gt_norm = (gt.T / torch.norm(gt, dim=1)).T

        diff = torch.sum(pred_norm * gt_norm, dim=1)
        diff_angle = torch.arccos(diff)
        diff_deg = torch.rad2deg(diff_angle)

        acc = torch.mean(diff_deg)
        return acc
