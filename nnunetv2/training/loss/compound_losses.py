import torch
from nnunetv2.training.loss.dice import SoftDiceLoss, MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss, TopKLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1
from torch import nn

import torch.nn.functional as F


class DC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
                 dice_class=SoftDiceLoss):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_CE_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

        print(f"-------------{self.ignore_label}---------------")
        print("this is DC_and_CE_loss")

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = (target != self.ignore_label).bool()
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        # print("-----------aaaaaaaaaa----------------")
        # print(net_output.shape)
        # print(target_dice.shape)
        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0].long()) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


class DC_and_BCE_loss(nn.Module):
    def __init__(self, bce_kwargs, soft_dice_kwargs, weight_ce=1, weight_dice=1, use_ignore_label: bool = False,
                 dice_class=MemoryEfficientSoftDiceLoss):
        """
        DO NOT APPLY NONLINEARITY IN YOUR NETWORK!

        target mut be one hot encoded
        IMPORTANT: We assume use_ignore_label is located in target[:, -1]!!!

        :param soft_dice_kwargs:
        :param bce_kwargs:
        :param aggregate:
        """
        super(DC_and_BCE_loss, self).__init__()
        if use_ignore_label:
            bce_kwargs['reduction'] = 'none'

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.use_ignore_label = use_ignore_label

        self.ce = nn.BCEWithLogitsLoss(**bce_kwargs)
        self.dc = dice_class(apply_nonlin=torch.sigmoid, **soft_dice_kwargs)
        print(f"-------------{self.use_ignore_label}---------------")
        print("this is DC_and_BCE_loss")

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        if self.use_ignore_label:
            # target is one hot encoded here. invert it so that it is True wherever we can compute the loss
            mask = (1 - target[:, -1:]).bool()
            # remove ignore channel now that we have the mask
            target_regions = torch.clone(target[:, :-1])
        else:
            target_regions = target
            mask = None

        dc_loss = self.dc(net_output, target_regions, loss_mask=mask)
        if mask is not None:
            ce_loss = (self.ce(net_output, target_regions) * mask).sum() / torch.clip(mask.sum(), min=1e-8)
        else:
            ce_loss = self.ce(net_output, target_regions)
        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


class DC_and_topk_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super().__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = TopKLoss(**ce_kwargs)
        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = (target != self.ignore_label).bool()
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


class EdgeLoss(nn.Module):
    def __init__(self):
        """
        :param edge_weight: 用于调整边缘损失的权重
        """
        super(EdgeLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor):
        """
        :param prediction: 辅助尺度的分割结果
        :param target: 标签，假设已生成对应的边缘掩码
        :return: 边缘损失
        """
        # 假设边缘掩码通过 target 提取（需要预处理生成边缘区域）
        edge_mask = self.extract_edges(target)

        # 计算边缘损失
        edge_loss = self.bce(prediction, edge_mask)
        return edge_loss

    def extract_edges(self, target: torch.Tensor):
        """
        简单的边缘提取，可以替换为更复杂的方法
        :param target: 原始分割标签
        :return: 边缘掩码
        """
        from torch.nn.functional import conv2d

        # 简单 Sobel 算子提取边缘（二维情况）
        kernel = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        kernel = kernel.to(target.device)
        edge_mask = torch.abs(conv2d(target.float(), kernel, padding=1))
        edge_mask = (edge_mask > 0).float()  # 二值化
        return edge_mask


class HybridLoss(nn.Module):
    def __init__(self, weight_main=1.0, weight_aux=0.5, ce_kwargs={}, soft_dice_kwargs=None, edge_kwargs=None):
        """
        :param weight_main: 权重，用于主损失（BCE + Dice）
        :param weight_aux: 权重，用于辅助损失（边缘损失）
        :param bce_kwargs: BCEWithLogitsLoss 的参数
        :param dice_kwargs: SoftDiceLoss 的参数
        :param edge_kwargs: EdgeLoss 的参数
        """
        super(HybridLoss, self).__init__()
        self.weight_main = weight_main
        self.weight_aux = weight_aux

        # 主损失的 BCE 和 Dice
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dice = MemoryEfficientSoftDiceLoss(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

        # 边缘损失
        self.edge_loss = EdgeLoss(**(edge_kwargs or {}))  # 假设已定义 EdgeLoss 类

    def change_weight(self, weight_main, weight_aux):
        self.weight_main = weight_main
        self.weight_aux = weight_aux

    def forward(self, outputs: torch.tensor, target: torch.tensor):
        """
        :param outputs: 分割网络的输出，列表形式，包含三个尺度的预测结果
                        outputs[0]: 主尺度预测结果
                        outputs[1]: 辅助尺度1预测结果
                        outputs[2]: 辅助尺度2预测结果
        :param target: 标签，形状为 (B, C, H, W) 或 (B, C, H, W, D)
        :return: 加权总损失
        """

        # outputs = outputs[:3]
        # target = target[:3]

        t = target
        mask_class_1 = torch.eq(t, 1).float()
        mask_class_0 = torch.eq(t, 0).float()
        target = torch.cat((mask_class_1, mask_class_0), dim=1)

        main_bce_loss = self.ce(outputs, target[:, 0].long())
        # 主尺度的损失（BCE + Dice）
        main_dice_loss = self.dice(outputs, target)
        main_loss = main_bce_loss + main_dice_loss

        # 辅助尺度的边缘损失
        aux_loss1 = self.edge_loss(outputs[:,1,:,:].unsqueeze(1), target)
        aux_loss = aux_loss1/10

        # print(type(main_loss))
        # print(type(aux_loss))

        # 加权损失
        total_loss = self.weight_main * main_loss + self.weight_aux * aux_loss

        return total_loss, main_loss, aux_loss
