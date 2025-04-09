from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from torch import nn

# 假设 get_UKAN_from_plans 函数已经实现，并且可以正确构造 UKAN 网络
from nnunetv2.nets.WFM import get_wfm_from_plans  # 确保从你的模块中导入 get_UKAN_from_plans 函数


class nnUNetTrainerWFM(nnUNetTrainer):
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        model = get_wfm_from_plans(plans_manager, dataset_json, configuration_manager,
                                                 num_input_channels, deep_supervision=enable_deep_supervision)

        # print("FFM: {}".format(model))
        print("UNet with mamba and wavfusion")

        return model
