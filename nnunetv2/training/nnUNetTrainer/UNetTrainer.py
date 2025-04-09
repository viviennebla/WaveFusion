from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from torch import nn

# 假设 get_UKAN_from_plans 函数已经实现，并且可以正确构造 UKAN 网络
from nnunetv2.nets.VanillaUNet import get_unet_from_plans  # 确保从你的模块中导入 get_UKAN_from_plans 函数


class UNetTrainer(nnUNetTrainer):
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        model = get_unet_from_plans(plans_manager, dataset_json, configuration_manager,
                                                 num_input_channels, deep_supervision=enable_deep_supervision)

        print("Vanilla UNet")
        # print("UNet: {}".format(model))

        return model