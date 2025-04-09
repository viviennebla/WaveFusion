from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from torch import nn

from monai.networks.nets import SwinUNETR

class SwinUNETRTrainer(nnUNetTrainer):
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        label_manager = plans_manager.get_label_manager(dataset_json)
        img_size = configuration_manager.patch_size
        spatial_dims = len(img_size)

        model = SwinUNETR(
            in_channels=num_input_channels,
            out_channels=label_manager.num_segmentation_heads,
            img_size=img_size,
            num_heads=(3, 6, 12, 24),
            norm_name="instance",
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            normalize=True,
            use_checkpoint=False,
            spatial_dims=spatial_dims,
            downsample="merging",
            use_v2=False,
            depths=(2, 2, 2, 2),
            feature_size=48
        )
        print("this is SwinUNETR")

        return model