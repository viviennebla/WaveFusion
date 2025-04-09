import os

import matplotlib
from batchgenerators.utilities.file_and_folder_operations import join

matplotlib.use('agg')
import seaborn as sns
import matplotlib.pyplot as plt


class nnUNetLogger(object):
    """
    This class is really trivial. Don't expect cool functionality here. This is my makeshift solution to problems
    arising from out-of-sync epoch numbers and numbers of logged loss values. It also simplifies the trainer class a
    little

    YOU MUST LOG EXACTLY ONE VALUE PER EPOCH FOR EACH OF THE LOGGING ITEMS! DONT FUCK IT UP
    """
    def __init__(self, verbose: bool = False):
        self.my_fantastic_logging = {
            'mean_fg_dice': list(),
            'ema_fg_dice': list(),
            'dice_per_class_or_region': list(),
            'train_losses': list(),
            'val_losses': list(),
            'lrs': list(),
            'epoch_start_timestamps': list(),
            'epoch_end_timestamps': list(),
            'ious_per_class':list(),
            'train_main_losses':list(),
            'train_aux_losses':list(),
        }
        self.verbose = verbose
        # shut up, this logging is great

    def log(self, key, value, epoch: int):
        """
        sometimes shit gets messed up. We try to catch that here
        """
        assert key in self.my_fantastic_logging.keys() and isinstance(self.my_fantastic_logging[key], list), \
            'This function is only intended to log stuff to lists and to have one entry per epoch'

        if self.verbose: print(f'logging {key}: {value} for epoch {epoch}')

        if len(self.my_fantastic_logging[key]) < (epoch + 1):
            self.my_fantastic_logging[key].append(value)
        else:
            assert len(self.my_fantastic_logging[key]) == (epoch + 1), 'something went horribly wrong. My logging ' \
                                                                       'lists length is off by more than 1'
            print(f'maybe some logging issue!? logging {key} and {value}')
            self.my_fantastic_logging[key][epoch] = value

        # handle the ema_fg_dice special case! It is automatically logged when we add a new mean_fg_dice
        if key == 'mean_fg_dice':
            if(epoch-1<len(self.my_fantastic_logging['mean_fg_dice'])):
                new_ema_pseudo_dice = self.my_fantastic_logging['ema_fg_dice'][epoch - 1] * 0.9 + 0.1 * value \
                    if len(self.my_fantastic_logging['ema_fg_dice']) > 0 else value
            else:
                new_ema_pseudo_dice = value
                print(f"current epoch:{epoch},current length of ema_pseudo_dice:{len(self.my_fantastic_logging['ema_fg_dice'])}")
            self.log('ema_fg_dice', new_ema_pseudo_dice, epoch)

    def plot_progress_png(self, output_folder):
        # we infer the epoch form our internal logging
        epoch = max([len(i) for i in self.my_fantastic_logging.values()]) - 1  # lists of epoch 0 have len 1
        if epoch < 0:
            print("No epochs found in the logs.")
            return

        sns.set(font_scale=2.5)
        fig, ax_all = plt.subplots(3, 1, figsize=(30, 54))

        # Check if the necessary data is available for plotting
        for key in ['train_losses', 'val_losses', 'mean_fg_dice', 'ema_fg_dice', 'epoch_end_timestamps',
                    'epoch_start_timestamps', 'lrs', 'ious_per_class']:  # 添加了 'ious_per_class'
            if len(self.my_fantastic_logging[key]) != epoch + 1:
                print(
                    f"Error: Incomplete data for {key}. Expected {epoch + 1}, found {len(self.my_fantastic_logging[key])}.")
                plt.close(fig)
                return

        # regular progress.png as we are used to from previous nnU-Net versions
        # 改进后的主损失和辅助损失绘制逻辑
        ax = ax_all[0]
        ax2 = ax.twinx()
        x_values = list(range(epoch + 1))

        # 绘制损失曲线
        ax.plot(x_values, self.my_fantastic_logging['train_losses'][:epoch + 1],
                color='blue', ls='-', label="Train Loss", linewidth=4)
        ax.plot(x_values, self.my_fantastic_logging['val_losses'][:epoch + 1],
                color='red', ls='-', label="Validation Loss", linewidth=4)

        if len(self.my_fantastic_logging['train_main_losses']):
            ax.plot(x_values, self.my_fantastic_logging['train_main_losses'],
                    label='Main Loss', color='purple', linewidth=3, alpha=0.7)
            ax.plot(x_values, self.my_fantastic_logging['train_aux_losses'],
                    label='Aux Loss', color='purple', ls='dotted',linewidth=3, alpha=0.7)

        # 绘制 Dice 和 IoU 指标
        ax2.plot(x_values, self.my_fantastic_logging['mean_fg_dice'][:epoch + 1],
                 color='green', ls='dotted', label="Pseudo Dice", linewidth=3, alpha=0.8)
        ax2.plot(x_values, self.my_fantastic_logging['ema_fg_dice'][:epoch + 1],
                 color='green', ls='-', label="Pseudo Dice (Mov. Avg.)", linewidth=4, alpha=0.8)
        ax2.plot(x_values, self.my_fantastic_logging['ious_per_class'][:epoch + 1],
                 color='orange', ls='--', label="IoU per Class", linewidth=3, alpha=0.8)

        # 调整 Y 轴范围
        # ax.set_ylim([-1, 2])  # 设置损失值的合理范围
        # ax2.set_ylim([0.5, 1.0])  # 针对 Dice 和 IoU 的范围

        # 设置轴标签
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax2.set_ylabel("Pseudo Dice / IoU")

        # 添加网格线和图例
        ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')

        # epoch times to see whether the training speed is consistent (inconsistent means there are other jobs
        # clogging up the system)
        ax = ax_all[1]
        ax.plot(x_values, [i - j for i, j in zip(self.my_fantastic_logging['epoch_end_timestamps'][:epoch + 1],
                                                 self.my_fantastic_logging['epoch_start_timestamps'])][:epoch + 1],
                color='b', ls='-', label="epoch duration", linewidth=4)
        ylim = [0] + [ax.get_ylim()[1]]
        ax.set(ylim=ylim)
        ax.set_xlabel("epoch")
        ax.set_ylabel("time [s]")
        ax.legend(loc=(0, 1))

        # learning rate
        ax = ax_all[2]
        ax.plot(x_values, self.my_fantastic_logging['lrs'][:epoch + 1], color='b', ls='-', label="learning rate",
                linewidth=4)
        ax.set_xlabel("epoch")
        ax.set_ylabel("learning rate")
        ax.legend(loc=(0, 1))

        plt.tight_layout()

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        fig.savefig(join(output_folder, "progress.png"))
        plt.close()


    def get_checkpoint(self):
        return self.my_fantastic_logging

    def load_checkpoint(self, checkpoint: dict):
        self.my_fantastic_logging = checkpoint
