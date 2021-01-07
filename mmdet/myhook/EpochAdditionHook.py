# Copyright (c) Open-MMLab. All rights reserved.
from mmcv.runner.hooks.hook import HOOKS, Hook

@HOOKS.register_module()
class EpochAdditionHook(Hook):
    def before_train_epoch(self, runner):
    # def before_train_iter(self, runner):
        runner.model.module.neck.epoch_num = runner._epoch
        # runner.model.module.neck.epoch_num += 1
        # runner.model.(runner.epoch)
