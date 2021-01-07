# Copyright (c) Open-MMLab. All rights reserved.
from mmcv.runner.hooks.hook import HOOKS, Hook

@HOOKS.register_module()
class AlternateTrainingHook(Hook):
    # def before_train_iter(self, runner):
    def before_train_epoch(self, runner):
        runner.model.module.neck.epoch_num = runner._epoch
        # if runner._iter < 2000:
        #     for param in runner.model.module.neck.parameters():
        #         param.requires_grad = False
        #     for param_0 in runner.model.module.neck.attention_block_0.parameters():
        #         param_0.requires_grad = True
        #     for param_1 in runner.model.module.neck.attention_block_1.parameters():
        #         param_1.requires_grad = True
        #     for param_2 in runner.model.module.neck.attention_block_2.parameters():
        #         param_2.requires_grad = True
        #     for param_3 in runner.model.module.neck.attention_block_3.parameters():
        #         param_3.requires_grad = True
        # if runner._iter > 2000:
        #     for param in runner.model.module.neck.parameters():
        #         param.requires_grad = False
        #     for param_0 in runner.model.module.neck.attention_block_0.parameters():
        #         param_0.requires_grad = False
        #     for param_1 in runner.model.module.neck.attention_block_1.parameters():
        #         param_1.requires_grad = False
        #     for param_2 in runner.model.module.neck.attention_block_2.parameters():
        #         param_2.requires_grad = False
        #     for param_3 in runner.model.module.neck.attention_block_3.parameters():
        #         param_3.requires_grad = False

        # runner.model.module.neck.attention_block_0.requires_grad = True
        # runner.model.module.neck.attention_block_1.requires_grad = True
        # runner.model.module.neck.attention_block_2.requires_grad = True
        # runner.model.module.neck.attention_block_3.requires_grad = True

        if runner._epoch % 2 == 0:
            for param in runner.model.module.neck.parameters():
                param.requires_grad = True
            for param_0 in runner.model.module.neck.attention_block_0.parameters():
                param_0.requires_grad = False
            for param_1 in runner.model.module.neck.attention_block_1.parameters():
                param_1.requires_grad = False
            for param_2 in runner.model.module.neck.attention_block_2.parameters():
                param_2.requires_grad = False
            for param_3 in runner.model.module.neck.attention_block_3.parameters():
                param_3.requires_grad = False
        else:
            for param in runner.model.module.neck.parameters():
                param.requires_grad = False
            for param_0 in runner.model.module.neck.attention_block_0.parameters():
                param_0.requires_grad = True
            for param_1 in runner.model.module.neck.attention_block_1.parameters():
                param_1.requires_grad = True
            for param_2 in runner.model.module.neck.attention_block_2.parameters():
                param_2.requires_grad = True
            for param_3 in runner.model.module.neck.attention_block_3.parameters():
                param_3.requires_grad = True
        # elif runner._iter == 100:
        #     for param in runner.model.module.neck.parameters():
        #         param.requires_grad = False
        #     runner.model.module.neck.attention_block_0.requires_grad = True
        #     runner.model.module.neck.attention_block_1.requires_grad = True
        #     runner.model.module.neck.attention_block_2.requires_grad = True
        #     runner.model.module.neck.attention_block_3.requires_grad = True
        #
        # elif runner._iter == 150:
        # # if runner._epoch % 2 == 0:
        #     for param in runner.model.module.neck.parameters():
        #         param.requires_grad = True
        #     runner.model.module.neck.attention_block_0.requires_grad = False
        #     runner.model.module.neck.attention_block_1.requires_grad = False
        #     runner.model.module.neck.attention_block_2.requires_grad = False
        #     runner.model.module.neck.attention_block_3.requires_grad = False
        #
        # elif runner._iter == 200:
        #     for param in runner.model.module.neck.parameters():
        #         param.requires_grad = False
        #     runner.model.module.neck.attention_block_0.requires_grad = True
        #     runner.model.module.neck.attention_block_1.requires_grad = True
        #     runner.model.module.neck.attention_block_2.requires_grad = True
        #     runner.model.module.neck.attention_block_3.requires_grad = True
        # elif runner._iter == 250:
        #     for param in runner.model.module.neck.parameters():
        #         param.requires_grad = True
        #     runner.model.module.neck.attention_block_0.requires_grad = False
        #     runner.model.module.neck.attention_block_1.requires_grad = False
        #     runner.model.module.neck.attention_block_2.requires_grad = False
        #     runner.model.module.neck.attention_block_3.requires_grad = False