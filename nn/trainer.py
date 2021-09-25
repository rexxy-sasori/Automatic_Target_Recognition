"""
Author: Rex Geng

trainer API. The abstract Trainer class defines the template.
The child classes defines the concrete computation steps during training.
The typical backprop trainer has been implemented. If new training techniques is needed
Simply inherit the parent Trainer class and define the computation step.
"""

import copy
import operator
import os
import time
from datetime import datetime

import torch

import IO.plotter as plotter
from IO.dconst import EPOCH_FMT_STR
from nn.models import __MODELS_INPUTS__
from nn.utils import AverageMeter, ProgressMeter, Profiler, ProfilerResult


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        if type(output) is tuple:
            _, _, output = output
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res, pred[0, :]


class NNResult:
    def __init__(
            self,
            profiler_result=None,
            trainer_usr_configs=None,
            test_accuracy=-1,
            validation_accuracy=-1,
    ):
        self.profiler_result = profiler_result
        self.trainer_usr_configs = trainer_usr_configs

        self.test_accuracy = test_accuracy
        self.validation_accuracy = validation_accuracy

        self.train_accuracy_history = []
        self.train_loss_history = []
        self.benchmarking_accuracy_history = []
        self.benchmarking_loss_history = []


class Trainer:
    def __init__(self, trainer_configs):
        self.trainer_configs = trainer_configs
        self.__setup__(trainer_configs)

    def __setup__(self, trainer_configs):
        self.metric_func = operator.lt
        self.curr_metric = 100

        self.batch_time = AverageMeter('Time', ':6.3f')
        self.data_time = AverageMeter('Data Loading', ':6.3f')
        self.data_copy_time = AverageMeter('Data to GPU', ':6.3f')
        self.losses = AverageMeter('Loss', ':.4e')
        self.top1 = AverageMeter('Acc@1', ':6.2f')
        self.progress = ProgressMeter(
            len(trainer_configs.train_loader),
            [self.batch_time, self.data_time, self.data_copy_time, self.losses, self.top1],
            prefix=EPOCH_FMT_STR.format(0)
        )

        if not os.path.exists(self.trainer_configs.result_dir):
            os.makedirs(self.trainer_configs.result_dir)

        self.benchmarking_loader = self.trainer_configs.validation_loader \
            if self.trainer_configs.validation_loader is not None else self.trainer_configs.test_loader
        self.use_validationset = self.trainer_configs.validation_loader is not None

        now = datetime.now()
        dt_str = now.strftime("_%Y_%m_%d_%H_%M_%S")

        model_dst_file_name = str(self.trainer_configs.model)
        model_dst_file_name = model_dst_file_name.replace('.pt.tar', dt_str + '.pt.tar')
        if self.trainer_configs.usr.dataset.aug:
            model_dst_file_name = model_dst_file_name.replace('.pt.tar', '_aug.pt.tar')
        self.model_dst_path = os.path.join(self.trainer_configs.result_dir, model_dst_file_name)

        # copy model such that the model to be trained is not registered (cuda could run out memory if model is big!!)
        if self.trainer_configs.usr.profile_complexity:
            profiler = Profiler(copy.deepcopy(trainer_configs.model))

            if hasattr(trainer_configs.model, 'R'):
                input_tensor = torch.randn(1, 1, trainer_configs.model.R, trainer_configs.model.R)
            else:
                input_tensor = __MODELS_INPUTS__.get(type(trainer_configs.model))

            profiler_result = profiler.profile(inputs=input_tensor)
        else:
            profiler_result = ProfilerResult()

        self.nnresult = NNResult(profiler_result, trainer_configs.usr)
        if self.trainer_configs.resume_from_best or self.trainer_configs.eval_model:
            self.load_ckpt()

        if self.trainer_configs.resume_from_best:
            test_acc, test_loss = self.eval(self.trainer_configs.test_loader)
            print('previous test_acc: {}'.format(test_acc))
            self.curr_metric = 100 - test_acc if self.trainer_configs.metric == 'acc' else test_loss

        self.best_state_dict = {
            'analysis': self.nnresult,
            'model_state': self.trainer_configs.model.state_dict(),
            'optimizer_state': self.trainer_configs.optimizer.state_dict(),
            'scheduler_state': self.trainer_configs.lr_scheduler.state_dict()
        }

        self.trainer_configs.model = self.trainer_configs.model.to(self.trainer_configs.device)

    def train(self):
        num_epoch = self.trainer_configs.num_epoch

        for epoch in range(num_epoch):
            cur_lr = self.trainer_configs.optimizer.param_groups[0]['lr']
            print('\nEpoch: {}/{} - LR: {}'.format(epoch, num_epoch, cur_lr))
            train_time_start = time.time()

            train_accuracy, train_loss = self.__train_single_epoch__(epoch)
            train_time_before_valid = time.time()
            print("training one epoch(s) befor valid: " + "{:.2f}".format(train_time_before_valid - train_time_start))

            benchmarking_acc, benchmarking_loss = self.eval(self.benchmarking_loader, print_acc=False)
            print('Average train loss: {}, Average train top1 acc: {}'.format(train_loss, train_accuracy))
            print(
                'Average benchmarking loss: {}, Average benchmarking top1 acc: {}'.format(
                    benchmarking_loss, benchmarking_acc
                )
            )
            train_time_end = time.time()
            print("training one epoch(s): " + "{:.2f}".format(train_time_end - train_time_start))

            self.update_saved_model(benchmarking_acc, benchmarking_loss, train_accuracy, train_loss)
            self.trainer_configs.lr_scheduler.step()

    def update_saved_model(self, new_benchmarking_accuracy, new_benchmarking_loss, new_train_accuracy, new_train_loss):
        self.nnresult.train_loss_history.append(new_train_loss)
        self.nnresult.train_accuracy_history.append(new_train_accuracy)
        self.nnresult.benchmarking_loss_history.append(new_benchmarking_loss)
        self.nnresult.benchmarking_accuracy_history.append(new_benchmarking_accuracy)

        new_metric = 100 - new_benchmarking_accuracy if self.trainer_configs.metric == 'acc' else new_benchmarking_loss
        if self.metric_func(new_metric, self.curr_metric):
            if self.use_validationset:
                test_accuracy, test_loss = self.eval(self.trainer_configs.test_loader)
                self.nnresult.validation_accuracy = new_benchmarking_accuracy
                self.nnresult.test_accuracy = test_accuracy
            else:
                self.nnresult.test_accuracy = new_benchmarking_accuracy

            print('updating model and optimizer state')
            self.best_state_dict['model_state'] = self.trainer_configs.model.state_dict()
            self.best_state_dict['optimizer_state'] = self.trainer_configs.optimizer.state_dict()
            self.best_state_dict['scheduler_state'] = self.trainer_configs.lr_scheduler.state_dict()
            self.curr_metric = new_metric
            torch.save(self.best_state_dict, self.model_dst_path)
        else:
            ckpt = torch.load(self.model_dst_path)
            ckpt['analysis'] = self.nnresult
            torch.save(ckpt, self.model_dst_path)

        dash = 100 - self.curr_metric if self.trainer_configs.metric == 'acc' else self.curr_metric
        print('curr benchmarking {}: {}'.format(self.trainer_configs.metric, dash))
        print('model saved at {}'.format(self.model_dst_path))

    def eval(self, data_loader, print_acc=True, cfm=False):
        self.trainer_configs.model.eval()
        top1 = AverageMeter('Acc@1', ':6.2f')
        losses = AverageMeter('Loss', ':.4e')
        if cfm:
            nclass = 10
            confusion_matrix = torch.zeros(nclass, nclass)
        with torch.no_grad():
            for idx, data in enumerate(data_loader):
                x, y = data
                x = x.to(self.trainer_configs.device)
                y = y.to(self.trainer_configs.device)
                n_data = y.size(0)

                acc, loss, predictions = self.__eval_single_batch_compute__(x, y)

                top1.update(acc.item(), n_data)
                losses.update(loss.item(), n_data)

                if cfm:
                    for t, p in zip(y.view(-1), predictions.view(-1)):
                        confusion_matrix[t.long(), p.long()] += 1
            if print_acc:
                print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))

        if cfm:
            plotter.plot_confusion_matrix(confusion_matrix.numpy().astype(int), normalize=True, nclass=nclass)
        return top1.avg, losses.avg

    def __train_single_epoch__(self, epoch):
        self.progress.prefix = EPOCH_FMT_STR.format(epoch)
        self.batch_time.reset()
        self.data_time.reset()
        self.data_copy_time.reset()
        self.losses.reset()
        self.top1.reset()

        self.trainer_configs.model.train()

        end = time.time()
        for i, data in enumerate(self.trainer_configs.train_loader):
            self.data_time.update(time.time() - end)
            x, y = data
            x = x.to(self.trainer_configs.device)
            y = y.to(self.trainer_configs.device)
            self.data_copy_time.update(time.time() - end)
            n_data = y.size(0)

            acc, loss = self.__train_single_epoch_compute__(x, y)

            self.losses.update(loss.item(), n_data)
            self.top1.update(acc.item(), n_data)
            self.batch_time.update(time.time() - end)
            end = time.time()

            if i % self.trainer_configs.log_update_feq == 0:
                self.progress.print2(i)

        return self.top1.avg, self.losses.avg

    def __train_single_epoch_compute__(self, x, y):
        return NotImplementedError

    def __eval_single_batch_compute__(self, x, y):
        return NotImplementedError

    def load_ckpt(self):
        ckpt = torch.load(self.trainer_configs.model_src_path)
        self.nnresult = ckpt.get('analysis')
        self.trainer_configs.model.load_state_dict(ckpt.get('model_state'))
        self.trainer_configs.optimizer.load_state_dict(ckpt.get('optimizer_state'))
        self.trainer_configs.lr_scheduler.load_state_dict(ckpt.get('scheduler_state'))


class BackPropTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(BackPropTrainer, self).__init__(*args, **kwargs)

    def __train_single_epoch_compute__(self, x, y):
        """
        computation during backprop training
        :param x: input
        :param y: target
        :return: training acc, training loss
        """
        self.trainer_configs.optimizer.zero_grad()
        output = self.trainer_configs.model(x)
        loss = self.trainer_configs.criterion(output, y)
        accs, _ = accuracy(output, y, topk=(1,))
        acc = accs[0]
        loss.backward()
        self.trainer_configs.optimizer.step()
        return acc, loss

    def __eval_single_batch_compute__(self, x, y):
        """
        computation during backprop evaluation
        :param x: input
        :param y: target
        :return: validation / test acc, valiation / test loss
        """
        output = self.trainer_configs.model(x)
        loss = self.trainer_configs.criterion(output, y)
        accs, predictions = accuracy(output, y, topk=(1,))
        acc = accs[0]
        return acc, loss, predictions


__TRAINER__ = {
    'backprop': BackPropTrainer,
}
