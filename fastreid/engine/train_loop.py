# encoding: utf-8
"""
credit:
https://github.com/facebookresearch/detectron2/blob/master/detectron2/engine/train_loop.py
"""

import logging
import time
import weakref

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel

import fastreid.utils.comm as comm
from fastreid.utils.events import EventStorage
torch.autograd.set_detect_anomaly(True)
__all__ = ["HookBase", "TrainerBase", "SimpleTrainer"]

from .ganbuild import build_lr_scheduler,build_optimizer_gan
from .ganbuild import discritor_cam
from .ganbuild import discritor_code
from torch import nn
class HookBase:
    """
    Base class for hooks that can be registered with :class:`TrainerBase`.
    Each hook can implement 4 methods. The way they are called is demonstrated
    in the following snippet:
    .. code-block:: python
        hook.before_train()
        for iter in range(start_iter, max_iter):
            hook.before_step()
            trainer.run_step()
            hook.after_step()
        hook.after_train()
    Notes:
        1. In the hook method, users can access `self.trainer` to access more
           properties about the context (e.g., current iteration).
        2. A hook that does something in :meth:`before_step` can often be
           implemented equivalently in :meth:`after_step`.
           If the hook takes non-trivial time, it is strongly recommended to
           implement the hook in :meth:`after_step` instead of :meth:`before_step`.
           The convention is that :meth:`before_step` should only take negligible time.
           Following this convention will allow hooks that do care about the difference
           between :meth:`before_step` and :meth:`after_step` (e.g., timer) to
           function properly.
    Attributes:
        trainer: A weak reference to the trainer object. Set by the trainer when the hook is
            registered.
    """

    def before_train(self):
        """
        Called before the first iteration.
        """
        pass

    def after_train(self):
        """
        Called after the last iteration.
        """
        pass

    def before_step(self):
        """
        Called before each iteration.
        """
        pass

    def after_step(self):
        """
        Called after each iteration.
        """
        pass


class TrainerBase:
    """
    Base class for iterative trainer with hooks.
    The only assumption we made here is: the training runs in a loop.
    A subclass can implement what the loop is.
    We made no assumptions about the existence of dataloader, optimizer, model, etc.
    Attributes:
        iter(int): the current iteration.
        start_iter(int): The iteration to start with.
            By convention the minimum possible value is 0.
        max_iter(int): The iteration to end training.
        storage(EventStorage): An EventStorage that's opened during the course of training.
    """

    def __init__(self):
        self._hooks = []
        self._hooks_new = []
        

    def register_hooks(self, hooks,hooks_new):
        """
        Register hooks to the trainer. The hooks are executed in the order
        they are registered.
        Args:
            hooks (list[Optional[HookBase]]): list of hooks
        """
        hooks = [h for h in hooks if h is not None]
        for h in hooks:
            assert isinstance(h, HookBase)
            # To avoid circular reference, hooks and trainer cannot own each other.
            # This normally does not matter, but will cause memory leak if the
            # involved objects contain __del__:
            # See http://engineering.hearsaysocial.com/2013/06/16/circular-references-in-python/
            h.trainer = weakref.proxy(self)
        self._hooks.extend(hooks)

        hooks_new = [h for h in hooks_new if h is not None]
        for h in hooks_new:
            assert isinstance(h, HookBase)
            # To avoid circular reference, hooks and trainer cannot own each other.
            # This normally does not matter, but will cause memory leak if the
            # involved objects contain __del__:
            # See http://engineering.hearsaysocial.com/2013/06/16/circular-references-in-python/
            h.trainer = weakref.proxy(self)
        self._hooks_new.extend(hooks_new)


    def train(self, start_iter: int, max_iter: int):
        """
        Args:
            start_iter, max_iter (int): See docs above
        """
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()
                for self.iter in range(start_iter, max_iter):
                    self.before_step()
                    self.run_step()
                    self.after_step()
            except Exception:
                logger.exception("Exception during training:")
            finally:
                self.after_train()

        with EventStorage(start_iter) as self.storage_new:
            try:
                self.before_train_new()
                for self.iter in range(start_iter, max_iter):
                    self.before_step_new()
                    self.run_step_new()
                    self.after_step_new()
            except Exception:
                logger.exception("Exception during training:")
            finally:
                self.after_train_new()

    def before_train(self):
        for h in self._hooks:
            h.before_train()
        for h in self._hooks_new:
            h.before_train()
    def after_train(self):
        for h in self._hooks:
            h.after_train()
        for h in self._hooks_new:
            h.after_train()
    def before_step(self):
        for h in self._hooks:
            h.before_step()
        for h in self._hooks_new:
            h.before_step()
    def after_step(self):
        for h in self._hooks:
            h.after_step()
        for h in self._hooks_new:
            h.after_step()
        # this guarantees, that in each hook's after_step, storage.iter == trainer.iter
        self.storage.step()

    def before_train_new(self):
        for h in self._hooks_new:
            h.before_train()
    def after_train_new(self):
        for h in self._hooks_new:
            h.after_train()
    def before_step_new(self):
        for h in self._hooks_new:
            h.before_step()
    def after_step_new(self):
        for h in self._hooks_new:
            h.after_step()
        # this guarantees, that in each hook's after_step, storage.iter == trainer.iter
        self.storage_new.step()
    def run_step(self):
        raise NotImplementedError
    def run_step_new(self):
        raise NotImplementedError
import cv2
import scipy.io as io
from torch.nn import functional as F
class SimpleTrainer(TrainerBase):
    """
    A simple trainer for the most common type of task:
    single-cost single-optimizer single-data-source iterative optimization.
    It assumes that every step, you:
    1. Compute the loss with a data from the data_loader.
    2. Compute the gradients with the above loss.
    3. Update the model with the optimizer.
    If you want to do anything fancier than this,
    either subclass TrainerBase and implement your own `run_step`,
    or write your own training loop.
    """

    def __init__(self, model,model_teacher, data_loader, optimizer,optimizer_teacher):
        """
        Args:
            model: a torch Module. Takes a data from data_loader and returns a
                dict of heads.
            data_loader: an iterable. Contains data to be used to call model.
            optimizer: a torch optimizer.
        """
        super().__init__()

        """
        We set the model to training mode in the trainer.
        However it's valid to train a model that's in eval mode.
        If you want your model (or a submodule of it) to behave
        like evaluation during training, you can overwrite its train() method.
        """
        model.train()
        model_teacher.train()
        self.model = model
        self.model_teacher=model_teacher
        self.data_loader = data_loader
        self._data_loader_iter = iter(data_loader)
        self.optimizer = optimizer
        self.optimizer_teacher = optimizer_teacher
        self.flag_teacher=True
        self.iternew=0
    def maskgenerate(self,outputs,width=384,height=128):
        masks=[]
        m=torch.nn.Tanh()
        
        
        for j in range(outputs.size(0)):
            am = outputs[j, ...].detach().cpu().numpy()
            am = cv2.resize(am, (width, height))
            am = 255 * (am - np.min(am)) / (
                        np.max(am) - np.min(am) + 1e-12
            )
            am = np.uint8(np.floor(am))
            am_ten=torch.from_numpy(am)
            am_ten=torch.reshape(am_ten,(8,width,height))
            am_ten=torch.sum(am_ten,dim=0)
            am_ten=am_ten.float()
            am_ten=F.softmax(am_ten,dim=1)#am_ten/torch.mean(am_ten) 
            zero = torch.zeros_like(am_ten)
            #ones=torch.ones_like(am_ten)
            am_ten=torch.where(am_ten < 0.05, zero, am_ten)
            #result1 = np.array(am_ten)
            #np.set_printoptions(precision = 3)
            #np.savetxt('npresult1.txt',result1)
            am_ten=am_ten.unsqueeze(0)
            masks.append(am_ten)
        masks=torch.cat(masks,dim=0)
        return masks
    def masknewgenerate(self,outputs,width=384,height=128):
        masks=[]
        m=nn.Tanh()
        
        
        for j in range(outputs.size(0)):
            am = outputs[j, ...].detach().cpu().numpy()
            am = cv2.resize(am, (width, height))
            am = 255 * (am - np.min(am)) / (
                        np.max(am) - np.min(am) + 1e-12
            )
            am = np.uint8(np.floor(am))
            am_ten=torch.from_numpy(am)
            am_ten=torch.reshape(am_ten,(8,width,height))
            am_ten=torch.sum(am_ten,dim=0)
            am_ten=am_ten.float()



            am_ten_sig=am_ten/(torch.mean(am_ten)*3) 
            am_ten_sig=m(am_ten_sig)
            zero = torch.zeros_like(am_ten_sig)
            am_ten_sig=torch.where(am_ten_sig < 0.5, zero, am_ten_sig)


            #result1 = np.array(am_ten_sig)
            #np.set_printoptions(precision = 3)
            #np.savetxt('npresult1.txt',result1)

            am_ten_soft=F.softmax(am_ten,dim=1)
            am_ten_soft=torch.where(am_ten_soft < 0.07, zero, am_ten_soft)
            
            am_ten_final=am_ten_sig+am_ten_soft*2

            #result1 = np.array(am_ten)
            #np.set_printoptions(precision = 3)
            #np.savetxt('npresult1.txt',result1)
            am_ten_final=am_ten_final.unsqueeze(0)
            masks.append(am_ten_final)
        masks=torch.cat(masks,dim=0)
        return masks
    def run_step(self):
        """
        Implement the standard training logic described above.
        """

        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        #assert self.model_teacher.training,"[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If your want to do something with the data, you can wrap the dataloader.
        """
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        """
        If your want to do something with the heads, you can wrap the model.
        """
        outputs_teacher, targets_teacher = self.model_teacher(data)
        _,_, _, _,_,_,_,_,_,_,featuremap_t=outputs_teacher
        
        masks=self.masknewgenerate(featuremap_t.detach())
        outputs, targets = self.model(data,masks=masks)

        # Compute loss
        if isinstance(self.model, DistributedDataParallel):
            loss_dict = self.model.module.losses(outputs, targets,outputs_teacher,targets_teacher.detach())
        else:
            loss_dict = self.model.losses(outputs, targets.detach(),outputs_teacher,targets_teacher.detach())
        losses = sum(loss_dict.values())

        if isinstance(self.model_teacher, DistributedDataParallel):
            loss_dict_teacher = self.model_teacher.module.losses(outputs, targets.detach(),outputs_teacher,targets_teacher)
        else:
            loss_dict_teacher = self.model_teacher.losses(outputs, targets.detach(),outputs_teacher,targets_teacher)
            
        losses_teacher = sum(loss_dict_teacher.values())



        self.optimizer.zero_grad()
        losses.backward(retain_graph=True)
        with torch.cuda.stream(torch.cuda.Stream()):
            metrics_dictnew = loss_dict_teacher
            metrics_dictnew["data_time"] = data_time
            metrics_dict = loss_dict
            metrics_dict["data_time"] = data_time
            self._write_metrics(metrics_dict,metrics_dictnew)
            self._detect_anomaly(losses, loss_dict)
            
            
        """
        if self.flag_teacher:
            with torch.cuda.stream(torch.cuda.Stream()):
                metrics_dictnew = loss_dict_teacher
                metrics_dictnew["data_time"] = data_time
                self._write_metrics(metrics_dictnew)
                self._detect_anomaly(losses_teacher, loss_dict_teacher)
        """ 
            
        if self.iternew==199:
            print(losses_teacher)
            print(self.flag_teacher)
            self.iternew=0
        

        if self.iternew>41999:
            self.flag_teacher=False
            self.optimizer.param_groups[0]['lr']=0.000035
            self.iternew=-1
        self.optimizer.step()
        

        if self.flag_teacher:
            self.optimizer_teacher.zero_grad()
            losses_teacher.backward()
            self.optimizer_teacher.step()  
            self.iternew+=1
        
        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method.
        """
        #if self.iter<27199:
        #    self.optimizer_teacher.zero_grad()
        #    losses_teacher.backward(retain_graph=True)
        #    self.optimizer_teacher.step()
        #else:
        #self.flag_teacher=False
        #self.model_teacher.eval()
        #self.optimizer.param_groups[0]['lr']=0.000035
            #self.optimizer.setlr(0.00035)


        #self.optimizer.defaults['lr']==0.00235
        #self.optimizer.param_groups[0]['lr']=0.00035
        #print(self.optimizer.defaults['lr'])
    def _detect_anomaly(self, losses, loss_dict):
        if not torch.isfinite(losses).all():
            raise FloatingPointError(
                "Loss became infinite or NaN at iteration={}!\nloss_dict = {}".format(
                    self.iter, loss_dict
                )
            )

    def _write_metrics(self, metrics_dict: dict,metrics_dictnew:dict):
        """
        Args:
            metrics_dict (dict): dict of scalar metrics
        """
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        """
        metrics_dictnew = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dictnew.items()
        }
        """
        # gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in fastreid.
        all_metrics_dict = comm.gather(metrics_dict)
        #all_metrics_dict_new = comm.gather(metrics_dictnew)

        if comm.is_main_process():
            if "data_time" in all_metrics_dict[0]:
                # data_time among workers can have high variance. The actual latency
                # caused by data_time is the maximum among workers.
                data_time = np.max([x.pop("data_time") for x in all_metrics_dict])


                self.storage.put_scalar("data_time", data_time)

            # average the rest metrics
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict]) for k in all_metrics_dict[0].keys()
            }
            total_losses_reduced = sum(loss for loss in metrics_dict.values())

            """
            metrics_dictnew = {
                k: np.mean([x[k] for x in all_metrics_dict_new]) for k in all_metrics_dict_new[0].keys()
            }
            total_losses_reduced_new = sum(loss for loss in metrics_dictnew.values())
            """

            self.storage.put_scalar("total_loss", total_losses_reduced)
            if len(metrics_dict) > 1:
                self.storage.put_scalars(**metrics_dict)

            """
            self.storage.put_scalar("total_loss_tea", total_losses_reduced_new)
            if len(metrics_dictnew) > 1:
                self.storage.put_scalars(**metrics_dictnew)
            """