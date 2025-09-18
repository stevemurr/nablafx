import torch
import wandb
from .base_system import BaseSystem
from .models import GreyBoxModel
from nablafx.utils.plotting import plot_gb_model


class GreyBoxSystem(BaseSystem):
    """
    Grey Box System with truncated back-propagation through time
    at the start of each batch.
    """

    def __init__(
        self,
        model: GreyBoxModel,
        loss: torch.nn.Module,
        lr: float = 1e-4,
        log_media_every_n_steps: int = 10000,
        use_callbacks: bool = False,
    ):
        super().__init__(loss, lr, log_media_every_n_steps, use_callbacks)
        self.model = model

    def common_step(
        self,
        batch: tuple,
        batch_idx: int,
        mode: str,
    ):
        """Model step used for validation and training.
        Args:
            batch (Tuple[Tensor, Tensor, Tensor]): Batch items containing input audio (x) target audio (y), and noise profile (w).
            batch_idx (int): Index of the batch within the current epoch.
            mode (str): One of "train", "val", "test".
        """
        train = True if mode == "train" else False

        # reset hidden states for recurrent layers
        self.model.reset_states()
        if mode == "train":
            self.model.detach_states()

        # get batch
        if self.model.num_controls > 0:
            input, target, controls = batch
        else:
            input, target = batch
            controls = None

        # run the model
        pred = self(input, controls, train=train)

        # calculate loss
        tot_batch_loss = self.compute_and_log_loss(pred, target, mode)

        # calculate metrics
        self.compute_and_log_metrics(pred, target, mode)

        return tot_batch_loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx, mode="val")
        # log media
        if batch_idx == 0:
            if (self.trainer.global_step / self.log_media_every_n_steps) > self.log_media_counter:
                if self.model.num_controls > 0:
                    input, target, controls = batch
                else:
                    input, target = batch
                    controls = None

                self.model.eval()
                self.model.reset_states()
                self.model.detach_states()
                with torch.no_grad():
                    pred = self(input, controls)
                self.model.train()
                input, target, pred = input.detach().cpu(), target.detach().cpu(), pred.detach().cpu()

                self.log_audio(batch_idx, input, target, pred, "val")
                # self.log_audio_at_each_block(input, controls)
                self.log_response_and_params_at_each_block(input, controls, "val")
                self.log_media_counter += 1

                input, target, pred = None, None, None
                if controls is not None:
                    controls = None
                torch.cuda.empty_cache()
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx, mode="test")

        # log media
        if self.model.num_controls > 0:
            input, target, controls = batch
        else:
            input, target = batch
            controls = None

        self.model.reset_states()
        with torch.no_grad():
            pred = self(input, controls)
        input, target, pred = input.detach().cpu(), target.detach().cpu(), pred.detach().cpu()

        if batch_idx < 10:
            self.log_audio(batch_idx, input, target, pred, "test")
            self.log_response_and_params_at_each_block(input, controls, "test")

        input, target, pred = None, None, None
        if controls is not None:
            controls = None
        torch.cuda.empty_cache()

        return loss

    def on_train_start(self):
        # Parameter visualization handled by callbacks if use_callbacks=True
        if not self.use_callbacks:
            batch = next(iter(self.trainer.train_dataloader))
            if self.model.num_controls > 0:
                input, target, controls = batch
                controls = controls.to(self.device)
            else:
                input, target = batch
                controls = None
            input = input.to(self.device)
            target = target.to(self.device)
            self.model.reset_states()
            self.log_response_and_params_at_each_block(input, controls, "val")
            # self.log_frequency_response() # atm needs too much memory

    def log_audio_at_each_block(self, input, controls):
        print("\nLogging audio at each block...")
        x = input.to(self.device)
        y = [input]
        control_params = self.model.controller(x, controls if controls is not None else None)
        for prc, ctrl_prms in zip(self.model.processor.processors, control_params):
            if ctrl_prms is not None:
                ctrl_prms = ctrl_prms.to(self.device)
            y_i, _ = prc(x, ctrl_prms, train=False)
            y.append(y_i)
            x = y[-1]

        for i, _ in enumerate(y):  # for each block in the chain
            for j, _ in enumerate(y[i]):  # for each item in the batch
                self.logger.experiment.log(
                    {f"audio/chain/{j}/block{i}": [wandb.Audio(y[i][j].cpu().numpy()[0, :], 48000, caption=f"pred_{i}_block{j}")]},
                    step=self.trainer.global_step,
                )

    def log_response_and_params_at_each_block(self, input, controls, mode="val"):
        """Log response and parameters at each block. No-op if using callbacks.

        WARNING: This is legacy code for backward compatibility.
        For new projects, use ParameterVisualizationCallback which provides better
        configurability and more efficient parameter visualization.
        """
        if self.use_callbacks:
            return  # Parameter visualization handled by ParameterVisualizationCallback

        print("\nLogging response and parameters at each block...")
        x = input.to(self.device)
        control_params = self.model.controller(x, controls if controls is not None else None)
        param_dict_list = []
        for prc, ctrl_prms in zip(self.model.processor.processors, control_params):
            if ctrl_prms is not None:
                ctrl_prms = ctrl_prms.to(self.device)
            _, param_dict = prc(x, ctrl_prms, train=False)
            param_dict_list.append(param_dict)

        for i, _ in enumerate(input):
            plot = plot_gb_model(self.model, param_dict_list, input, i)
            self.logger.experiment.log(
                {f"response_blocks/{mode}/{i}": [wandb.Image(plot, caption=f"response_{i}")]}, step=self.trainer.global_step
            )

    def configure_optimizers(self):
        parameters = []
        # include all parameters in the model with their respective learning rates
        print("\nPer layer learning rates:")
        for module in self.model.processor.processors:
            print(type(module))
            for idx, (name, param) in enumerate(module.named_parameters()):
                # append layer parameters
                print(idx, name, self.lr * module.lr_multiplier)
                parameters += [
                    {
                        "params": [p for n, p in module.named_parameters() if n == name and p.requires_grad],
                        "lr": self.lr * module.lr_multiplier,
                    }
                ]
        print()
        for module in self.model.controller.controllers:
            print(type(module))
            for idx, (name, param) in enumerate(module.named_parameters()):
                # append layer parameters
                print(idx, name, self.lr * module.lr_multiplier)
                parameters += [
                    {
                        "params": [p for n, p in module.named_parameters() if n == name and p.requires_grad],
                        "lr": self.lr * module.lr_multiplier,
                    }
                ]

        optimizer = torch.optim.AdamW(
            parameters,
            # lr=self.lr,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=20, verbose=True)

        return [optimizer], [{"scheduler": lr_scheduler, "monitor": "loss/val/tot", "interval": "epoch", "frequency": 1}]


class GreyBoxSystemWithTBPTT(GreyBoxSystem):
    """
    Grey Box System with truncated back-propagation through time
    and gradient update every N samples.
    """

    def __init__(
        self,
        model: GreyBoxModel,
        loss: torch.nn.Module,
        lr: float = 1e-4,
        log_media_every_n_steps: int = 10000,
        step_num_samples: int = 2048,
    ):
        super().__init__(model, loss, lr, log_media_every_n_steps)
        self.step_num_samples = step_num_samples
        self.automatic_optimization = False  # disables lightning optimization
        self.apply_gradient_clipping = False
        self.gradient_clip_val = 2.0
        self.gradient_clip_algorithm = "norm"

    def common_step(
        self,
        batch: tuple,
        batch_idx: int,
        mode: str,
    ):
        train = True if mode == "train" else False

        # reset hidden states for recurrent layers
        self.model.reset_states()
        if mode == "train":
            self.model.detach_states()
            optimizer = self.optimizers()
            scheduler = self.lr_schedulers()
            optimizer.zero_grad()

        # get batch
        if self.model.num_controls > 0:
            input, target, controls = batch
        else:
            input, target = batch
            controls = None

        # run the model in steps
        seq_len = input.size(-1)
        pred_chunks = []

        for start_idx in range(0, seq_len, self.step_num_samples):
            end_idx = start_idx + self.step_num_samples
            if end_idx > seq_len:
                end_idx = seq_len

            step_input = input[:, :, start_idx:end_idx]
            step_target = target[:, :, start_idx:end_idx]
            step_pred = self(step_input, controls, train=train)

            pred_chunks.append(step_pred)

            if mode == "train":
                tot_step_loss = self.compute_and_log_loss(step_pred, step_target, mode, should_log=False)
                self.manual_backward(tot_step_loss)
                # need to apply gradient clipping manually
                if self.apply_gradient_clipping:
                    self.clip_gradients(optimizer, self.gradient_clip_val, self.gradient_clip_algorithm)
                optimizer.step()
                self.model.detach_states()
                optimizer.zero_grad()

        pred = torch.cat(pred_chunks, dim=-1)
        assert input.shape == target.shape == pred.shape

        # calculate loss
        tot_batch_loss = self.compute_and_log_loss(pred, target, mode, should_log=True)

        # calculate metrics
        self.compute_and_log_metrics(pred, target, mode)

        return tot_batch_loss

    def on_validation_epoch_end(self):
        self.lr_schedulers().step(self.trainer.logged_metrics["loss/val/tot"])
