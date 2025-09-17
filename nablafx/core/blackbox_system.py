import torch
from .base_system import BaseSystem
from .models import BlackBoxModel


class BlackBoxSystem(BaseSystem):
    """
    Black Box System with truncated back-propagation through time
    at the start of each batch.
    """

    def __init__(
        self,
        model: BlackBoxModel,
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
            batch (Tuple[Tensor, Tensor, Tensor]): Batch items containing input audio (x) target audio (y).
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
        # log media (skip if using callbacks)
        if not self.use_callbacks and batch_idx == 0:
            if (self.trainer.global_step / self.log_media_every_n_steps) > self.log_media_counter:
                if self.model.num_controls > 0:
                    input, target, controls = batch
                else:
                    input, target = batch
                    controls = None

                self.model.reset_states()
                self.model.detach_states()
                pred = self(input, controls)

                input, target, pred = input.detach().cpu(), target.detach().cpu(), pred.detach().cpu()

                self.log_audio(batch_idx, input, target, pred, "val")
                self.log_media_counter += 1

                input, target, pred = None, None, None
                if controls is not None:
                    controls = None
                # torch.cuda.empty_cache()
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx, mode="test")

        # log media (skip if using callbacks)
        if not self.use_callbacks:
            if self.model.num_controls > 0:
                input, target, controls = batch
            else:
                input, target = batch
                controls = None

            self.model.reset_states()
            with torch.no_grad():
                pred = self(input, controls)
            input, target, pred = input.detach().cpu(), target.detach().cpu(), pred.detach().cpu()

            self.log_audio(batch_idx, input, target, pred, "test")

            input, target, pred = None, None, None
            if controls is not None:
                controls = None
            torch.cuda.empty_cache()

        return loss


class BlackBoxSystemWithTBPTT(BlackBoxSystem):
    """
    Black Box System with truncated back-propagation through time
    and gradient update every N samples.
    """

    def __init__(
        self,
        model: BlackBoxModel,
        loss: torch.nn.Module,
        lr: float = 1e-4,
        log_media_every_n_steps: int = 10000,
        step_num_samples: int = 2048,
        use_callbacks: bool = False,
    ):
        super().__init__(model, loss, lr, log_media_every_n_steps, use_callbacks)
        self.step_num_samples = step_num_samples
        self.automatic_optimization = False  # disables lightning optimization
        self.apply_gradient_clipping = False
        self.gradient_clip_val = 1.0
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

                # run the model in steps
                seq_len = input.size(-1)
                pred_chunks = []

                for start_idx in range(0, seq_len, self.step_num_samples):
                    end_idx = start_idx + self.step_num_samples
                    if end_idx > seq_len:
                        end_idx = seq_len

                    step_input = input[:, :, start_idx:end_idx]
                    step_target = target[:, :, start_idx:end_idx]
                    step_pred = self(step_input, controls)

                    pred_chunks.append(step_pred)

                pred = torch.cat(pred_chunks, dim=-1)
                assert input.shape == target.shape == pred.shape
                input, target, pred = input.detach().cpu(), target.detach().cpu(), pred.detach().cpu()

                self.log_audio(batch_idx, input, target, pred, "val")
                self.log_media_counter += 1

                input, target, pred = None, None, None
                if controls is not None:
                    controls = None
        return loss

    def on_validation_epoch_end(self):
        self.lr_schedulers().step(self.trainer.logged_metrics["loss/val/tot"])
