import argparse
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
import utils.metrics
import utils.losses
import numpy as np


class SupervisedForecastTask(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        regressor="linear",
        loss="mse",
        pre_len: int = 3,
        learning_rate: float = 1e-3,
        weight_decay: float = 1.5e-3,
        feat_max_val: float = 1.0,
        **kwargs
    ):
        super(SupervisedForecastTask, self).__init__()
        self.save_hyperparameters()
        self.model = model
        self.regressor = (
            nn.Linear(
                self.model.hyperparameters.get("hidden_dim")
                or self.model.hyperparameters.get("output_dim"),
                self.hparams.pre_len,
            )
            if regressor == "linear"
            else regressor
        )
        self._loss = loss
        self.feat_max_val = feat_max_val

    def forward(self, x):
        # (batch_size, seq_len, num_nodes)
        batch_size, _, num_nodes = x.size()
        # (batch_size, num_nodes, hidden_dim)
        hidden = self.model(x)
        # (batch_size * num_nodes, hidden_dim)
        hidden = hidden.reshape((-1, hidden.size(2)))
        # (batch_size * num_nodes, pre_len)
        if self.regressor is not None:
            predictions = self.regressor(hidden)
        else:
            predictions = hidden
        predictions = predictions.reshape((batch_size, num_nodes, -1))
        return predictions

    def shared_step(self, batch, batch_idx):
        # (batch_size, seq_len/pre_len, num_nodes)
        x, y = batch
        num_nodes = x.size(2)
        predictions = self(x)
        predictions = predictions.transpose(1, 2).reshape((-1, num_nodes))
        y = y.reshape((-1, y.size(2)))
        return predictions, y

    def loss(self, inputs, targets):
        if self._loss == "mse":
            return F.mse_loss(inputs, targets)
        if self._loss == "mse_with_regularizer":
            return utils.losses.mse_with_regularizer_loss(inputs, targets, self)
        raise NameError("Loss not supported:", self._loss)

    def training_step(self, batch, batch_idx):
        predictions, y = self.shared_step(batch, batch_idx)
        loss = self.loss(predictions, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        predictions, y = self.shared_step(batch, batch_idx)
        predictions = predictions * self.feat_max_val

        # np.savez("prediction.npz",predictions2.cpu(),allow_pickle=True)
        # np.savez("true.npz",y2.cpu(),allow_pickle=True)

        predictions_shape=predictions.shape
        y = y * self.feat_max_val
        y_shape=y.shape

        predictions1 = predictions[0:3374, :]
        y1 = y[0:3374, :]
        predictions2 = predictions[3374:6749, :]
        y2 = y[3374:6749, :]
        predictions3 = predictions[6749:10124, :]
        y3= y[6749:10124, :]
        predictions4 = predictions[10124:13499, :]
        y4= y[10124:13499, :]
        predictions5 = predictions[13499:16874, :]
        y5 = y[13499:16874, :]
        predictions6 = predictions[16874:20249, :]
        y6 = y[16874:20249, :]
        predictions7 = predictions[20249:23624, :]
        y7= y[20249:23624, :]
        predictions8 = predictions[23624:26999, :]
        y8 = y[23624:26999, :]
        predictions9 = predictions[26999:30374, :]
        y9 = y[26999:30374, :]
        predictions10 = predictions[30374:33749, :]
        y10 = y[30374:33749, :]

        predictions11 = predictions[33794:37124, :]
        y11 = y[33794:37124, :]

        predictions12=predictions[37124:40499,:]
        y12=y[37124:40499,:]
        mae1 = torchmetrics.functional.mean_absolute_error(predictions1, y1)
        rmse1 = torch.sqrt(torchmetrics.functional.mean_squared_error(predictions1, y1))

        mae2 = torchmetrics.functional.mean_absolute_error(predictions2, y2)
        rmse2 = torch.sqrt(torchmetrics.functional.mean_squared_error(predictions2, y2))

        mae3 = torchmetrics.functional.mean_absolute_error(predictions3, y3)
        rmse3 = torch.sqrt(torchmetrics.functional.mean_squared_error(predictions3, y3))

        mae4= torchmetrics.functional.mean_absolute_error(predictions4, y4)
        rmse4 = torch.sqrt(torchmetrics.functional.mean_squared_error(predictions4, y4))

        mae5 = torchmetrics.functional.mean_absolute_error(predictions5, y5)
        rmse5 = torch.sqrt(torchmetrics.functional.mean_squared_error(predictions5, y5))

        mae6 = torchmetrics.functional.mean_absolute_error(predictions6, y6)
        rmse6 = torch.sqrt(torchmetrics.functional.mean_squared_error(predictions6, y6))

        mae7 = torchmetrics.functional.mean_absolute_error(predictions7, y7)
        rmse7 = torch.sqrt(torchmetrics.functional.mean_squared_error(predictions7, y7))

        mae8 = torchmetrics.functional.mean_absolute_error(predictions8, y8)
        rmse8 = torch.sqrt(torchmetrics.functional.mean_squared_error(predictions8, y8))

        mae9 = torchmetrics.functional.mean_absolute_error(predictions9, y9)
        rmse9 = torch.sqrt(torchmetrics.functional.mean_squared_error(predictions9, y9))

        mae10 = torchmetrics.functional.mean_absolute_error(predictions10, y10)
        rmse10 = torch.sqrt(torchmetrics.functional.mean_squared_error(predictions10, y10))

        mae11 = torchmetrics.functional.mean_absolute_error(predictions11, y11)
        rmse11 = torch.sqrt(torchmetrics.functional.mean_squared_error(predictions11, y11))

        mae12 = torchmetrics.functional.mean_absolute_error(predictions12, y12)
        rmse12 = torch.sqrt(torchmetrics.functional.mean_squared_error(predictions12, y12))

        average_mae=(mae1+mae2+mae3+mae4+mae5+mae6+mae7+mae8+mae9+mae10+mae11+mae12)/12
        average_rmse=(rmse1+rmse2+rmse3+rmse4+rmse5+rmse6+rmse7+rmse8+rmse9+rmse10+rmse11+rmse12)/12

        loss = self.loss(predictions, y)
        rmse = torch.sqrt(torchmetrics.functional.mean_squared_error(predictions, y))
        mae = torchmetrics.functional.mean_absolute_error(predictions, y)
        mape=torchmetrics.functional.mean_absolute_percentage_error(predictions, y)
        accuracy = utils.metrics.accuracy(predictions, y)
        r2 = utils.metrics.r2(predictions, y)
        explained_variance = utils.metrics.explained_variance(predictions, y)
        metrics = {
            "predictions shape1": predictions_shape[0],
            "predictions shape2": predictions_shape[1],
            "y shape1": y_shape[0],
            "y shape2": y_shape[1],
            "mae1":mae1,
            "rmse1":rmse1,

            "mae2": mae2,
            "rmse2": rmse2,

            "mae3": mae3,
            "rmse3": rmse3,

            "mae4": mae4,
            "rmse4": rmse4,

            "mae5": mae5,
            "rmse5": rmse5,

            "mae6": mae6,
            "rmse6": rmse6,

            "mae7": mae7,
            "rmse7": rmse7,

            "mae8": mae8,
            "rmse8": rmse8,

            "mae9": mae9,
            "rmse9": rmse9,

            "mae10": mae10,
            "rmse10": rmse10,

            "mae11": mae11,
            "rmse11": rmse11,

            "mae12": mae12,
            "rmse12": rmse12,

            "average_mae":average_mae,
            "average_rmse":average_rmse,

            "RMSE": rmse,
            "MAE": mae,
            "MAPE":mape,
            "accuracy": accuracy,
            "R2": r2,
            "ExplainedVar": explained_variance,

        }
        self.log_dict(metrics)
        return predictions.reshape(batch[1].size()), y.reshape(batch[1].size())

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

    @staticmethod
    def add_task_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--learning_rate", "--lr", type=float, default=1e-3)
        parser.add_argument("--weight_decay", "--wd", type=float, default=1.5e-3)
        parser.add_argument("--loss", type=str, default="mse")
        return parser
