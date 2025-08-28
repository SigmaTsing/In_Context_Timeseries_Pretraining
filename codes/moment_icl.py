import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from samay.dataset import BaseDataset
from samay.model import MomentModel
import os
from tqdm import tqdm

class MomentMixedDataset(BaseDataset):
    """
    Dataset class for Moment model
    Data Format:
    Dict with keys:
    input_ts: np.ndarray, historical time series data
    actual_ts: np.ndarray, actual time series data
    """

    def __init__(
        self,
        name=None,
        datetime_col=None,
        path=None,
        batchsize=8,
        mode="train",
        boundaries=[0, 0, 0],
        horizon=0,
        task_name="forecasting",
        label_col=None,
        stride=10,
        num_icl_examples = 8,
        icl_task_types = [0, 1],
        seq_len = 96,
        **kwargs,
    ):
        super().__init__(
            name=name,
            datetime_col=datetime_col,
            path=path,
            batchsize=batchsize,
            mode=mode,
        )
        self.task_name = task_name
        self.label_col = "label" if label_col is None else label_col

        self.seq_len = seq_len
        self.stride = stride
        self.forecast_horizon = horizon
        self.boundaries = boundaries

        self.icl_task_types = icl_task_types
        self.num_icl_examples = num_icl_examples
        self._read_data()
        self.required_len = self.seq_len + self.forecast_horizon
        self.pad = False
        self.pad_len = 0
        if self.length_timeseries < self.required_len:
            self.pad = True
            

    def _read_data(self):
        self.scaler = StandardScaler()
        self.df = pd.read_csv(self.data_path)

        if self.boundaries[0] == 0:
            self.boundaries[0] = int(len(self.df) * 0.6)
        if self.boundaries[1] == 0:
            self.boundaries[1] = int(len(self.df) * 0.8)
        if self.boundaries[2] == 0:
            self.boundaries[2] = int(len(self.df) - 1)

        self.n_channels = self.df.shape[1] - 1

        if self.datetime_col:
            self.df.drop(columns=[self.datetime_col], inplace=True)

        self.df = self.df.infer_objects(copy=False).interpolate(method="cubic")

        self.scaler.fit(self.df[slice(0, self.boundaries[0])].values)
        self.df = self.scaler.transform(self.df.values)


        if self.mode == "train":
            self.data = self.df[slice(0, self.boundaries[0]), :]
        elif self.mode == "test":
            self.data = self.df[slice(self.boundaries[1], self.boundaries[2]), :]

        self.length_timeseries = self.data.shape[0]
        print(self.data.shape)
        # self.icl_label = np.random.randint(0, 2, size = self.__len__())
        self.icl_label = np.array([self.icl_task_types[i] for i in np.random.randint(0, len(self.icl_task_types), size = self.__len__())])

    def pad_sequence(self):
        self.pad_len = self.required_len - self.length_timeseries
        # Pad data with zeros from the left
        self.data = np.pad(self.data, ((self.pad_len, 0), (0, 0)))
        self.length_timeseries = self.data.shape[0]

    def get_single(self, index):
        if self.pad:
            self.pad_sequence()

        seq_start = self.stride * index
        seq_end = seq_start + self.seq_len
        input_mask = np.ones(self.seq_len)
        # if the sequence is padded, mask of padded part is 0
        input_mask[: self.pad_len] = 0

        pred_end = seq_end + self.forecast_horizon

        icl_class = self.icl_label[index]
        if pred_end > self.length_timeseries:
            pred_end = self.length_timeseries
            seq_start = seq_end - self.seq_len
            seq_end = pred_end - self.forecast_horizon
            
        if icl_class == 0: # Future
            input_seq = self.data[seq_start:seq_end, :].T
            forecast_seq = self.data[seq_end:pred_end, :].T
        elif icl_class == 1: # Past
            seq_end = seq_end - self.seq_len + self.forecast_horizon
            input_seq = self.data[seq_end:pred_end, :].T
            forecast_seq = self.data[seq_start:seq_end, :].T
        elif icl_class == 2: #Imputation
            # pad_len = int((self.seq_len / 8)*0.3)*8
            pad_len = self.forecast_horizon
            pad_start = np.random.randint(1, int(self.seq_len / 8) - int(pad_len / 8))*8
            forecast_seq = self.data[seq_start + pad_start : seq_start + pad_start + pad_len, :].T
            input_seq = self.data[seq_start:seq_end, :].T
            input_mask[pad_start:pad_start + pad_len] = 0
            # input_seq[:, seq_start + pad_start : seq_start + pad_start + pad_len] = 0
            
        return input_seq, input_mask, forecast_seq

    def __getitem__(self, index):
        input_seq, input_mask, forecast_seq = self.get_single(index)
        if self.num_icl_examples == 0:
            return input_seq, input_mask, forecast_seq
        # Get ICL examples. TODO: avoid index of target
        # print(self.icl_label == self.icl_label[index])
        candidates = np.random.choice(np.argwhere(self.icl_label == self.icl_label[index]).squeeze(), self.num_icl_examples)
        examples = [self.get_single(i) for i in candidates]
        # print([i.shape for i in examples[0]])
        masks = np.concatenate([np.concatenate((res[1], np.ones(res[2].shape[1])), axis = 0) for res in examples], axis = 0)
        examples = np.concatenate([np.concatenate((r1, r3), axis=1) for r1, r3 in [(res[0], res[2]) for res in examples]], axis = 1)
        # print(masks.shape, examples.shape)
        return np.concatenate((examples, input_seq), axis = 1), np.concatenate((masks, input_mask), axis = 0), forecast_seq

    def __len__(self):
        if self.length_timeseries < self.seq_len + self.forecast_horizon:
            return 1
        return (
            self.length_timeseries - self.seq_len - self.forecast_horizon
        ) // self.stride + 1

    def get_data_loader(self):
        if self.mode == "train":
            return DataLoader(self, batch_size=self.batchsize, shuffle=True)
        else:
            return DataLoader(self, batch_size=self.batchsize, shuffle=False)

    def _transform_labels(self, labels: np.ndarray):
        unq_labels = np.unique(labels)  # Move the labels to {0, ..., L-1}
        transform = {}
        for i, l in enumerate(unq_labels):
            transform[l] = i

        labels = np.vectorize(transform.get)(labels)

        return labels
    
def finetune_moment(moment, dataset, task_name="forecasting", **kwargs):
        # arguments
        max_lr = 1e-4 if "lr" not in kwargs else kwargs["lr"]
        max_epoch = 2 if "epoch" not in kwargs else kwargs["epoch"]
        max_norm = 5.0 if "norm" not in kwargs else kwargs["norm"]
        mask_ratio = 0.25 if "mask_ratio" not in kwargs else kwargs["mask_ratio"]

        dataloader = dataset.get_data_loader()
        criterion = torch.nn.MSELoss()
        if task_name == "classification":
            criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(moment.model.parameters(), lr=max_lr)
        criterion.to(moment.device)
        scaler = torch.amp.GradScaler()

        total_steps = len(dataloader) * max_epoch
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=max_lr, total_steps=total_steps, pct_start=0.3
        )
        moment.model.to(moment.device)
        moment.model.train()

        for epoch in range(max_epoch):
            losses = []
            for i, data in tqdm(enumerate(dataloader), total = int(len(dataset) / dataset.batchsize)):
            # for i, data in enumerate(dataloader):
                # unpack the data
                if task_name == "forecasting":
                    timeseries, input_mask, forecast = data
                    # Move the data to the GPU
                    timeseries = timeseries.float().to(moment.device)
                    input_mask = input_mask.to(moment.device)
                    forecast = forecast.float().to(moment.device)
                    with torch.amp.autocast(device_type="cuda"):
                        output = moment.model(x_enc=timeseries, input_mask=input_mask)
                    loss = criterion(output.forecast, forecast)

                optimizer.zero_grad(set_to_none=True)
                # Scales the loss for mixed precision training
                scaler.scale(loss).backward()

                # Clip gradients
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(moment.model.parameters(), max_norm)

                scaler.step(optimizer)
                scaler.update()

                losses.append(loss.item())

            losses = np.array(losses)
            average_loss = np.average(losses)
            print(f"Epoch {epoch}: Train loss: {average_loss:.3f}")

            scheduler.step()

        return moment.model
    
if __name__ == '__main__':
    
    seq_len = 192   
    horizon = 96
    
    train_batch_size = 12
    test_batch_size = 12
    num_icl_examples = 4
    
    # data_path = "/nethome/sli999/TSFMProject/src/tsfmproject/models/moment/data/ETTh1.csv"
    
    # train_data_path = '/nethome/sxu452/Time-LLM/dataset/weather/weather.csv'
    train_data_path = "/nethome/sxu452/Time-LLM/dataset/ETT-small/ETTh1.csv"
    # train_data_path = '/nethome/sxu452/Time-LLM/dataset/exchange_rate/exchange_rate.csv'
    # train_data_path = 
    
    # test_data_path = '/nethome/sxu452/Time-LLM/dataset/electricity/electricity.csv'
    test_data_path = train_data_path
    
    train_dataset = MomentMixedDataset(
        name="ett",
        datetime_col="date",
        path= train_data_path,
        mode="train",
        horizon=horizon,
        seq_len=seq_len,
        batchsize=train_batch_size,
        num_icl_examples=num_icl_examples,
        icl_task_types=[0,1]
    )
    # val_forecast = MomentMixedDataset(
    #     name="ett",
    #     datetime_col="date",
    #     path= test_data_path,
    #     mode="test",
    #     horizon=horizon,
    #     seq_len=seq_len,
    #     batchsize=test_batch_size,
    #     num_icl_examples=num_icl_examples,
    #     icl_task_types=[0]
    # )
    # val_history = MomentMixedDataset(
    #     name="ett",
    #     datetime_col="date",
    #     path = test_data_path,
    #     mode="test",
    #     horizon=horizon,
    #     seq_len=seq_len,
    #     batchsize=test_batch_size,
    #     num_icl_examples=num_icl_examples,
    #     icl_task_types=[1]
    # )
    val_imputation = MomentMixedDataset(
        name="ett",
        datetime_col="date",
        path = test_data_path,
        mode="test",
        horizon=horizon,
        seq_len=seq_len,
        batchsize=test_batch_size,
        num_icl_examples=0,
        icl_task_types=[2]
    )
    
    repo = "AutonLab/MOMENT-1-large"
    config = {
        "task_name": "forecasting",
        "forecast_horizon": train_dataset[0][2].shape[-1],
        'seq_len': train_dataset[0][0].shape[-1],
        "head_dropout": 0.1,
        "weight_decay": 0,
        "freeze_encoder": False,  # Freeze the patch embedding layer
        "freeze_embedder": False,  # Freeze the transformer encoder
        "freeze_head": False,  # The linear forecasting head must be trained
    }
    mmt = MomentModel(config=config, repo=repo)

    for run in tqdm(range(200)):
        # imp_loss, _, _, _ = mmt.evaluate(val_imputation)
        # _, trues, preds, _ = mmt.evaluate(val_forecast)
        # forecast_mse = np.mean((trues - preds) ** 2)
        # forecast_mae = np.mean(np.abs(trues - preds))
        # _, trues, preds, _ = mmt.evaluate(val_history)
        # history_mse = np.mean((trues - preds) ** 2)
        # history_mae = np.mean(np.abs(trues - preds))
        # print(f'Future: MSE {forecast_mse:.3f} MAE {forecast_mae:.3f} Past: MSE {history_mse:.3f} MAE {history_mae:.3f}')
        imp_loss, _, _, _ = mmt.evaluate(val_imputation)
        print(f'Imputation: {imp_loss:.5f}')
        finetune_moment(mmt, train_dataset, epoch = 1)