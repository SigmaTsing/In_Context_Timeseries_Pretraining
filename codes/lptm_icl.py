import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from samay.dataset import BaseDataset
from samay.model import LPTMModel
import os
from tqdm import tqdm

class LPTMMixedCIDataset(BaseDataset):
    """
    Dataset class for lptm model
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
        self.per_channel_length = ((
            self.length_timeseries - self.seq_len - self.forecast_horizon
        ) // self.stride + 1)
        print(self.data.shape)
        # self.icl_label = np.random.randint(0, 2, size = self.__len__())
        self.icl_label = np.array([self.icl_task_types[i] for i in np.random.randint(0, len(self.icl_task_types), size = self.__len__())])

    def pad_sequence(self):
        self.pad_len = self.required_len - self.length_timeseries
        # Pad data with zeros from the left
        self.data = np.pad(self.data, ((self.pad_len, 0), (0, 0)))
        self.length_timeseries = self.data.shape[0]

    def get_single(self, index, channel = None):
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
            
        if channel is None: # no CI applied
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
        else:
            if icl_class == 0: # Future
                input_seq = self.data[seq_start:seq_end, channel].reshape((-1, 1)).T
                forecast_seq = self.data[seq_end:pred_end, channel].reshape((-1, 1)).T
            elif icl_class == 1: # Past
                seq_end = seq_end - self.seq_len + self.forecast_horizon
                input_seq = self.data[seq_end:pred_end, channel].reshape((-1, 1)).T
                forecast_seq = self.data[seq_start:seq_end, channel].reshape((-1, 1)).T
            elif icl_class == 2: #Imputation
                # pad_len = int((self.seq_len / 8)*0.3)*8
                pad_len = self.forecast_horizon
                pad_start = np.random.randint(1, int(self.seq_len / 8) - int(pad_len / 8))*8
                forecast_seq = self.data[seq_start + pad_start : seq_start + pad_start + pad_len, channel].reshape((-1, 1)).T
                input_seq = self.data[seq_start:seq_end, channel].reshape((-1, 1)).T
                input_mask[pad_start:pad_start + pad_len] = 0
        
        return input_seq, input_mask, forecast_seq

    def __getitem__(self, index):
        index_in_channel = index % self.per_channel_length
        channel = int(index / self.per_channel_length)
        input_seq, input_mask, forecast_seq = self.get_single(index_in_channel, channel)
        if self.num_icl_examples == 0:
            return input_seq, input_mask, forecast_seq
        # Get ICL examples. TODO: avoid index of target
        # print(self.icl_label == self.icl_label[index])
        candidates = np.argwhere(self.icl_label == self.icl_label[index]).squeeze() - channel*self.per_channel_length
        candidates = candidates[(candidates >= 0)&(candidates < self.per_channel_length)]
        candidates = np.random.choice(candidates, self.num_icl_examples)
        examples = [self.get_single(i, channel) for i in candidates]
        # print([i.shape for i in examples[0]])
        masks = np.concatenate([np.concatenate((res[1], np.ones(res[2].shape[1])), axis = 0) for res in examples], axis = 0)
        examples = np.concatenate([np.concatenate((r1, r3), axis=1) for r1, r3 in [(res[0], res[2]) for res in examples]], axis = 1)
        # print(masks.shape, examples.shape)
        return np.concatenate((examples, input_seq), axis = 1), np.concatenate((masks, input_mask), axis = 0), forecast_seq

    def __len__(self):
        if self.length_timeseries < self.seq_len + self.forecast_horizon:
            return 1
        return ((
            self.length_timeseries - self.seq_len - self.forecast_horizon
        ) // self.stride + 1)*self.n_channels

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
    
def finetune_lptm(lptm, dataset, task_name="forecasting", **kwargs):
        # arguments
    max_lr = 1e-4 if "lr" not in kwargs else kwargs["lr"]
    max_epoch = 2 if "epoch" not in kwargs else kwargs["epoch"]
    max_norm = 5.0 if "norm" not in kwargs else kwargs["norm"]
    mask_ratio = 0.25 if "mask_ratio" not in kwargs else kwargs["mask_ratio"]

    dataloader = dataset.get_data_loader()
    criterion = torch.nn.MSELoss()
    if task_name == "classification":
        criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(lptm.model.parameters(), lr=max_lr)
    criterion.to(lptm.device)
    scaler = torch.amp.GradScaler()

    total_steps = len(dataloader) * max_epoch
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=max_lr, total_steps=total_steps, pct_start=0.3
    )
    lptm.model.to(lptm.device)
    lptm.model.train()

    for epoch in range(max_epoch):
        losses = []
        # for i, data in tqdm(enumerate(dataloader), total = int(len(dataset) / dataset.batchsize)):
        for i, data in enumerate(dataloader):
            # unpack the data
            if task_name == "forecasting":
                timeseries, input_mask, forecast = data
                # Move the data to the GPU
                timeseries = timeseries.float().to(lptm.device)
                input_mask = input_mask.to(lptm.device)
                forecast = forecast.float().to(lptm.device)
                with torch.amp.autocast(device_type="cuda"):
                    output = lptm.model(x_enc=timeseries, input_mask=input_mask)
                loss = criterion(output.forecast, forecast)

            optimizer.zero_grad(set_to_none=True)
            # Scales the loss for mixed precision training
            scaler.scale(loss).backward()

            # Clip gradients
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(lptm.model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()

            losses.append(loss.item())

        losses = np.array(losses)
        average_loss = np.average(losses)
        print(f"Epoch {epoch}: Train loss: {average_loss:.3f}")

        scheduler.step()

    return lptm.model

if __name__ == '__main__':
    
    seq_len = 384      
    horizon = 192
    
    train_batch_size = 256
    test_batch_size = 256
    num_icl_examples = 4
    
    # data_path = "/nethome/sli999/TSFMProject/src/tsfmproject/models/moment/data/ETTh1.csv"
    
    # train_data_path = '/nethome/sxu452/Time-LLM/dataset/weather/weather.csv'
    # train_data_path = "/nethome/sxu452/Time-LLM/dataset/ETT-small/ETTm1.csv"
    train_data_path = '/nethome/sxu452/Time-LLM/dataset/exchange_rate/exchange_rate.csv'
    datetime_col = 'date'
    
    
    # train_data_path = '/nethome/sxu452/Samay/example/metr-la_20.csv'
    # datetime_col = None
    
    # test_data_path = '/nethome/sxu452/Time-LLM/dataset/electricity/electricity.csv'
    test_data_path = train_data_path
    
    train_dataset = LPTMMixedCIDataset(
        name="ett",
        datetime_col=datetime_col,
        path= train_data_path,
        mode="train",
        horizon=horizon,
        seq_len=seq_len,
        batchsize=train_batch_size,
        num_icl_examples=num_icl_examples,
        icl_task_types=[0,1]
    )
    val_forecast = LPTMMixedCIDataset(
        name="ett",
        datetime_col=datetime_col,
        path= test_data_path,
        mode="test",
        horizon=horizon,
        seq_len=seq_len,
        batchsize=test_batch_size,
        num_icl_examples=num_icl_examples,
        icl_task_types=[0]
    )
    val_history = LPTMMixedCIDataset(
        name="ett",
        datetime_col=datetime_col,
        path = test_data_path,
        mode="test",
        horizon=horizon,
        seq_len=seq_len,
        batchsize=test_batch_size,
        num_icl_examples=num_icl_examples,
        icl_task_types=[1]
    )
    val_imputation = LPTMMixedCIDataset(
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
    
    config = {
        "task_name": "forecasting2",
        "forecast_horizon": train_dataset[0][2].shape[-1],
        'seq_len': train_dataset[0][0].shape[-1],
        "head_dropout": 0,
        "weight_decay": 0,
        "max_patch": 16,
        "freeze_encoder": False,  # Freeze the patch embedding layer
        "freeze_embedder": False,  # Freeze the transformer encoder
        "freeze_head": False,  # The linear forecasting head must be trained
        "freeze_segment": False,  # Freeze the segmention module
    }
    model = LPTMModel(config)
    print(model.device)

    for run in (bar:=tqdm(range(300))):
        # _, trues, preds, _ = model.evaluate(val_forecast)
        # forecast_mse = np.mean((trues - preds) ** 2)
        # forecast_mae = np.mean(np.abs(trues - preds))
        finetune_lptm(model, train_dataset, epoch = 1)
        _, trues, preds, _ = model.evaluate(val_history)
        history_mse = np.mean((trues - preds) ** 2)
        history_mae = np.mean(np.abs(trues - preds))
        # forecast_scores = model.evaluate(val_forecast)
        # forecast_mse, forecast_mae = forecast_scores['mse'], forecast_scores['mae']
        # history_scores = model.evaluate(val_history)
        # history_mse, history_mae = history_scores['mse'], history_scores['mae']
        _, trues, preds, _ = model.evaluate(val_imputation)
        imputation_mse = np.mean((trues - preds) ** 2)
        imputation_mae = np.mean(np.abs(trues - preds))
        bar.set_description(f'Past: MSE {history_mse:.3f} MAE {history_mae:.3f} Imputation: MSE {imputation_mse:.3f} MAE {imputation_mae:.3f} ')
        