import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from samay.dataset import BaseDataset
from samay.model import TimesfmModel
from samay.models.timesfm.timesfm import pytorch_patched_decoder as ppd
import os
from tqdm import tqdm
from samay.models.timesfm.timesfm import time_features
import logging
import torch.nn.functional as F

class CustomTorchDataset(Dataset):
    def __init__(self, gen_fn, output_shapes):
        self.gen_fn = gen_fn
        self.output_shapes = output_shapes

        self.data = list(self._generate_data())
    
    def _generate_data(self):
        for item in self.gen_fn():
            yield item

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        return tuple(torch.tensor(value, dtype=dtype) 
                     for value, dtype in zip(item, self.output_shapes))

class TimeSeriesdata(object):
  """Data loader class."""

  def __init__(
      self,
      data_path,
      datetime_col,
      num_cov_cols,
      cat_cov_cols,
      ts_cols,
      train_range,
      val_range,
      test_range,
      hist_len,
      pred_len,
      batch_size,
      freq='H',
      normalize=True,
      epoch_len=None,
      holiday=False,
      permute=True,
  ):
    """Initialize objects.

    Args:
      data_path: path to csv file
      datetime_col: column name for datetime col
      num_cov_cols: list of numerical global covariates
      cat_cov_cols: list of categorical global covariates
      ts_cols: columns corresponding to ts
      train_range: tuple of train ranges
      val_range: tuple of validation ranges
      test_range: tuple of test ranges
      hist_len: historical context
      pred_len: prediction length
      batch_size: batch size (number of ts in a batch)
      freq: freq of original data
      normalize: std. normalize data or not
      epoch_len: num iters in an epoch
      holiday: use holiday features or not
      permute: permute ts in train batches or not

    Returns:
      None
    """
    self.data_df = pd.read_csv(open(data_path, 'r'))
    if not num_cov_cols:
      self.data_df['ncol'] = np.zeros(self.data_df.shape[0])
      num_cov_cols = ['ncol']
    if not cat_cov_cols:
      self.data_df['ccol'] = np.zeros(self.data_df.shape[0])
      cat_cov_cols = ['ccol']
    self.data_df.fillna(0, inplace=True)
    self.data_df.set_index(pd.DatetimeIndex(self.data_df[datetime_col]),
                           inplace=True)
    self.num_cov_cols = num_cov_cols
    self.cat_cov_cols = cat_cov_cols
    self.ts_cols = ts_cols
    self.train_range = train_range
    self.val_range = val_range
    self.test_range = test_range
    data_df_idx = self.data_df.index
    date_index = data_df_idx.union(
        pd.date_range(
            data_df_idx[-1] + pd.Timedelta(1, freq=freq),
            periods=pred_len + 1,
            freq=freq,
        ))
    self.time_df = time_features.TimeCovariates(
        date_index, holiday=holiday).get_covariates()
    self.hist_len = hist_len
    self.pred_len = pred_len
    self.batch_size = batch_size
    self.freq = freq
    self.normalize = normalize
    self.data_mat = self.data_df[self.ts_cols].to_numpy().transpose()      # unique_id, y  
    self.data_mat = self.data_mat[:, 0:self.test_range[1]]         
    self.time_mat = self.time_df.to_numpy().transpose()
    self.num_feat_mat = self.data_df[num_cov_cols].to_numpy().transpose()       # 0
    self.cat_feat_mat, self.cat_sizes = self._get_cat_cols(cat_cov_cols)        # 0
    self.normalize = normalize
    if normalize:
      self._normalize_data()
    logging.info(
        'Data Shapes: %s, %s, %s, %s',
        self.data_mat.shape,
        self.time_mat.shape,
        self.num_feat_mat.shape,
        self.cat_feat_mat.shape,
    )
    self.epoch_len = epoch_len
    self.permute = permute

  def _get_cat_cols(self, cat_cov_cols):
    """Get categorical columns."""
    cat_vars = []
    cat_sizes = []
    for col in cat_cov_cols:
      dct = {x: i for i, x in enumerate(self.data_df[col].unique())}
      cat_sizes.append(len(dct))
      mapped = self.data_df[col].map(lambda x: dct[x]).to_numpy().transpose()  # pylint: disable=cell-var-from-loop
      cat_vars.append(mapped)
    return np.vstack(cat_vars), cat_sizes

  def _normalize_data(self):
    self.scaler = StandardScaler()
    train_mat = self.data_mat[:, 0:self.train_range[1]]
    self.scaler = self.scaler.fit(train_mat.transpose())
    self.data_mat = self.scaler.transform(self.data_mat.transpose()).transpose()

  def train_gen(self):
    """Generator for training data."""
    num_ts = len(self.ts_cols)
    perm = np.arange(
        self.train_range[0] + self.hist_len,
        self.train_range[1] - self.pred_len,
    )
    perm = np.random.permutation(perm)
    hist_len = self.hist_len
    logging.info('Hist len: %s', hist_len)
    if not self.epoch_len:
      epoch_len = len(perm)
    else:
      epoch_len = self.epoch_len
    for idx in perm[0:epoch_len]:
      for _ in range(num_ts // self.batch_size + 1):
        if self.permute:
          tsidx = np.random.choice(num_ts, size=self.batch_size, replace=False)
        else:
          tsidx = np.arange(num_ts)
        dtimes = np.arange(idx - hist_len, idx + self.pred_len)
        (
            bts_train,
            bts_pred,
            bfeats_train,
            bfeats_pred,
            bcf_train,
            bcf_pred,
        ) = self._get_features_and_ts(dtimes, tsidx, hist_len)

        all_data = [
            bts_train,
            bfeats_train,
            bcf_train,
            bts_pred,
            bfeats_pred,
            bcf_pred,
            tsidx,
        ]
        yield tuple(all_data)

  def test_val_gen(self, mode='val', shift=1):
    """Generator for validation/test data."""
    if mode == 'val':
      start = self.val_range[0]
      end = self.val_range[1] - self.pred_len + 1
    elif mode == 'test':
      start = self.test_range[0]
      end = self.test_range[1] - self.pred_len + 1
    else:
      raise NotImplementedError('Eval mode not implemented')
    num_ts = len(self.ts_cols)
    hist_len = self.hist_len
    logging.info('Hist len: %s', hist_len)
    perm = np.arange(start, end)
    if self.epoch_len:
      epoch_len = self.epoch_len
    else:
      epoch_len = len(perm)
    for i in range(0, epoch_len, shift):
      idx = perm[i]
      for batch_idx in range(0, num_ts, self.batch_size):
        tsidx = np.arange(batch_idx, min(batch_idx + self.batch_size, num_ts))
        dtimes = np.arange(idx - hist_len, idx + self.pred_len)
        (
            bts_train,
            bts_pred,
            bfeats_train,
            bfeats_pred,
            bcf_train,
            bcf_pred,
        ) = self._get_features_and_ts(dtimes, tsidx, hist_len)
        all_data = [
            bts_train,
            bfeats_train,
            bcf_train,
            bts_pred,
            bfeats_pred,
            bcf_pred,
            tsidx,
        ]
        yield tuple(all_data)

  def _get_features_and_ts(self, dtimes, tsidx, hist_len=None):
    """Get features and ts in specified windows."""
    if hist_len is None:
      hist_len = self.hist_len
    data_times = dtimes[dtimes < self.data_mat.shape[1]]
    bdata = self.data_mat[:, data_times]   # unique_id, y
    bts = bdata[tsidx, :]
    bnf = self.num_feat_mat[:, data_times] # 0
    bcf = self.cat_feat_mat[:, data_times] # 0
    btf = self.time_mat[:, dtimes]  # holiday features
    if bnf.shape[1] < btf.shape[1]:
      rem_len = btf.shape[1] - bnf.shape[1]
      rem_rep = np.repeat(bnf[:, [-1]], repeats=rem_len)
      rem_rep_cat = np.repeat(bcf[:, [-1]], repeats=rem_len)
      bnf = np.hstack([bnf, rem_rep.reshape(bnf.shape[0], -1)])
      bcf = np.hstack([bcf, rem_rep_cat.reshape(bcf.shape[0], -1)])
    bfeats = np.vstack([btf, bnf])   # holiday features, 0
    bts_train = bts[:, 0:hist_len]   # unique_id, y
    bts_pred = bts[:, hist_len:]     # unique_id, y
    bfeats_train = bfeats[:, 0:hist_len]  # holiday features, 0
    bfeats_pred = bfeats[:, hist_len:]    # holiday features, 0
    bcf_train = bcf[:, 0:hist_len]        # 0
    bcf_pred = bcf[:, hist_len:]          # 0
    return bts_train, bts_pred, bfeats_train, bfeats_pred, bcf_train, bcf_pred

  def torch_dataset(self, mode='train', shift=1):
    """Tensorflow Dataset."""
    if mode == 'train':
      gen_fn = self.train_gen
    else:
      gen_fn = lambda: self.test_val_gen(mode, shift)
    # output_types = tuple([tf.float32] * 2 + [tf.int32] + [tf.float32] * 2 +
    #                      [tf.int32] * 2)
    output_shapes = tuple([torch.float32] * 2 + [torch.int32] + [torch.float32] * 2 +
                          [torch.int32] * 2)
    dataset = CustomTorchDataset(gen_fn, output_shapes)
    return dataset

class TimesfmBasicDataset(BaseDataset):
    """
    Dataset class for TimesFM model
    Data Format:
    Dict with keys:
    input_ts: np.ndarray, historical time series data
    actual_ts: np.ndarray, actual time series data
    """

    def __init__(
        self,
        name=None,
        datetime_col="ds",
        path=None,
        batchsize=32,
        mode="train",
        boundaries=(0, 0, 0),
        context_len=128,
        horizon_len=32,
        freq="h",
        normalize=True,
        num_icl_examples = 8,
        icl_task_types = [0, 1],
        **kwargs,
    ):
        super().__init__(
            name=name,
            datetime_col=datetime_col,
            path=path,
            batchsize=batchsize,
            mode=mode,
        )
        self.context_len = context_len
        self.horizon_len = horizon_len
        self.freq = freq
        self.normalize = normalize
        self.data = pd.read_csv(self.data_path)
        if boundaries == (0, 0, 0):
            # Default boundaries: train 60%, val 20%, test 20%
            self.boundaries = [
                int(len(self.data) * 0.6),
                int(len(self.data) * 0.8),
                len(self.data) - 1,
            ]
        else:
            self.boundaries = boundaries
        self.ts_cols = [col for col in self.data.columns if col != self.datetime_col]
        tfdtl = TimeSeriesdata(
            data_path=self.data_path,
            datetime_col=self.datetime_col,
            num_cov_cols=None,
            cat_cov_cols=None,
            ts_cols=np.array(self.ts_cols),
            train_range=[0, self.boundaries[0]],
            val_range=[self.boundaries[0], self.boundaries[1]],
            test_range=[self.boundaries[1], self.boundaries[2]],
            hist_len=self.context_len,
            pred_len=self.horizon_len,
            batch_size=self.batchsize,
            freq=self.freq,
            normalize=self.normalize,
            epoch_len=None,
            holiday=False,
            permute=False,
        )
        if self.mode == "train":
            tfset = tfdtl.torch_dataset(mode="train", shift=1)
        else:
            tfset = tfdtl.torch_dataset(mode="test", shift=self.horizon_len)
        self.dataset = tfset

    def get_data_loader(self):
        if self.mode == "train":
            return DataLoader(self.dataset, batch_size=self.batchsize, shuffle=True)
        else:
            return DataLoader(self.dataset, shuffle=False)

    def preprocess_train_batch(self, data):
        past_ts = data[0].reshape(self.batchsize * len(self.ts_cols), -1)
        actual_ts = data[3].reshape(self.batchsize * len(self.ts_cols), -1)
        return {"input_ts": past_ts, "actual_ts": actual_ts}

    def preprocess_eval_batch(self, data):
        past_ts = data[0]
        actual_ts = data[3]
        return {"input_ts": past_ts, "actual_ts": actual_ts}

    def preprocess(self, data):
        if self.mode == "train":
            return self.preprocess_train_batch(data)
        else:
            return self.preprocess_eval_batch(data)

class TimesfmICLWrapper(Dataset):
  def __init__(self, base_dataset, context_len, horizon_len, icl_task_types = [0,1], num_icl_examples = 4):
    super().__init__()
    self.dataset = base_dataset
    self.context_len = context_len
    self.pred_len = horizon_len
    self.icl_task_types = icl_task_types
    self.num_icl_examples = num_icl_examples
    
    self.icl_label = np.array([self.icl_task_types[i] for i in np.random.randint(0, len(self.icl_task_types), size = len(self.dataset))])
    
  def get_single(self, index):
    input_ts, _, _, actual_ts, _, _, _ = self.dataset[index]
    if self.icl_label[index] == 0: # Future
      return input_ts, actual_ts
    elif self.icl_label[index] == 1: # History
      len_input = input_ts.shape[-1]
      len_actual = actual_ts.shape[-1]
      concated = torch.concatenate((input_ts, actual_ts), dim = 1)
      return torch.flip(concated[:, -len_input:], dims = [1]), torch.flip(concated[:, :len_actual], dims = [1])
      # return concated[:, -len_input:], concated[:, :len_actual]
    elif self.icl_label[index] == 2: # Imputation
      len_input = input_ts.shape[-1]
      len_actual = actual_ts.shape[-1]
      pad_start = np.random.randint(1, int(len_input / 8) - int(len_actual / 8))*8
      inp = input_ts.clone().detach()
      outp = input_ts[:, pad_start : pad_start +  len_actual]
      inp[:, pad_start: pad_start + len_actual] = 0
      # inp[:, -pad_start:] = inp[:, :pad_start]
      # inp[:, :-pad_start] = 0
      return inp, outp
    
  def __len__(self):
    return len(self.dataset)
    
  def __getitem__(self, index):
    input_seq, forecast_seq = self.get_single(index)
    if self.num_icl_examples > 0:
      candidates = np.random.choice(np.argwhere(self.icl_label == self.icl_label[index]).squeeze(), self.num_icl_examples)
      examples = [self.get_single(i) for i in candidates]
      # examples = np.concatenate([np.concatenate((r1, r3), axis=1) for r1, r3 in [(res[0], res[1]) for res in examples]], axis = 1)
      examples = torch.concatenate([torch.concatenate((torch.Tensor(r1), torch.Tensor(r3)), dim=1) for r1, r3 in [(res[0], res[1]) for res in examples]], dim = 1)
      # return torch.Tensor(np.concatenate((examples, input_seq), axis = 1)), torch.Tensor(forecast_seq)
      return torch.concatenate((examples, input_seq), dim = 1), forecast_seq
    else:
      return torch.Tensor(input_seq), torch.Tensor(forecast_seq)

def _quantile_loss(pred, actual, quantile):
  """Calculates quantile loss."""
  dev = actual - pred
  loss_first = dev * quantile
  loss_second = -dev * (1.0 - quantile)
  return 2 * torch.where(loss_first >= 0, loss_first, loss_second)

def compute_predictions_fixed(model, input_batch,train_horizon_len=128):
  horizon_len = train_horizon_len
  output_patch_len = model.core_layer.config.patch_len
  input_ts = input_batch["input_ts"]
  # input_padding = torch.zeros_like(input_ts)
  input_padding = torch.zeros([input_ts.shape[0], input_ts.shape[1] + horizon_len], dtype=torch.float32)

  context_len = input_ts.shape[1]
  input_patch_len = model.core_layer.config.patch_len
  context_pad = ((context_len + input_patch_len - 1) // input_patch_len) * input_patch_len - context_len

  input_ts = F.pad(input_ts, (context_pad, 0))
  input_padding = F.pad(input_padding, (context_pad, 0), value=1)
  freq = torch.ones([input_ts.shape[0], 1], dtype=torch.int32) * model.freq

  # Check device
  input_padding = input_padding.to(input_ts.device)
  freq = freq.to(input_ts.device)

  # return self.core_layer(input_ts, input_padding, freq)
  return model.core_layer.decode(input_ts, input_padding, freq, horizon_len, output_patch_len)
  
def compute_loss_fixed(model, prediction_output, input_batch):
  output_ts = prediction_output[1]
  actual_ts = input_batch["actual_ts"]

  # pred_ts = output_ts[:, -1, :actual_ts.shape[1], :]
  pred_ts = output_ts
  loss = torch.square(pred_ts[:, :, 0] - actual_ts).mean()

  for i, q in enumerate(model.core_layer.config.quantiles):
    loss += _quantile_loss(pred_ts[:, :, i + 1], actual_ts, q).mean()

  return loss
  
repo = "google/timesfm-1.0-200m-pytorch"

horizon_len = 96 
context_len = 192

config = {
    "context_len": context_len,
    "horizon_len": horizon_len,
    "backend": "gpu",
    "per_core_batch_size": 32,
    "input_patch_len": 32,
    "output_patch_len": 128,
    "num_layers": 20,
    "model_dims": 1280,
    "quantiles": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
}

# path = '/nethome/sxu452/Time-LLM/dataset/weather/weather.csv'
# path = '/nethome/sxu452/Time-LLM/dataset/ETT-small/ETTm1.csv'
# path = '/nethome/sxu452/Time-LLM/dataset/exchange_rate/exchange_rate.csv'
# datetime_col = 'date'


path = '/nethome/sxu452/Samay/example/pems-bays_20.csv'
datetime_col = 'date'
train_icl_tasks = [0,2]

test_dataset = TimesfmBasicDataset(
    name="ett",
    datetime_col=datetime_col,
    path=path,
    mode="test",
    context_len=config["context_len"],
    horizon_len=horizon_len,
)

batch_size = 7
num_icl_examples = 4
wrapped_test_dataset = TimesfmICLWrapper(test_dataset.dataset, test_dataset.context_len, test_dataset.horizon_len, icl_task_types=[0], num_icl_examples= num_icl_examples)
test_loader = DataLoader(wrapped_test_dataset, batch_size = batch_size)

print(wrapped_test_dataset[0][0].shape, wrapped_test_dataset[0][1].shape)
tfm_config = {
    "context_len": wrapped_test_dataset[0][0].shape[-1],
    "horizon_len": wrapped_test_dataset[0][1].shape[-1],
    "backend": "gpu",
    "per_core_batch_size": batch_size,
    "input_patch_len": 32,
    "output_patch_len": 128,
    "num_layers": 20,
    "model_dims": 1280,
    "quantiles": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
}
tfm = TimesfmModel(config=tfm_config, repo=repo)

core_layer_tpl = tfm.model._model
# Todo: whether add freq
FinetunedModel = ppd.PatchedDecoderFinetuneModel(core_layer_tpl=core_layer_tpl)
FinetunedModel.to(tfm.device)
FinetunedModel.train()

wrapped_test_dataset_history = TimesfmICLWrapper(test_dataset.dataset, test_dataset.context_len, test_dataset.horizon_len, icl_task_types=[1], num_icl_examples= num_icl_examples)
test_loader_history = DataLoader(wrapped_test_dataset_history, batch_size = batch_size)

wrapped_test_dataset_imp = TimesfmICLWrapper(test_dataset.dataset, test_dataset.context_len, test_dataset.horizon_len, icl_task_types=[2], num_icl_examples= num_icl_examples)
test_loader_imp = DataLoader(wrapped_test_dataset_imp, batch_size = batch_size)

train_dataset = TimesfmBasicDataset(
    name="ett",
    datetime_col=datetime_col,
    path=path,
    mode="train",
    context_len=config["context_len"],
    horizon_len=horizon_len,
)

print(len(train_dataset))
wrapper_train_dataset = TimesfmICLWrapper(train_dataset.dataset, train_dataset.context_len, train_dataset.horizon_len, icl_task_types=train_icl_tasks, num_icl_examples= 4)
train_loader = DataLoader(wrapper_train_dataset, batch_size = batch_size)
optimizer = torch.optim.Adam(FinetunedModel.parameters(), lr=1e-6)

# for i, (input_ts, actual_ts) in enumerate(train_loader):
#   print(input_ts.shape, actual_ts.shape)
for epoch in (bar:=tqdm(range(100))):
  avg_loss = 0
  for i, (input_ts, actual_ts) in tqdm(enumerate(train_loader), total = len(train_loader)):
      input_ts = input_ts.squeeze().reshape((input_ts.shape[0]*input_ts.shape[1], input_ts.shape[2])).to(tfm.device)
      actual_ts = actual_ts.squeeze().reshape((actual_ts.shape[0]*actual_ts.shape[1], actual_ts.shape[2])).to(tfm.device)

      optimizer.zero_grad()
      # outputs = FinetunedModel.compute_predictions(
      #     {'input_ts': input_ts, 'actual_ts': actual_ts}
      # )  # b, n, seq_len, 1+quantiles
      # loss = FinetunedModel.compute_loss(outputs, {'input_ts': input_ts, 'actual_ts': actual_ts})
      outputs = compute_predictions_fixed(FinetunedModel, {'input_ts': input_ts, 'actual_ts': actual_ts}, horizon_len)
      loss = compute_loss_fixed(FinetunedModel, outputs, {'input_ts': input_ts, 'actual_ts': actual_ts})
      loss.backward()
      optimizer.step()
      avg_loss += loss.item()
  avg_loss /= len(train_loader)
  bar.set_description(f"Epoch {epoch}, Loss: {avg_loss}")
  
  if True:
    with torch.no_grad():
      trues, preds, histories, losses = [], [], [], []
      for i, (input_ts, actual_ts) in enumerate(test_loader):
          input_ts = input_ts.squeeze().reshape((input_ts.shape[0]*input_ts.shape[1], input_ts.shape[2]))
          actual_ts = actual_ts.detach().cpu().numpy()
          actual_ts = actual_ts.squeeze().reshape((actual_ts.shape[0]*actual_ts.shape[1], actual_ts.shape[2]))

          output, _ = tfm.model.forecast(input_ts)
          output = output[:, 0 : actual_ts.shape[1]]

          loss = np.mean((output - actual_ts) ** 2)
          losses.append(loss.item())
          trues.append(actual_ts)
          preds.append(output)
          histories.append(input_ts)

      losses = np.array(losses)
      average_loss = np.average(losses)
      trues = np.concatenate(trues, axis=0)
      preds = np.concatenate(preds, axis=0)
      forecast_mse = np.mean((trues - preds) ** 2)
      forecast_mae = np.mean(np.abs(trues - preds))
      trues, preds, histories, losses = [], [], [], []
      for i, (input_ts, actual_ts) in enumerate(test_loader_history):
          input_ts = input_ts.squeeze().reshape((input_ts.shape[0]*input_ts.shape[1], input_ts.shape[2]))
          actual_ts = actual_ts.detach().cpu().numpy()
          actual_ts = actual_ts.squeeze().reshape((actual_ts.shape[0]*actual_ts.shape[1], actual_ts.shape[2]))

          output, _ = tfm.model.forecast(input_ts)
          output = output[:, 0 : actual_ts.shape[1]]

          loss = np.mean((output - actual_ts) ** 2)
          losses.append(loss.item())
          trues.append(actual_ts)
          preds.append(output)
          histories.append(input_ts)

      losses = np.array(losses)
      average_loss = np.average(losses)
      trues = np.concatenate(trues, axis=0)
      preds = np.concatenate(preds, axis=0)
      history_mse = np.mean((trues - preds) ** 2)
      history_mae = np.mean(np.abs(trues - preds))
      trues, preds, histories, losses = [], [], [], []
      for i, (input_ts, actual_ts) in enumerate(test_loader_imp):
          input_ts = input_ts.squeeze().reshape((input_ts.shape[0]*input_ts.shape[1], input_ts.shape[2]))
          actual_ts = actual_ts.detach().cpu().numpy()
          actual_ts = actual_ts.squeeze().reshape((actual_ts.shape[0]*actual_ts.shape[1], actual_ts.shape[2]))

          output, _ = tfm.model.forecast(input_ts)
          output = output[:, 0 : actual_ts.shape[1]]

          loss = np.mean((output - actual_ts) ** 2)
          losses.append(loss.item())
          trues.append(actual_ts)
          preds.append(output)
          histories.append(input_ts)

      losses = np.array(losses)
      average_loss = np.average(losses)
      trues = np.concatenate(trues, axis=0)
      preds = np.concatenate(preds, axis=0)
      imp_mse = np.mean((trues - preds) ** 2)
      imp_mae = np.mean(np.abs(trues - preds))
      print(f'Forecast MSE {forecast_mse:.3f} MAE {forecast_mae:.3f} History: MSE {history_mse:.3f} MAE {history_mae:.3f}  Imputation: MSE {imp_mse:.3f} MAE {imp_mae:.3f}')
