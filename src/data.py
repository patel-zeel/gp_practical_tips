from os.path import join

import torch

from scipy.io import loadmat

def load_data(cfg):
    torch_dtypes = {"float32": torch.float32, "float64": torch.float64, "int32": torch.int32, "int64": torch.int64}
    
    if cfg.data.name == "pol":
        data = loadmat(join(cfg.data.base_path, "pol/pol.mat"))
        data = torch.tensor(data['data'], dtype=torch_dtypes[cfg.data.dtype])
        X = data[:, :-1]
        y = data[:, -1]
        
    torch.manual_seed(cfg.data.seed)
    # split data into train, validation & test
    idx = torch.randperm(X.shape[0])
    train_size = int(len(idx)*cfg.data.train_fraction)
    val_size = int(len(idx)*cfg.data.val_fraction)
    
    train_idx = idx[:train_size]
    val_idx = idx[train_size:train_size+val_size]
    test_idx = idx[train_size+val_size:]
    
    X_train, y_train = X[train_idx], y[train_idx]        
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    # normalize data
    X_scale = X_train.std(dim=0)
    X_mean = X_train.mean(dim=0)
    y_scale = y_train.std()
    y_mean = y_train.mean()
    
    # print("y_mean: ", y_mean)
    
    X_train = (X_train - X_mean) / X_scale
    X_val = (X_val - X_mean) / X_scale
    X_test = (X_test - X_mean) / X_scale
    
    y_train = (y_train - y_mean) / y_scale
    y_val = (y_val - y_mean) / y_scale
    y_test = (y_test - y_mean) / y_scale
    
    # print(f"Train: {X_train.shape}, {y_train.shape}")
    # print(f"Val: {X_val.shape}, {y_val.shape}")
    # print(f"Test: {X_test.shape}, {y_test.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test