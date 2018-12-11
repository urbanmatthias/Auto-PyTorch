import sklearn.metrics as metrics

# classification metrics
def accuracy(y_pred, y_true):
    return (undo_ohe(y_true) == undo_ohe(y_pred)).float().mean().item() * 100

def auc_metric(y_pred, y_true):
    return (2 * metrics.roc_auc_score(y_true.cpu().numpy(), y_pred.cpu().numpy()) - 1) * 100


# multilabel metric
def multilabel_accuracy(y_pred, y_true):
    return (y_true.long() == (y_pred > 0.5).long()).float().mean().item() * 100

# regression metric
def mean_distance(y_pred, y_true):
    return (y_true - y_pred).abs().mean().item()

def undo_ohe(y):
    if len(y.shape) == 1:
        return(y)
    return y.max(1)[1]