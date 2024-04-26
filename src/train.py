import torch

### Quantile loss implementation
def D_p(y_pred, y_true, p):
    # return torch.max(p * (y_pred - y_true), (p-1) * (y_pred - y_true))
    return (p - (y_true < y_pred)*1) * (y_true - y_pred)

def R_p(y_preds, y_trues, p):
    return torch.sum(D_p(y_preds, y_trues, p)), torch.sum(torch.abs(y_trues))

# Mean absolute percentage error
def mape(y_preds, y_trues):
    return torch.mean(torch.abs(y_trues - y_preds) / torch.abs(y_trues))

def _train_step(model, compute_loss, train_dataloader, optimizer, mask, device):
    """
    Train a model for a single epoch/step.
    :param model: Transformer model
    :param compute_loss: function that takes:
        - the model,
        - src_X (input covariates),
        - src_fX (input features),
        - tgt_X (target covariates),
        - tgt_fX (target features),
        - and the subsequent mask
        and applies the model and calculates the loss.
        This is useful to abstract the implementation of decoder-only and encoder-decoder models.
        Function returns the loss and the output of the model.
    :param train_dataloader: torch training set dataloader
    :param optimizer: torch optimizer (e.g., torch.optim.Adam)
    :param mask: matrix that masks the inputs from attending to future inputs
    :param device: the device to train on.
    :return: returns the batch training loss.
    """
    model.train()
    train_loss = 0
    n = 0
    mape_loss = 0
    mask = mask.to(device)
    for sample in train_dataloader:
        src_X, src_fX, tgt_X, tgt_fX = (v.to(device) for v in sample)

        loss, out = compute_loss(model, src_X, src_fX, tgt_X, tgt_fX, mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * src_X.shape[0]
        mape_loss += mape(out, tgt_fX).item() * src_X.shape[0]

        n += src_X.shape[0]

    return train_loss / n, mape_loss / n

def _val_step(model, compute_loss, val_dataloader, p, mask, device):
    """
    validate a model for a single epoch/step.
    :param model: Transformer model
    :param compute_loss: function that takes:
        - the model,
        - src_X (input covariates),
        - src_fX (input features),
        - tgt_X (target covariates),
        - tgt_fX (target features),
        - and the subsequent mask
        and applies the model and calculates the loss.
        This is useful to abstract the implementation of decoder-only and encoder-decoder models.
        Function returns the loss and the output of the model.
    :param p: list of quantiles from which to compute the quantile loss.
    :param mask: matrix that masks the inputs from attending to future inputs
    :param device: the device to train on.
    :return: returns the batch training loss, and a tensor of quantile loss values based on p.
    """
    model.eval()
    val_loss = 0
    n = 0
    num = torch.zeros((len(p)), device=device)
    den = torch.zeros((len(p)), device=device)
    mape_loss = 0
    mask = mask.to(device)
    with torch.no_grad():
        for sample in val_dataloader:
            src_X, src_fX, tgt_X, tgt_fX = (v.to(device) for v in sample)

            loss, out = compute_loss(model, src_X, src_fX, tgt_X, tgt_fX, mask)
            val_loss += loss.item() * src_X.shape[0]
            mape_loss += mape(out, tgt_fX).item() * src_X.shape[0]
            n += src_X.shape[0]

            for i, p_i in enumerate(p):
                num_i, den_i = R_p(out, tgt_fX, p_i)
                num[i] += num_i
                den[i] += den_i
    return val_loss / n, mape_loss / n, (2 * num) / den


### Decoder-only
"""
wrapper aroound _train_step that injects a compute_loss function that works with the decoder-only Transformer.
"""
def train_step_dec_only(model, criterion, train_dataloader, optimizer, mask, shift, device):
    """
    shift: only applicable if the tgt sequence does not have the same length as the src sequence.
    In that case, 'shift' is the amount of steps that the tgt sequence is shifted from the src sequence.
    0 means no shift and assumes that len(src) == len(tgt).
    """
    # create loss function that complies with decoder-only interface
    def compute_loss(model, src_X, src_fX, _, tgt_fX, mask):
        out = model(src_X, src_fX, mask)
        return criterion(out[:, -shift:], tgt_fX), out[:, -shift:]

    return _train_step(model, compute_loss, train_dataloader, optimizer, mask, device)

"""
wrapper aroound _val_step that injects a compute_loss function that works with the decoder-only Transformer.
"""
def val_step_dec_only(model, criterion, val_dataloader, p, mask, shift, device):
    """
    shift: only applicable if the tgt sequence does not have the same length as the src sequence.
    In that case, 'shift' is the amount of steps that the tgt sequence is shifted from the src sequence.
    0 means no shift and assumes that len(src) == len(tgt).
    """
    # create loss function that complies with decoder-only interface
    def compute_loss(model, src_X, src_fX, _, tgt_fX, mask):
        out = model(src_X, src_fX, mask)
        return criterion(out[:, -shift:], tgt_fX), out[:, -shift:]

    return _val_step(model, compute_loss, val_dataloader, p, mask, device)


### Probabilistic Decoder-only
def prob_train_step_dec_only(model, criterion, train_dataloader, optimizer, mask, shift, device):
    """
    shift: only applicable if the tgt sequence does not have the same length as the src sequence.
    In that case, 'shift' is the amount of steps that the tgt sequence is shifted from the src sequence.
    0 means no shift and assumes that len(src) == len(tgt).
    """
    # create loss function that complies with decoder-only interface
    def compute_loss(model, src_X, src_fX, _, tgt_fX, mask):
        mu, sigma = model(src_X, src_fX, mask)
        return criterion(mu[:, -shift:], tgt_fX, sigma[:, -shift:]), mu[:, -shift:]

    return _train_step(model, compute_loss, train_dataloader, optimizer, mask, device)

def prob_val_step_dec_only(model, criterion, val_dataloader, p, mask, shift, device):
    """
    shift: only applicable if the tgt sequence does not have the same length as the src sequence.
    In that case, 'shift' is the amount of steps that the tgt sequence is shifted from the src sequence.
    0 means no shift and assumes that len(src) == len(tgt).
    """
    # create loss function that complies with decoder-only interface
    def compute_loss(model, src_X, src_fX, _, tgt_fX, mask):
        mu, sigma = model(src_X, src_fX, mask)
        return criterion(mu[:, -shift:], tgt_fX, sigma[:, -shift:]), mu[:, -shift:]

    return _val_step(model, compute_loss, val_dataloader, p, mask, device)

### Encoder-decoder
def train_step_enc_dec(model, criterion, train_dataloader, optimizer, mask, shift, device):
    # create function that complies with encoder-decoder interface
    def compute_loss(model, src_X, src_fX, tgt_X, tgt_fX, mask):
        out = model(src_X, src_fX, tgt_X, tgt_fX, mask)
        return criterion(out[:, -shift:], tgt_fX), out[:, -shift:]

    return _train_step(model, compute_loss, train_dataloader, optimizer, mask, device)

def val_step_enc_dec(model, criterion, val_dataloader, p, mask, shift, device):
    # create function that complies with encoder-decoder interface
    def compute_loss(model, src_X, src_fX, tgt_X, tgt_fX, mask):
        out = model(src_X, src_fX, tgt_X, tgt_fX, mask)
        return criterion(out[:, -shift:], tgt_fX), out[:, -shift:]

    return _val_step(model, compute_loss, val_dataloader, p, mask, device)