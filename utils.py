import torch


def replicate_input(x):
    return x.detach().clone()


def to_one_hot(y, num_classes=10):
    """
    Take a batch of label y with n dims and convert it to
    1-hot representation with n+1 dims.
    Link: https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/24
    """
    y = replicate_input(y).view(-1, 1)
    y_one_hot = y.new_zeros((y.size()[0], num_classes)).scatter_(1, y, 1)
    return y_one_hot
