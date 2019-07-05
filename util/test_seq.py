import torch
from torch.autograd import Variable

def generate_test_seq():
	inputs = Variable(torch.LongTensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 0], [3, 4 ,5, 0, 0]]))
    mask = torch.gt(inputs, 0).float()
    return inputs, mask