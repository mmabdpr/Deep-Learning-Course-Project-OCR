import torch
from torch import nn

#Constants
INPUT_SHAPE = (96, None, 1)
IN_CHANNELS = INPUT_SHAPE[2]
TOTAL_BATCHES = 20
BATCH_SIZE = 32

class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut) # bi-directional LSTM has two hidden states!

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output

class CRNN(nn.Module):
    def __init__(self):
        super(CRNN, self).__init__()
        cnn = nn.Sequential()
        cnn.add_module(f'conv_{0}', nn.Conv2d(1, 32, kernel_size=(5, 5), stride=1))
        cnn.add_module(f'pooling_{0}', nn.MaxPool2d(2, 2))
        cnn.add_module(f'conv_{1}', nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1))
        cnn.add_module(f'pooling_{1}', nn.MaxPool2d(2, 2))
        cnn.add_module(f'conv_{2}', nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1))
        cnn.add_module(f'pooling_{2}', nn.MaxPool2d(2, 2))
        cnn.add_module(f'conv_{3}', nn.Conv2d(128, 256, kernel_size=(3, 1), stride=1))
        cnn.add_module(f'pooling_{3}', nn.MaxPool2d(3, 1))
        cnn.add_module(f'conv_{4}', nn.Conv2d(256, 512, kernel_size=(3, 1), stride=1))
        cnn.add_module(f'conv_{5}', nn.Conv2d(512, 512, kernel_size=(4, 1), stride=1))
        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512,nHidden=512, nOut=256),
            BidirectionalLSTM(nIn=256, nHidden=256, nOut=12)
        )
    def forward(self, input):
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # rnn features
        output = self.rnn(conv)
        return output