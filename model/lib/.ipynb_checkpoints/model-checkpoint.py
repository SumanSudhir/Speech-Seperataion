from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np

class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()

    def forward(self, x):
        # Do your print / debug stuff here
        print(x.shape)
        return x


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)
    

class AV_model(nn.Module):

    def __init__(self, number_of_people=2):
        super(AV_model, self).__init__()

        # --------------------------- Audio network ---------------------------

        # Audio_net network
        # Input is audio spectogram of size (298,257,2)
        # define the layers as described in the paper
        self.number_of_people = number_of_people
        self.Audio_net = nn.Sequential(
            # conv1
            nn.Conv2d(in_channels=2, out_channels=96, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3),
                      dilation=(1, 1)),
            nn.BatchNorm2d(num_features=96),
            nn.ReLU(),
            # PrintLayer(),
            # conv2
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0),
                      dilation=(1, 1)),
            nn.BatchNorm2d(num_features=96),
            nn.ReLU(),
            # PrintLayer(),
            # conv3
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2),
                      dilation=(1, 1)),
            nn.BatchNorm2d(num_features=96),
            nn.ReLU(),
            # PrintLayer(),
            # conv4
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(5, 5), stride=(1, 1), padding=(4, 2),
                      dilation=(2, 1)),
            nn.BatchNorm2d(num_features=96),
            nn.ReLU(),
            # PrintLayer(),
            # conv5
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(5, 5), stride=(1, 1), padding=(8, 2),
                      dilation=(4, 1)),
            nn.BatchNorm2d(num_features=96),
            nn.ReLU(),
            # PrintLayer(),
            # conv6
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(5, 5), stride=(1, 1), padding=(16, 2),
                      dilation=(8, 1)),
            nn.BatchNorm2d(num_features=96),
            nn.ReLU(),
            # PrintLayer(),
            # conv7
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(5, 5), stride=(1, 1), padding=(32, 2),
                      dilation=(16, 1)),
            nn.BatchNorm2d(num_features=96),
            nn.ReLU(),
            # PrintLayer(),
            # conv8
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(5, 5), stride=(1, 1), padding=(64, 2),
                      dilation=(32, 1)),
            nn.BatchNorm2d(num_features=96),
            nn.ReLU(),
            # PrintLayer(),
            # conv9
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2),
                      dilation=(1, 1)),
            nn.BatchNorm2d(num_features=96),
            nn.ReLU(),
            # PrintLayer(),
            # conv10
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(5, 5), stride=(1, 1), padding=(4, 4),
                      dilation=(2, 2)),
            nn.BatchNorm2d(num_features=96),
            nn.ReLU(),
            # PrintLayer(),
            # conv11
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(5, 5), stride=(1, 1), padding=(8, 8),
                      dilation=(4, 4)),
            nn.BatchNorm2d(num_features=96),
            nn.ReLU(),
            # PrintLayer(),
            # conv12
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(5, 5), stride=(1, 1), padding=(16, 16),
                      dilation=(8, 8)),
            nn.BatchNorm2d(num_features=96),
            nn.ReLU(),
            # PrintLayer(),
            # conv13
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(5, 5), stride=(1, 1), padding=(32, 32),
                      dilation=(16, 16)),
            nn.BatchNorm2d(num_features=96),
            nn.ReLU(),
            # PrintLayer(),
            # conv14
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(5, 5), stride=(1, 1), padding=(64, 64),
                      dilation=(32, 32)),
            nn.BatchNorm2d(num_features=96),
            nn.ReLU(),
            # PrintLayer(),
            # conv15
            nn.Conv2d(in_channels=96, out_channels=8, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0),
                      dilation=(1, 1)),
            nn.BatchNorm2d(num_features=8),
            nn.ReLU(),
            # PrintLayer(),
            Reshape(-1, 298, 257 * 8),
            # PrintLayer(),
        )

        # --------------------------- Video network ---------------------------
        # Video_net network
        # Input is video spectogram of size (1792,75,1,)
        # define the layers as described in the paper

        self.Video_net = nn.Sequential(
            # conv1
            nn.Conv2d(in_channels=1792, out_channels=256, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0),
                      dilation=(1, 1)),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            # PrintLayer(),
            # conv2
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0),
                      dilation=(1, 1)),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            # PrintLayer(),
            # conv3
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(5, 1), stride=(1, 1), padding=(4, 0),
                      dilation=(2, 1)),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            # PrintLayer(),
            # conv4
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(5, 1), stride=(1, 1), padding=(8, 0),
                      dilation=(4, 1)),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            # PrintLayer(),
            # conv5
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(5, 1), stride=(1, 1), padding=(16, 0),
                      dilation=(8, 1)),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            # PrintLayer(),
            # conv6
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(5, 1), stride=(1, 1), padding=(32, 0),
                      dilation=(16, 1)),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            # PrintLayer(),
            # upsample
            Reshape(-1, 1, 75, 256),
            nn.Upsample(size=(298, 256), mode='bilinear', align_corners=False),
            Reshape(-1, 298, 256),
            # PrintLayer()
        )

        # --------------------------- fusion network ---------------------------
        # Fusion_net network
        # Av fused input is  of size (298,8*257 + number_of_people*256)
        # define the layers as described in the paper

        self.lstm = nn.LSTM(input_size=256 * self.number_of_people + 8 * 257, hidden_size=400, bidirectional=True, batch_first=True)
        
#         self.lstm_AO = nn.LSTM(input_size=256 * 0 + 8 * 257, hidden_size=400, bidirectional=True, batch_first=True)
#         self.lstm_AO_final = nn.LSTM(input_size=8 * 257, hidden_size=self.number_of_people*257*2, bidirectional=True, batch_first=True)
#         self.reshape_AO_final = Reshape(-1, 298, 257, 2, self.number_of_people)
        
#         self.lstm_final = nn.LSTM(input_size=256 * self.number_of_people + 8 * 257, hidden_size=self.number_of_people*257*2, bidirectional=True, batch_first=True)
#         self.reshape_final = Reshape(-1, 298, 257, 2, self.number_of_people)
        
#         self.Fusion_net = nn.Sequential(
#             #FC Layers
#             # PrintLayer(),
#             nn.Linear(in_features=400, out_features=600),
#             # PrintLayer(),
#             nn.BatchNorm1d(num_features=298),
#             # batchnorm num_features here is basically Temporal Batch Normalization. Read
#             # https://pytorch.org/docs/stable/nn.html#batchnorm1d for more details
#             nn.ReLU(),
#             # PrintLayer(),
#             nn.Linear(in_features=600, out_features=600),
#             nn.BatchNorm1d(num_features=298),
#             nn.ReLU(),
#             # PrintLayer(),
#             nn.Linear(in_features=600, out_features=600),
#             nn.BatchNorm1d(num_features=298),
#             nn.ReLU(),
#             # PrintLayer(),
#             nn.Linear(in_features=600, out_features=self.number_of_people*257*2),
#             nn.BatchNorm1d(num_features=298),
#             nn.ReLU(),
#             # PrintLayer(),
#             Reshape(-1, 298, 257, 2, self.number_of_people),
#             # PrintLayer()
#             )
        
        self.Fusion_net = nn.Sequential(
            #FC Layers
            # PrintLayer(),
            nn.Linear(in_features=400, out_features=600),
            # PrintLayer(),
            nn.BatchNorm1d(num_features=600),
            nn.ReLU(),
            # PrintLayer(),
#             nn.Linear(in_features=600, out_features=600),
#             nn.BatchNorm1d(num_features=600),
#             nn.ReLU(),
            # PrintLayer(),
#             nn.Linear(in_features=600, out_features=600),
#             nn.BatchNorm1d(num_features=600),
#             nn.ReLU(),
            # PrintLayer(),
            nn.Linear(in_features=600, out_features=self.number_of_people*257*2),
            nn.BatchNorm1d(num_features=self.number_of_people*257*2),
            # PrintLayer(),
            nn.Sigmoid(),
            Reshape(-1, 298, 257, 2, self.number_of_people),
            # PrintLayer()
            )


    def forward(self, audio_input, video_input):
        """

        :param audio_input: shape = (batch, 2, 298, 257)
        :param video_input: shape = (batch, 1792, 75, 1, number_of_people)
        :return:
        """

        AS_out = self.Audio_net(audio_input)
        AVfusion = AS_out
#         print('AS_out',np.unique(AVfusion.cpu().detach().numpy()),AVfusion.shape)
#         print('AVfusion:', AVfusion.shape)
        # AVfusion = torch.randn(2, 298, 257 * 8)
        audio_input_mask = audio_input
        for i in range(self.number_of_people):
            single_person_frames = video_input[:, :, :, :, i]
            single_person_frames_out = self.Video_net(single_person_frames)
            AVfusion = torch.cat((AVfusion, single_person_frames_out), dim=2)
            if i < self.number_of_people - 1:
                audio_input_mask = torch.cat((audio_input_mask, audio_input), dim=3)

        audio_input_mask = audio_input_mask.view(-1, 298, 257, 2, self.number_of_people)
#         print('video',np.unique(AVfusion.cpu().detach().numpy()),AVfusion.shape)
        AVfusion, _ = self.lstm(AVfusion)
#         AVfusion, _ = self.lstm_AO(AVfusion)
    
#         AVfusion, _ = self.lstm_AO_final(AVfusion)
#         print('lstm',np.unique(AVfusion.cpu().detach().numpy()),AVfusion.shape)
        AVfusion = AVfusion.view(AVfusion.shape[0], AVfusion.shape[1], 2, int(AVfusion.shape[2] / 2))
        AVfusion = AVfusion[:, :, 0, :] + AVfusion[:, :, 1, :]
#         AVfusion = self.reshape_AO_final(AVfusion)
#         print('lstm bidirec',np.unique(AVfusion.cpu().detach().numpy()),AVfusion.shape)
        
        AVfusion = AVfusion.contiguous().view(AVfusion.shape[0]*298, 400)
        AVfusion = self.Fusion_net(AVfusion)
    
#         AVfusion, _ = self.lstm_final(AVfusion)
#         print('lstm final',np.unique(AVfusion.cpu().detach().numpy()),AVfusion.shape)
#         AVfusion = AVfusion.view(AVfusion.shape[0], AVfusion.shape[1], 2, int(AVfusion.shape[2] / 2))
#         AVfusion = AVfusion[:, :, 0, :] + AVfusion[:, :, 1, :]
#         print('lstm final bider',np.unique(AVfusion.cpu().detach().numpy()),AVfusion.shape)
#         AVfusion = self.reshape_final(AVfusion)
#         print('reshape_final',np.unique(AVfusion.cpu().detach().numpy()),AVfusion.shape)
        
#         print('fusion',np.unique(AVfusion.cpu().detach().numpy()),AVfusion.shape)
#         print('audio input',np.unique(audio_input_mask.cpu().detach().numpy()),AVfusion.shape)
        AVfusion = AVfusion * audio_input_mask
#         print('mask',np.unique(AVfusion.cpu().detach().numpy()),AVfusion.shape)
        return AVfusion
        
# Net = AV_model(number_of_people=2)
# audio_input = torch.randn(2, 2, 298, 257)
# video_input = torch.randn(2, 1792, 75, 1, 2)
# # print(Net)
# Net(audio_input, video_input)
