import torch
import torch.nn as nn
import torch.nn.functional as F

from efficientnet_pytorch import EfficientNet

from .layers import AttBlockV2
from .utils import init_layer, interpolate, pad_framewise_output


class EfficientNetSED(nn.Module):
    def __init__(self, base_model_name: str, pretrained=False,
                 num_classes=264):
        super().__init__()
        self.interpolate_ratio = 30  # Downsampled ratio
        if pretrained:
            self.base_model = EfficientNet.from_pretrained(base_model_name)
        else:
            self.base_model = EfficientNet.from_name(base_model_name)

        in_features = self.base_model._fc.in_features

        self.fc1 = nn.Linear(in_features, in_features, bias=True)
        self.att_block = AttBlockV2(in_features, num_classes, activation="sigmoid")

        self.init_weight()

    def init_weight(self):
        init_layer(self.fc1)

    def forward(self, input):
        frames_num = input.size(3)

        # (batch_size, channels, freq, frames)
        x = self.base_model.extract_features(input)

        # (batch_size, channels, frames)
        x = torch.mean(x, dim=2)

        # channel smoothing
        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2

        x = F.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = x.transpose(1, 2)
        x = F.dropout(x, p=0.5, training=self.training)
        (clipwise_output, norm_att, segmentwise_output) = self.att_block(x)
        logit = torch.sum(norm_att * self.att_block.cla(x), dim=2)
        segmentwise_logit = self.att_block.cla(x).transpose(1, 2)
        segmentwise_output = segmentwise_output.transpose(1, 2)

        # Get framewise output
        framewise_output = interpolate(segmentwise_output,
                                       self.interpolate_ratio)
        framewise_output = pad_framewise_output(framewise_output, frames_num)

        framewise_logit = interpolate(segmentwise_logit, self.interpolate_ratio)
        framewise_logit = pad_framewise_output(framewise_logit, frames_num)

        output_dict = {
            "framewise_output": framewise_output,
            "segmentwise_output": segmentwise_output,
            "logit": logit,
            "framewise_logit": framewise_logit,
            "clipwise_output": clipwise_output
        }

        return output_dict
