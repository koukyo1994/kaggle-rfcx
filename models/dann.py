import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import init_layer, interpolate, pad_framewise_output


class TimmEfficientNetSEDMax(nn.Module):
    def __init__(self, base_model_name: str, pretrained=False,
                 num_classes=264):
        super().__init__()
        self.interpolate_ratio = 30  # Downsampled ratio
        self.base_model = timm.create_model(base_model_name, pretrained=pretrained)

        in_features = self.base_model.classifier.in_features

        modules = list(self.base_model.children())
        self.base_model = nn.Sequential(*modules[:-2])

        self.fc1 = nn.Linear(in_features, in_features, bias=True)
        self.fc_audioset = nn.Linear(in_features, num_classes, bias=True)

        self.domain_classifier = nn.Sequential(
            nn.Linear(in_features, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )

        self.init_weight()

    def init_weight(self):
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, input):
        frames_num = input.size(3)

        # (batch_size, channels, freq, frames)
        x = self.base_model(input)

        # (batch_size, channels, frames)
        x = torch.mean(x, dim=2)

        # channel smoothing
        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2

        x = F.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        segmentwise_logit = self.fc_audioset(x)
        (clipwise_logit, _) = torch.max(segmentwise_logit, dim=1)

        segmentwise_output = torch.sigmoid(segmentwise_logit)
        clipwise_output = torch.sigmoid(clipwise_logit)

        interpolate_ratio = frames_num // segmentwise_output.size(1)

        # Get framewise output
        framewise_output = interpolate(segmentwise_output,
                                       interpolate_ratio)
        framewise_output = pad_framewise_output(framewise_output, frames_num)

        framewise_logit = interpolate(segmentwise_logit, interpolate_ratio)
        framewise_logit = pad_framewise_output(framewise_logit, frames_num)

        output_dict = {
            "framewise_output": framewise_output,
            "segmentwise_output": segmentwise_output,
            "logit": clipwise_logit,
            "framewise_logit": framewise_logit,
            "clipwise_output": clipwise_output
        }

        return output_dict
