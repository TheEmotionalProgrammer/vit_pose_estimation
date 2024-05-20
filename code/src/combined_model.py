import torch
import torch.nn as nn
import timm
from torchvision.models import mobilenet_v2

class Combined_ViT_MobileNet(nn.Module):
    def __init__(self, dropout_rate, n_ori_outputs, n_pos_outputs):
        super(Combined_ViT_MobileNet, self).__init__()

        # Load pretrained MobileNetV2
        self.mobilenet = mobilenet_v2(pretrained=False)
        num_ftrs_mobilenet = self.mobilenet.classifier[1].in_features
        self.mobilenet.classifier = nn.Identity()  # Remove the classifier

        # Load pretrained ViT
        self.vit = timm.create_model('vit_tiny_patch16_384', pretrained=False, num_classes=0)  # No final layer
        num_ftrs_vit = self.vit.embed_dim

        combined_features_dim = num_ftrs_mobilenet + num_ftrs_vit

        # Position branch
        self.pos = nn.Sequential(
            nn.Linear(combined_features_dim, n_pos_outputs)
        )

        # Orientation branch
        self.ori = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(combined_features_dim, n_ori_outputs)
        )

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # Extract features from MobileNetV2
        mobilenet_features = self.mobilenet.features(x)
        mobilenet_features = nn.functional.adaptive_avg_pool2d(mobilenet_features, 1).reshape(mobilenet_features.shape[0], -1)

        # Extract features from ViT
        vit_features = self.vit(x)

        # Concatenate features
        combined_features = torch.cat((mobilenet_features, vit_features), dim=1)

        # Get position and orientation predictions
        pos = self.pos(combined_features)
        ori = self.ori(combined_features)

        return ori, pos

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def copy_state_dict(src_state_dict, dst_state_dict):
    for name, param in src_state_dict.items():
        if name in dst_state_dict:
            if isinstance(param, nn.Parameter):
                param = param.data
            dst_state_dict[name].copy_(param)
    return dst_state_dict

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def import_combined_model(dropout_rate, ori_type, n_ori_bins, pretrained=True):
    n_ori_outputs = 4 if ori_type == 'Regression' else n_ori_bins ** 3
    model = Combined_ViT_MobileNet(dropout_rate, n_ori_outputs, n_pos_outputs=3)

    print(f"Number of parameters: {count_parameters(model)}")

    if pretrained:
        # Load pretrained ViT weights
        vit_model = timm.create_model('vit_tiny_patch16_384', pretrained=True)
        model.vit.load_state_dict(copy_state_dict(vit_model.state_dict(), model.vit.state_dict()))

        # Load pretrained MobileNetV2 weights (already loaded in the constructor)
        mobilenet_model = mobilenet_v2(pretrained=True)
        model.mobilenet.load_state_dict(copy_state_dict(mobilenet_model.state_dict(), model.mobilenet.state_dict()))

    return model
