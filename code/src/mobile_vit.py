
import torch
import torch.nn as nn
import timm

class Mobile_Vit(nn.Module):
    def __init__(self, dropout_rate, n_ori_outputs, n_pos_outputs):
        super(Mobile_Vit, self).__init__()

        self.features = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=0)  # No final layer

        self.ori_head = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.features.embed_dim, n_ori_outputs)
        )
        self.pos_head = nn.Linear(self.features.embed_dim, n_pos_outputs)

    def forward(self, x):

        x = self.features(x)

        ori = self.ori_head(x)
        pos = self.pos_head(x)

        return ori, pos

def copy_state_dict(src_state_dict, dst_state_dict):
    for name, param in src_state_dict.items():
        if name in dst_state_dict:
            if isinstance(param, nn.Parameter):
                param = param.data
            dst_state_dict[name].copy_(param)
    return dst_state_dict

def import_my_vit_ursonet(dropout_rate, ori_type, n_ori_bins, pretrained=True):
    n_ori_outputs = 4 if ori_type == 'Regression' else n_ori_bins ** 3
    model = Mobile_Vit(dropout_rate, n_ori_outputs, n_pos_outputs=3)

    if pretrained:
        vit_model = timm.create_model('vit_base_patch16_224', pretrained=True)
        model.features.load_state_dict(copy_state_dict(vit_model.state_dict(), model.features.state_dict()))

    return model

