import torch
import torch.nn as nn

class HomoscedasticUncertaintyLoss(nn.Module):

    def __init__(self):
        super().__init__()
        (
            self.reconstruction_loss_weight,
            self.distillation_loss_weight,
            self.feature_loss_weight
        ) = (
            nn.Parameter(torch.randn(()), requires_grad = True),
            nn.Parameter(torch.randn(()), requires_grad = True),
            nn.Parameter(torch.randn(()), requires_grad = True),
        )

    def forward(self, reconstruction_loss, distillation_loss, feature_loss):
        return (
            torch.exp(-self.reconstruction_loss_weight) * reconstruction_loss + 
            torch.exp(-self.distillation_loss_weight) * distillation_loss + 
            torch.exp(-self.feature_loss_weight) * feature_loss
        ) + self.reconstruction_loss_weight + self.distillation_loss_weight + self.feature_loss_weight