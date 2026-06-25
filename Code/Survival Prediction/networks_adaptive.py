import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from utils import init_net

def define_optimizer(opt, model):
    return torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), weight_decay=opt.weight_decay)

def define_scheduler(opt, optimizer):
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            return 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    return None

def define_reg(opt, model):
    if opt.reg_type == 'none': return 0
    reg_loss = sum(torch.norm(param, 1) for param in model.parameters() if param.requires_grad)
    return 1e-4 * reg_loss

def define_net(opt, k):
    net = MultimodalAdaptiveFusion(opt=opt)
    return init_net(net, opt.init_type, opt.init_gain, opt.gpu_ids)

def adaptive_masked_mean(data, masks, weights, eps=1e-8):
    # Data: tuple of (B, H), Masks: tuple of (B,), Weights: (4,)
    weighted_data = sum(X * mask[:, None].float() * weights[i] for i, (X, mask) in enumerate(zip(data, masks)))
    weighted_counts = sum(mask.float() * weights[i] for i, mask in enumerate(masks))[:, None]
    return weighted_data / torch.clamp(weighted_counts, min=eps)

class MultimodalAdaptiveFusion(nn.Module):
    def __init__(self, opt):
        super(MultimodalAdaptiveFusion, self).__init__()
        self.opt = opt
        h = opt.mmhid

        # Modality-specific Encoders (Aligned with Baseline Dimensions)
        self.enc_path = nn.Sequential(nn.Linear(opt.path_dim, h), nn.ReLU(), nn.Linear(h, h))
        self.enc_rad  = nn.Sequential(nn.Linear(opt.rad_dim, h), nn.ReLU(), nn.Linear(h, h))
        self.enc_demo = nn.Sequential(nn.Linear(opt.demo_dim, h), nn.ReLU(), nn.Linear(h, h))
        self.enc_omic = nn.Sequential(nn.Linear(opt.omic_dim, h), nn.ReLU(), nn.Linear(h, h))

        # Shared Reconstruction Head (Simplifies Structure)
        if opt.recon:
            self.recon_head = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, h))

        # Fusion & Classification
        self.fuse_fc = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Dropout(0.1))
        self.classifier = nn.Sequential(nn.Linear(h, opt.label_dim))
        
        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)

    def forward(self, x_path, x_rad, x_demo, x_omic, x_radiomics, x_masks, x_keep_masks, modality_weights=None):
        # Encode
        h_p = self.enc_path(x_path.view(x_path.shape[0], -1))
        h_r = self.enc_rad(x_rad.view(x_rad.shape[0], -1))
        h_d = self.enc_demo(x_demo.view(x_demo.shape[0], -1))
        h_o = self.enc_omic(x_omic.view(x_omic.shape[0], -1))

        # Weighted Adaptive Fusion
        if modality_weights is None:
            modality_weights = torch.ones(4, device=h_p.device) / 4.0
            
        fused = adaptive_masked_mean((h_p, h_r, h_d, h_o), 
                                     (x_keep_masks[:,0], x_keep_masks[:,1], x_keep_masks[:,2], x_keep_masks[:,3]), 
                                     weights=modality_weights)

        # Reconstruction (Using Mean Squared Error on fused representation)
        recon_loss = torch.tensor(0.0, device=fused.device)
        if self.opt.recon:
            recon_loss = torch.mean((self.recon_head(fused) - fused)**2)

        hazard = torch.sigmoid(self.classifier(self.fuse_fc(fused))) * self.output_range + self.output_shift
        return {"recon_loss": recon_loss}, hazard