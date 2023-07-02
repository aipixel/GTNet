import torch
import torch.nn as nn
import numpy as np
from utils.model_utils import *

proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(proj_dir, "utils/Pointnet2.PyTorch/pointnet2"))
import pointnet2_utils as pn2


class CONV_Res(nn.Module):
    def __init__(self, input_size=10, hide_size=64, output_size=128):
        super(CONV_Res, self).__init__()
        self.conv_shortcut = nn.Conv2d(input_size, output_size, 1)
        
        self.conv1 = nn.Sequential(nn.Conv2d(input_size, hide_size, 1),
                                   nn.LeakyReLU(negative_slope=0.2,inplace=True))
                                   
        self.conv2 = nn.Sequential(nn.Conv2d(hide_size, output_size, 1),
                                   nn.LeakyReLU(negative_slope=0.2,inplace=True),
                                   nn.Conv2d(output_size, output_size, 1))
    def forward(self, x):
        """
        Args:
            x: (B, out_dim, n)
        """
        shortcut = self.conv_shortcut(x)
        out = self.conv2(self.conv1(x)) + shortcut
        
        return out
        
        
class PointNetSetFeat(nn.Module):
    def __init__(self, knn, hide_size, output_size):
        super(PointNetSetFeat, self).__init__()
        self.nsample = knn
        self.conv1 = CONV_Res(10, hide_size, hide_size)
        self.conv2 = CONV_Res(hide_size*2, hide_size, output_size)

    def forward(self, xyz):
        """
        Input:
            xyz: input points position data, [B, 3, N]
        Return:
            new_xyz: sampled points position data, [B, C, N] 
        """
        x = get_knn_feature(xyz, k=self.nsample)
        x = self.conv1(x)
        x1 = torch.max(x, 2, keepdim=True)[0].repeat(1, 1, self.nsample, 1)
        x = torch.cat([x, x1], 1)
        x = self.conv2(x)
        x, _ = torch.max(x, 2)        
        
        return x
        
        
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SA_Layer(nn.Module):
    def __init__(self, channels):
        super(SA_Layer, self).__init__()
        self.q_conv = nn.Linear(channels, channels // 4, bias=False)
        self.k_conv = nn.Linear(channels, channels // 4, bias=False)
        self.v_conv = nn.Linear(channels, channels, bias=False)
        
        self.trans_conv = Mlp(in_features=channels, hidden_features=channels // 4, act_layer=nn.GELU, drop=0.1)

        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(0.1)
                
        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(0.1)
        
        self.drop_path = DropPath(0.1)
        
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)

    def forward(self, x, xyz): 
        identity = x.transpose(1, 2).contiguous() # b, n, c
        
        x = (x + xyz).transpose(1, 2).contiguous() # b, n, c
        x = self.norm1(x)
        
        x_q = self.q_conv(x)  # b, n, c
        x_k = self.k_conv(x).transpose(1, 2).contiguous()  # b, c, n
        x_v = self.v_conv(x) # b, n, c
        
        energy = torch.bmm(x_q, x_k) # b, n, n
        energy = energy / np.sqrt(x_q.size(-1))
        attention = self.softmax(energy) 
        attention = self.attn_drop(attention)

        x_r = torch.bmm(attention, x_v)  # b, n, c
        x_r = self.proj(x_r)
        x_r = self.proj_drop(x_r)
        x_r = identity - x_r
        
        x_r = self.trans_conv(self.norm2(x_r))
        out = identity + self.drop_path(x_r)
        
        return out.transpose(1, 2).contiguous()


class Point_Transformer(nn.Module):
    def __init__(self, channels=128, out_channels=512):
        super(Point_Transformer, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, 1)                                  
        self.conv2 = nn.Conv1d(channels*4, out_channels, 1)
        self.sa1 = SA_Layer(channels)
        self.sa2 = SA_Layer(channels)
        self.sa3 = SA_Layer(channels)
        self.sa4 = SA_Layer(channels)
        self.pos_xyz = nn.Sequential(nn.Conv1d(3, channels, 1),
                                     nn.LeakyReLU(negative_slope=0.2,inplace=True),
                                     nn.Conv1d(channels, channels, 1))

    def forward(self, pos, feats):     
        # B, D, N
        x = self.conv1(feats)
        xyz = self.pos_xyz(pos)
        
        x1 = self.sa1(x, xyz)
        x2 = self.sa2(x1, xyz)
        x3 = self.sa3(x2, xyz)
        x4 = self.sa4(x3, xyz)     
        
        global_feat = torch.cat((x1, x2, x3, x4), dim=1)
        global_feat = self.conv2(global_feat)
        global_feat, _ = torch.max(global_feat, 2)
        
        return global_feat


class Feature_Extraction(nn.Module):
    def __init__(self, knn=32, local_feature_dim=128, global_feature_dim=512):
        super(Feature_Extraction, self).__init__()      
        self.local_encoder = PointNetSetFeat(knn=knn, hide_size=64, output_size=local_feature_dim)    
        self.pt_transformer = Point_Transformer(local_feature_dim, global_feature_dim)
        
    def forward(self, pos):
        f_local = self.local_encoder(pos)
        f_global = self.pt_transformer(pos, f_local)
        
        return f_local, f_global
        
        
class GPool(nn.Module):
    def __init__(self, ratio, feature_dim):
        super(GPool, self).__init__()
        self.ratio = ratio               
        self.Score = nn.Sequential(nn.Conv1d(feature_dim, feature_dim // 4, 1, bias=True),
                                   nn.ReLU(inplace=True),
                                   nn.Conv1d(feature_dim // 4, 1, 1, bias=False),
                                   nn.Sigmoid())   
    def forward(self, pos, feat, x):       
        batchsize = x.size(0)
        if self.ratio < 1:
            k = int(x.size(2) * self.ratio)
        else:
            k = self.ratio
        # pos       : B * 3 * N
        # x         : B * Fin * N       
        score = (self.Score(x)).squeeze(1)  # B * N
        top_idx = torch.argsort(score, dim=1, descending=True)[:, 0:k]  # B * k 
        score_k = torch.gather(score, dim=1, index=top_idx)  # B * k

        pos_k = torch.gather(pos, dim=-1, index=top_idx.unsqueeze(1).expand(batchsize, 3, k))
        loacl_feat_k = torch.gather(feat, dim=-1, index=top_idx.unsqueeze(1).expand(batchsize, feat.size(1), k))
        feat_k = torch.gather(x, dim=-1, index=top_idx.unsqueeze(1).expand(batchsize, x.size(1), k))
        feat_k = feat_k * score_k.unsqueeze(1).expand_as(feat_k)   

        return pos_k, loacl_feat_k, feat_k


class MLP_Res(nn.Module):
    def __init__(self, input_size=10, hide_size=64, output_size=128):
        super(MLP_Res, self).__init__()
 
        self.conv_shortcut = nn.Conv1d(input_size, output_size, 1)

        self.conv1 = nn.Sequential(nn.Conv1d(input_size, hide_size, 1),
                                   nn.LeakyReLU(negative_slope=0.2,inplace=True))

        self.conv2 = nn.Sequential(nn.Conv1d(hide_size, output_size, 1),
                                   nn.LeakyReLU(negative_slope=0.2,inplace=True),
                                   nn.Conv1d(output_size, output_size, 1))
    def forward(self, x):
        """
        Args:
            x: (B, out_dim, n)
        """
        shortcut = self.conv_shortcut(x)
        out = self.conv2(self.conv1(x)) + shortcut
        return out
        
   
class Pos_Displace(nn.Module):
    def __init__(self, feature_dim=1024):
        super(Pos_Displace, self).__init__()
        self.conv1 = MLP_Res(3, 64, 128)
        self.conv2 = MLP_Res(feature_dim+3, 128, 64)
        self.conv3 = nn.Conv1d(64, 3, 1)
        self.act = nn.LeakyReLU(negative_slope=0.2,inplace=True)
    def forward(self, pos, x):
        feat = self.conv1(pos)
        feat = torch.max(feat, 2, keepdim=True)[0].repeat((1, 1, x.size(2)))
        feat = torch.cat([pos, feat, x], 1)
        feat = self.act(self.conv2(feat))
        feat = self.conv3(feat)
        
        return feat


class PG(nn.Module):
    def __init__(self, local_feature_dim=128, global_feature_dim=512, ratio=0.5, pre_filter=True):
        super(PG, self).__init__()
        self.pre_filter = pre_filter
        self.score = GPool(ratio, local_feature_dim+global_feature_dim)  
        self.denoise = Pos_Displace(local_feature_dim*2+global_feature_dim)
        
    def forward(self, pos, f_local, global_local_feats):
        """
        :param  pos:    (B, 3, N)
        :param  x:      (B, d, N)
        :return (B, rN, d)
        """ 
        pos_k, f_local_k, global_local_feats_k = self.score(pos, f_local, global_local_feats)                      
         
        if self.pre_filter:
            pos_k = pos_k + self.denoise(pos_k, global_local_feats_k)
       
        return pos_k, f_local_k, global_local_feats_k


class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = MLP_Res(3, 64, 128)
        self.conv2 = MLP_Res(channel+3, 128, 128)

        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 9)
        self.act = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, pos, x):
        x1 = self.conv1(pos)
        x1 = torch.max(x1, 2, keepdim=True)[0].repeat(1, 1, x.size(2))
        x = torch.cat([pos, x1, x], 1)
        x = self.conv2(x)
        x = torch.max(x, 2)[0]
        x = x.view(-1, 128)

        x = self.act(self.fc2(x))
        x = self.fc3(x)

        x = x.view(-1, 3, 3)
        return x


class PointMisFeat(nn.Module):
    def __init__(self, knn, hide_size, output_size):
        super(PointMisFeat, self).__init__()
        self.nsample = knn
        self.conv1 = CONV_Res(10, hide_size, hide_size)
        self.conv2 = CONV_Res(hide_size*2, hide_size, output_size)

    def forward(self, xyz, mis_pos):
        """
        Input:
            xyz: input points position data, [B, 3, N]
        Return:
            new_xyz: sampled points position data, [B, C, N] 
        """
        #x = get_knn_feature(xyz, k=self.nsample)
        x = get_knn_feature_mis(xyz, mis_pos, k=self.nsample)
        x = self.conv1(x)
        x1 = torch.max(x, 2, keepdim=True)[0].repeat(1, 1, self.nsample, 1)
        x = torch.cat([x, x1], 1)
        x = self.conv2(x)
        x, _ = torch.max(x, 2)
        #print("*************************x.shape*************************",x.shape)                  
        return x

class STN(nn.Module):
    def __init__(self, local_feature_dim=128, global_feature_dim=512, post_poinet=True, post_filter=True):
        super(STN, self).__init__()
        self.post_poinet = post_poinet
        self.post_filter = post_filter
        self.mlp1 = nn.Sequential(nn.Linear(global_feature_dim, local_feature_dim),
                                 nn.LeakyReLU(negative_slope=0.2),
                                 nn.Linear(local_feature_dim, local_feature_dim)) 
        self.mlp2 = PointMisFeat(knn=32, hide_size=64, output_size=128) 
        self.pos_offset = Pos_Displace(local_feature_dim*2+global_feature_dim)
        self.stn3d = STN3d(local_feature_dim*2+global_feature_dim)
        
    def forward(self, pos, pos_k, local_feat_k, global_local_feats_k, f_global):
        """
        :param  pos:    (B, 3, N)
        :param  x:      (B, d, N)
        :return (B, rN, d)
        """ 
        trans3d = self.stn3d(pos_k, global_local_feats_k)      
        mis_pos = pos_k.transpose(1, 2).contiguous()
        mis_pos = torch.bmm(mis_pos, trans3d)
        mis_pos = mis_pos.transpose(1, 2).contiguous()
        
        if self.post_poinet:
            mis_feat_k = self.mlp2(pos, mis_pos)
        else:
            global_feat_k = self.mlp1(f_global)
            mis_feat_k = global_feat_k.unsqueeze(-1).repeat(1, 1, local_feat_k.size(-1)) - local_feat_k
        
        global_feats_k = f_global.unsqueeze(-1).repeat(1, 1, mis_feat_k.size(-1))
        feats = torch.cat([global_feats_k, mis_feat_k], dim=1)
        
        if self.post_filter:
            mis_pos = mis_pos + self.pos_offset(mis_pos, feats)
        
        return mis_pos, mis_feat_k


class GTNet_decoder(nn.Module):
    def __init__(self, local_feature_dim=128, global_feature_dim=512, ratio=0.5, up_ratio=8):
        super(GTNet_decoder, self).__init__()
        self.n_primitives = up_ratio
        self.patch_generator = nn.ModuleList(
            [
                PG(local_feature_dim, global_feature_dim, ratio)
                for i in range(self.n_primitives)
            ])
        self.structure_transformation = nn.ModuleList(
            [
                STN(local_feature_dim, global_feature_dim)
                for i in range(self.n_primitives)
            ])
        
    def forward(self, pos, f_local, f_global):
        """
        :param  pos:    (B, 3, N)
        :param  x:      (B, d, N)
        :return (B, 3, rN)
        """
        batch_size, _, n_pts = pos.size()
        global_feats = f_global.unsqueeze(-1).repeat(1, 1, n_pts) 
        global_local_feats = torch.cat((global_feats, f_local), dim=1)  
        
        coarse = []
        coarse_features = []     
        for i in range(self.n_primitives):
            pos_k, f_local_k, global_local_feats_k = self.patch_generator[i](pos, f_local, global_local_feats) 
            mis_pos_k, mis_feat_k = self.structure_transformation[i](pos, pos_k, f_local_k, global_local_feats_k, f_global)

            patch_pos = torch.cat((pos_k, mis_pos_k), dim=-1)
            patch_feat = torch.cat((f_local_k, mis_feat_k), dim=-1) 
            
            coarse.append(patch_pos)
            coarse_features.append(patch_feat)  
            
        coarse = torch.cat(coarse, 2).contiguous()
        coarse_features = torch.cat(coarse_features, 2).contiguous()
       
        return coarse, coarse_features


class FoldingNet(nn.Module):
    def __init__(self, local_feature_dim=128, global_feature_dim=512):
        super(FoldingNet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(3, 128, 1),
                                   nn.LeakyReLU(negative_slope=0.2,inplace=True),
                                   nn.Conv1d(128, 128, 1))
                                   
        self.conv2 = nn.Sequential(nn.Conv1d(global_feature_dim+local_feature_dim*2+5, 128, 1),
                                   nn.LeakyReLU(negative_slope=0.2,inplace=True),
                                   nn.Conv1d(128, 32, 1),
                                   nn.LeakyReLU(negative_slope=0.2,inplace=True),
                                   nn.Conv1d(32, 3, 1))
        
    def forward(self, pos, feat, global_feat, grid_feat):     
        x = self.conv1(pos)
        x = torch.max(x, 2, keepdim=True)[0].repeat((1, 1, feat.size(2)))
        x = torch.cat([pos, x, feat, global_feat, grid_feat], 1)
        x = self.conv2(x)
        fine = x + pos
   
        return fine


class DRN(nn.Module):
    def __init__(self, npoints=2048, local_feature_dim=128, global_feature_dim=512):
        super(DRN, self).__init__()
        self.num_points = npoints
        self.grid = gen_grid_up(self.num_points, 0.2).cuda().contiguous()
        self.foldingnet = FoldingNet(local_feature_dim, global_feature_dim)
        
    def forward(self, coarse, coarse_features, f_global):
        """
        :param  pos:    (B, 3, N)
        :param  x:      (B, d, N)
        :return (B, 3, rN)
        """  
        batchsize, _, n_pts = coarse.size()
        global_feats = f_global.unsqueeze(-1).repeat(1, 1, self.num_points).contiguous()        
        if n_pts > self.num_points:
            idx_fps = pn2.furthest_point_sample(coarse.transpose(1, 2).contiguous(), self.num_points)
            coarse = pn2.gather_operation(coarse, idx_fps)
            coarse_features = pn2.gather_operation(coarse_features, idx_fps) 
            
        grid = self.grid.clone().detach()
        grid_feats = grid.unsqueeze(0).repeat(batchsize, 1, 1).contiguous().cuda()
        
        fine = self.foldingnet(coarse, coarse_features, global_feats, grid_feats)

        return fine
        
        
class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.knn = args.knn
        self.train_loss = args.loss
        self.num_points = args.num_points
        self.local_feature_dim = args.local_feature_dim
        self.global_feature_dim = args.global_feature_dim
        self.sample_ratio = args.sample_ratio
        self.up_ratio = args.up_ratio        
        self.encoder = Feature_Extraction(knn=self.knn, local_feature_dim=self.local_feature_dim, 
                                          global_feature_dim=self.global_feature_dim)     
        self.decoder = GTNet_decoder(local_feature_dim=self.local_feature_dim, global_feature_dim=self.global_feature_dim, 
                                     ratio=self.sample_ratio, up_ratio=self.up_ratio)
        self.detail_refinement = DRN(npoints=self.num_points, local_feature_dim=self.local_feature_dim, 
                                     global_feature_dim=self.global_feature_dim)
           
    def forward(self, pos, gt, is_training=True, alpha=None):
        f_local, f_global = self.encoder(pos)
        out1, coarse_features = self.decoder(pos, f_local, f_global)
        out2 = self.detail_refinement(out1, coarse_features, f_global)
             
        out1 = out1.transpose(1, 2).contiguous()
        out2 = out2.transpose(1, 2).contiguous()
        print(out1.shape,out2.shape)

        if is_training:
            if self.train_loss == 'emd':
                loss1 = calc_emd(out1, gt)
                loss2 = calc_emd(out2, gt)
            elif self.train_loss == 'cd':
                loss1, _ = calc_cd(out1, gt)
                loss2, _ = calc_cd(out2, gt)
            else:
                raise NotImplementedError('Train loss is either CD or EMD!')
            
            total_train_loss = (loss1.mean() + loss2.mean() * alpha) * 1e3
            return out2, loss2, total_train_loss
        else:
            emd = calc_emd(out2, gt, eps=0.004, iterations=3000)
            cd_p, cd_t, f1 = calc_cd(out2, gt, calc_f1=True)
            return {'out1': out1, 'out2': out2, 'emd': emd, 'cd_p': cd_p, 'cd_t': cd_t, 'f1': f1}