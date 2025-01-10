import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.nn import Module, Parameter, Sequential, Linear, ReLU

class Encoder_overall(Module):
      
    """\
    Overall encoder.

    Parameters
    ----------
    dim_in_feat_omics1 : int
        Dimension of input features for omics1.
    dim_in_feat_omics2 : int
        Dimension of input features for omics2. 
    dim_out_feat_omics1 : int
        Dimension of latent representation for omics1.
    dim_out_feat_omics2 : int
        Dimension of latent representation for omics2, which is the same as omics1.
    dropout: int
        Dropout probability of latent representations.
    act: Activation function. By default, we use ReLU.    

    Returns
    -------
    results: a dictionary including representations and modality weights.

    """
     
    def __init__(self, dim_in_feat_omics1, dim_out_feat_omics1, dim_in_feat_omics2, dim_out_feat_omics2, dropout=0.0, act=F.relu):
        super(Encoder_overall, self).__init__()
        self.dim_in_feat_omics1 = dim_in_feat_omics1
        self.dim_in_feat_omics2 = dim_in_feat_omics2
        self.dim_out_feat_omics1 = dim_out_feat_omics1
        self.dim_out_feat_omics2 = dim_out_feat_omics2
        self.dropout = dropout
        self.act = act
        
        self.encoder_omics1 = Encoder(self.dim_in_feat_omics1, self.dim_out_feat_omics1)
        self.decoder_omics1 = Decoder(self.dim_out_feat_omics1, self.dim_in_feat_omics1)
        self.encoder_omics2 = Encoder(self.dim_in_feat_omics2, self.dim_out_feat_omics2)
        self.decoder_omics2 = Decoder(self.dim_out_feat_omics2, self.dim_in_feat_omics2)
        
        self.atten_omics1 = AttentionLayer(self.dim_out_feat_omics1, self.dim_out_feat_omics1)
        self.atten_omics2 = AttentionLayer(self.dim_out_feat_omics2, self.dim_out_feat_omics2)
        self.atten_cross = AttentionLayer(self.dim_out_feat_omics1, self.dim_out_feat_omics2)
        
    def forward(self, features_omics1, features_omics2, adj_spatial_omics1, adj_feature_omics1, adj_spatial_omics2, adj_feature_omics2):
        
        # graph1
        emb_latent_spatial_omics1 = self.encoder_omics1(features_omics1, adj_spatial_omics1)  
        emb_latent_spatial_omics2 = self.encoder_omics2(features_omics2, adj_spatial_omics2)
        
        # graph2
        emb_latent_feature_omics1 = self.encoder_omics1(features_omics1, adj_feature_omics1)
        emb_latent_feature_omics2 = self.encoder_omics2(features_omics2, adj_feature_omics2)
        
        # within-modality attention aggregation layer
        emb_latent_omics1, alpha_omics1 = self.atten_omics1(emb_latent_spatial_omics1, emb_latent_feature_omics1)
        emb_latent_omics2, alpha_omics2 = self.atten_omics2(emb_latent_spatial_omics2, emb_latent_feature_omics2)
        
        # between-modality attention aggregation layer
        emb_latent_combined, alpha_omics_1_2 = self.atten_cross(emb_latent_omics1, emb_latent_omics2)
        #print('emb_latent_combined:', emb_latent_combined)
        
        # reverse the integrated representation back into the original expression space with modality-specific decoder
        emb_recon_omics1 = self.decoder_omics1(emb_latent_combined, adj_spatial_omics1)
        emb_recon_omics2 = self.decoder_omics2(emb_latent_combined, adj_spatial_omics2)
        
        # consistency encoding
        emb_latent_omics1_across_recon = self.encoder_omics2(self.decoder_omics2(emb_latent_omics1, adj_spatial_omics2), adj_spatial_omics2) 
        emb_latent_omics2_across_recon = self.encoder_omics1(self.decoder_omics1(emb_latent_omics2, adj_spatial_omics1), adj_spatial_omics1)
        
        results = {'emb_latent_omics1':emb_latent_omics1,
                   'emb_latent_omics2':emb_latent_omics2,
                   'emb_latent_combined':emb_latent_combined,
                   'emb_recon_omics1':emb_recon_omics1,
                   'emb_recon_omics2':emb_recon_omics2,
                   'emb_latent_omics1_across_recon':emb_latent_omics1_across_recon,
                   'emb_latent_omics2_across_recon':emb_latent_omics2_across_recon,
                   'alpha_omics1':alpha_omics1,
                   'alpha_omics2':alpha_omics2,
                   'alpha':alpha_omics_1_2
                   }
        
        return results





class Encoder(Module):
    """
    Modality-specific GraphSAGE encoder.

    Parameters
    ----------
    in_feat: int
        Dimension of input features.
    out_feat: int
        Dimension of output features.
    dropout: float
        Dropout probability for latent representations.
    act: Activation function. By default, we use ReLU.

    Returns
    -------
    Latent representation.
    """

    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(Encoder, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.dropout = dropout
        self.act = act

        # Learnable weights
        self.weight_self = Parameter(torch.FloatTensor(self.in_feat, self.out_feat))  # Self-loop weights
        self.weight_neigh = Parameter(torch.FloatTensor(self.in_feat, self.out_feat))  # Neighbor weights

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights with Xavier uniform distribution
        torch.nn.init.xavier_uniform_(self.weight_self)
        torch.nn.init.xavier_uniform_(self.weight_neigh)

    def forward(self, feat, adj):
        # Compute neighbor aggregation: mean pooling
        neigh_feat = torch.spmm(adj, feat)  # Sparse matrix multiplication for neighbor aggregation

        # Apply learnable weights to self and neighbor features
        self_feat = torch.mm(feat, self.weight_self)  # Transformation of self features
        neigh_feat = torch.mm(neigh_feat, self.weight_neigh)  # Transformation of aggregated neighbor features

        # Combine self and neighbor features
        out = self_feat + neigh_feat

        # Apply activation function
        out = self.act(out)

        # Apply dropout
        out = F.dropout(out, p=self.dropout, training=self.training)

        return out




class Decoder(Module):
    """
    Modality-specific GraphSAGE encoder.

    Parameters
    ----------
    in_feat: int
        Dimension of input features.
    out_feat: int
        Dimension of output features.
    dropout: float
        Dropout probability for latent representations.
    act: Activation function. By default, we use ReLU.

    Returns
    -------
    Latent representation.
    """

    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(Decoder, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.dropout = dropout
        self.act = act

        # Learnable weights
        self.weight_self = Parameter(torch.FloatTensor(self.in_feat, self.out_feat))  # Self-loop weights
        self.weight_neigh = Parameter(torch.FloatTensor(self.in_feat, self.out_feat))  # Neighbor weights

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights with Xavier uniform distribution
        torch.nn.init.xavier_uniform_(self.weight_self)
        torch.nn.init.xavier_uniform_(self.weight_neigh)

    def forward(self, feat, adj):
        # Compute neighbor aggregation: mean pooling
        neigh_feat = torch.spmm(adj, feat)  # Sparse matrix multiplication for neighbor aggregation

        # Apply learnable weights to self and neighbor features
        self_feat = torch.mm(feat, self.weight_self)  # Transformation of self features
        neigh_feat = torch.mm(neigh_feat, self.weight_neigh)  # Transformation of aggregated neighbor features

        # Combine self and neighbor features
        out = self_feat + neigh_feat

        # Apply activation function
        out = self.act(out)

        # Apply dropout
        out = F.dropout(out, p=self.dropout, training=self.training)

        return out




class AttentionLayer(Module):
    
    """\
    Attention layer.

    Parameters
    ----------
    in_feat: int
        Dimension of input features.
    out_feat: int
        Dimension of output features.     

    Returns
    -------
    Aggregated representations and modality weights.

    """
    
    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(AttentionLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        
        self.w_omega = Parameter(torch.FloatTensor(in_feat, out_feat))
        self.u_omega = Parameter(torch.FloatTensor(out_feat, 1))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.w_omega)
        torch.nn.init.xavier_uniform_(self.u_omega)
        
    def forward(self, emb1, emb2):
        emb = []
        emb.append(torch.unsqueeze(torch.squeeze(emb1), dim=1))
        emb.append(torch.unsqueeze(torch.squeeze(emb2), dim=1))
        self.emb = torch.cat(emb, dim=1)
        
        self.v = F.tanh(torch.matmul(self.emb, self.w_omega))
        self.vu=  torch.matmul(self.v, self.u_omega)
        self.alpha = F.softmax(torch.squeeze(self.vu) + 1e-6)  
        
        emb_combined = torch.matmul(torch.transpose(self.emb,1,2), torch.unsqueeze(self.alpha, -1))
    
        return torch.squeeze(emb_combined), self.alpha      
