import torch
import torch.nn as nn
from neural_fingerprint import NeuralFingerprint
from drug_gene_attention import DrugGeneAttention
from ltr_loss import point_wise_mse, list_wise_listnet, list_wise_listmle, pair_wise_ranknet, list_wise_rankcosine, \
    list_wise_ndcg, combine_loss, mse_plus_homophily, class_combine_loss
import pdb
import math

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """

    def __init__(self, sigma=0.1, is_relative_detach=True, device = 'cpu'):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.noise = torch.tensor(0).to(device)

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.std().detach() if self.is_relative_detach else self.sigma * x.std()
            sampled_noise = self.noise.repeat(*x.size()).float().normal_() * scale
            x = x + sampled_noise
        return x 

class DeepCESub(nn.Module):

    """
    submodel of the deepce
    """

    def __init__(self, drug_input_dim, drug_emb_dim, conv_size, degree, gene_input_dim, gene_emb_dim, num_gene,
                 hid_dim, dropout, loss_type, device, initializer=None, pert_type_input_dim=None,
                 cell_id_input_dim=None, pert_idose_input_dim=None,
                 pert_type_emb_dim=None, cell_id_emb_dim=None, pert_idose_emb_dim=None, use_pert_type=False,
                 use_cell_id=False, use_pert_idose=False):
        super(DeepCESub, self).__init__()
        assert drug_emb_dim == gene_emb_dim, 'Embedding size mismatch'
        self.use_pert_type = use_pert_type
        self.use_cell_id = use_cell_id
        self.use_pert_idose = use_pert_idose
        self.drug_emb_dim = drug_emb_dim
        self.gene_emb_dim = gene_emb_dim
        self.drug_fp = NeuralFingerprint(drug_input_dim['atom'], drug_input_dim['bond'], conv_size, drug_emb_dim,
                                         degree, device)
        self.gene_embed = nn.Linear(gene_input_dim, gene_emb_dim)
        self.drug_gene_attn = DrugGeneAttention(gene_emb_dim, gene_emb_dim, n_layers=2, n_heads=4, pf_dim=512,
                                                dropout=dropout, device=device)
        self.linear_dim = self.drug_emb_dim + self.gene_emb_dim
        if self.use_pert_type:
            self.pert_type_embed = nn.Linear(pert_type_input_dim, pert_type_emb_dim)
            self.linear_dim += pert_type_emb_dim
        if self.use_cell_id:
            #self.cell_id_2 = nn.Linear(100, cell_id_emb_dim)
            self.cell_id_1 = nn.Linear(cell_id_input_dim, 200)
            self.cell_id_2 = nn.Linear(200, 100)
            self.cell_id_3 = nn.Linear(100, 50)
            self.cell_id_embed_linear_only = nn.Sequential(self.cell_id_1, nn.ReLU(),
                                                           self.cell_id_2, nn.ReLU(),
                                                           self.cell_id_3, nn.ReLU())
            
            self.cell_id_embed = nn.Sequential(nn.Linear(cell_id_input_dim, 200), nn.Linear(200, 50))
            self.trans_cell_embed_dim = 32
            # self.cell_id_embed_1 = nn.Linear(1, self.trans_cell_embed_dim)
            self.cell_id_transformer = nn.Transformer(d_model = self.trans_cell_embed_dim, nhead = 4,
                                                    num_encoder_layers = 1, num_decoder_layers = 1,
                                                    dim_feedforward = self.trans_cell_embed_dim * 4)

            self.expand_to_num_gene = nn.Linear(50, 978)
            self.pos_encoder = PositionalEncoding(self.trans_cell_embed_dim)
            self.linear_dim += 50
        if self.use_pert_idose:
            self.pert_idose_embed = nn.Linear(pert_idose_input_dim, pert_idose_emb_dim)
            self.linear_dim += pert_idose_emb_dim
        self.linear_1 = nn.Linear(self.linear_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.num_gene = num_gene
        self.loss_type = loss_type
        self.initializer = initializer
        self.device = device
        self.init_weights()

    def init_weights(self):
        print('Initialized deepce sub\'s weight............')
        if self.initializer is None:
            return
        for name, parameter in self.named_parameters():
            # if 'drug_gene_attn' not in name:
            #     if parameter.dim() == 1:
            #         nn.init.constant_(parameter, 10**-7)
            #     else:
            #         self.initializer(parameter)
            if 'drug_gene_attn' not in name:
                if parameter.dim() == 1:
                    nn.init.constant_(parameter, 10**-7)
                else:
                    self.initializer(parameter)

    def forward(self, input_drug, input_gene, mask, input_pert_type, input_cell_id, input_pert_idose, epoch = 0, linear_only=False):
        # input_drug = {'molecules': molecules, 'atom': node_repr, 'bond': edge_repr}
        # gene_embed = [num_gene * gene_emb_dim]
        num_batch = input_drug['molecules'].batch_size
        drug_atom_embed = self.drug_fp(input_drug)
        # drug_atom_embed = [batch * num_node * drug_emb_dim]
        drug_embed = torch.sum(drug_atom_embed, dim=1)
        # drug_embed = [batch * drug_emb_dim]
        drug_embed = drug_embed.unsqueeze(1)
        # drug_embed = [batch * 1 *drug_emb_dim]
        drug_embed = drug_embed.repeat(1, self.num_gene, 1)
        # drug_embed = [batch * num_gene * drug_emb_dim]
        gene_embed = self.gene_embed(input_gene)
        # gene_embed = [num_gene * gene_emb_dim]
        gene_embed = gene_embed.unsqueeze(0)
        # gene_embed = [1 * num_gene * gene_emb_dim]
        gene_embed = gene_embed.repeat(num_batch, 1, 1)
        # gene_embed = [batch * num_gene * gene_emb_dim]
        drug_gene_embed, _ = self.drug_gene_attn(gene_embed, drug_atom_embed, None, mask)
        # drug_gene_embed = [batch * num_gene * gene_emb_dim]
        drug_gene_embed = torch.cat((drug_gene_embed, drug_embed), dim=2)
        # drug_gene_embed = [batch * num_gene * (drug_emb_dim + gene_emb_dim)]
        if self.use_pert_type:
            pert_type_embed = self.pert_type_embed(input_pert_type)
            # pert_type_embed = [batch * pert_type_emb_dim]
            pert_type_embed = pert_type_embed.unsqueeze(1)
            # pert_type_embed = [batch * 1 * pert_type_emb_dim]
            pert_type_embed = pert_type_embed.repeat(1, self.num_gene, 1)
            # pert_type_embed = [batch * num_gene * pert_type_emb_dim]
            drug_gene_embed = torch.cat((drug_gene_embed, pert_type_embed), dim=2)
        if self.use_cell_id:
            if epoch == 20:
                pass
                # pdb.set_trace()
            if linear_only:
                # input_cell_id = [batch * 978]
                if epoch % 100 == 80:
                    print('------followings are deepce sub before linear embed ----------')
                    print(input_cell_id)

                cell_id_embed = self.cell_id_embed_linear_only(input_cell_id)
                if epoch % 100 == 80:
                    print('------followings are deepce sub after linear embed ----------')
                    print(cell_id_embed)
                # cell_id_embed = [batch * cell_id_emb_dim(32)]
                cell_id_embed = cell_id_embed.unsqueeze(1)
                # cell_id_embed = [batch * 1 * cell_id_emb_dim]
                cell_hidden_ = cell_id_embed.contiguous().view(cell_id_embed.size(0), -1)
                cell_id_embed = cell_id_embed.repeat(1, self.num_gene, 1)
                # cell_id_embed = [batch * num_gene * cell_id_emb_dim(32)]
            else:
                # input_cell_id = [batch * 978]
                cell_id_embed = self.cell_id_embed(input_cell_id) # Transformer
                # cell_id_embed = [batch * 50]
                cell_id_embed = cell_id_embed.unsqueeze(-1)  # Transformer
                # cell_id_embed = [batch * 50 * 1]
                cell_id_embed = cell_id_embed.repeat(1,1,self.trans_cell_embed_dim)
                # cell_id_embed = self.cell_id_embed_1(cell_id_embed) # Transformer
                # cell_id_embed = [batch * 50 * 32(trans_cell_embed_dim)]
                if epoch % 100 == 80:
                    print('------followings are deepce sub before transformer ----------')
                    print(cell_id_embed)
                    torch.save(cell_id_embed, 'cell_id_embed_pre.pt')
                cell_id_embed = self.pos_encoder(cell_id_embed)
                cell_id_embed = self.cell_id_transformer(cell_id_embed, cell_id_embed) # Transformer
                # cell_id_embed = [batch * 50 * 32(trans_cell_embed_dim)]
                if epoch % 100 == 80:
                    print('------followings are deepce sub after transformer ----------')
                    print(cell_id_embed)
                    torch.save(cell_id_embed, 'cell_id_embed_post.pt')
                cell_hidden_, _ = torch.max(cell_id_embed, -1)
                # cell_hidden_ = [batch * 50]
                cell_id_embed = cell_hidden_.unsqueeze(1).repeat(1, self.num_gene, 1)
                # cell_hidden_ = cell_id_embed.contigous().view(cell_id_embed.size(0), -1) ## just to return the hidden representation 
                # cell_id_embed = self.expand_to_num_gene(cell_id_embed.transpose(-1,-2)).transpose(-1,-2) # Transformer
                # cell_id_embed = [batch * num_gene * 50]
            drug_gene_embed = torch.cat((drug_gene_embed, cell_id_embed), dim=2) # Transformer
        if self.use_pert_idose:
            pert_idose_embed = self.pert_idose_embed(input_pert_idose)
            # pert_idose_embed = [batch * pert_idose_emb_dim]
            pert_idose_embed = pert_idose_embed.unsqueeze(1)
            # pert_idose_embed = [batch * 1 * pert_idose_emb_dim]
            pert_idose_embed = pert_idose_embed.repeat(1, self.num_gene, 1)
            # pert_idose_embed = [batch * num_gene * pert_idose_emb_dim]
            drug_gene_embed = torch.cat((drug_gene_embed, pert_idose_embed), dim=2)
        # drug_gene_embed = [batch * num_gene * (drug_embed + gene_embed + pert_type_embed + cell_id_embed + pert_idose_embed)]
        drug_gene_embed = self.relu(drug_gene_embed)
        # drug_gene_embed = [batch * num_gene * (drug_embed + gene_embed + pert_type_embed + cell_id_embed + pert_idose_embed)]
        out = self.linear_1(drug_gene_embed)
        # out = [batch * num_gene * hid_dim]
        return out, cell_hidden_

    def gradual_unfreezing(self, unfreeze_pattern=[True, True, True]):
        ### unfreeze_pattern is a list a Three boolean value to indicate whether each layer should be frozen
        assert len(unfreeze_pattern) == 3, "number of boolean values doesn't match with the number of layers"
        if unfreeze_pattern[0]:
            print("All layers are unfrozen")
            for name, parameter in self.named_parameters():
                parameter.requires_grad = True
        elif unfreeze_pattern[1]:
            print("The first layer is still frozen")
            for name, parameter in self.named_parameters():
                if 'fp' not in name and 'embed' not in name:
                    parameter.requires_grad = True
                else:
                    print(name)

        elif unfreeze_pattern[2]:
            print("The first two layer is still frozen")
            for name, parameter in self.linear_1.named_parameters():
                parameter.requires_grad = True
        else:
            print("The first three layers are frozen")
            for name, parameter in self.named_parameters():
                parameter.requires_grad = False

class DeepCE(nn.Module):

    """
    The main model deepce
    """

    def __init__(self, drug_input_dim, drug_emb_dim, conv_size, degree, gene_input_dim, gene_emb_dim, num_gene,
                 hid_dim, dropout, loss_type, device, initializer=None, pert_type_input_dim=None,
                 cell_id_input_dim=None, pert_idose_input_dim=None,
                 pert_type_emb_dim=None, cell_id_emb_dim=None, pert_idose_emb_dim=None, use_pert_type=False,
                 use_cell_id=False, use_pert_idose=False):
        super(DeepCE, self).__init__()
        self.sub_deepce = DeepCESub(drug_input_dim, drug_emb_dim, conv_size, degree, gene_input_dim, gene_emb_dim, num_gene,
                 hid_dim, dropout, loss_type, device, initializer=initializer, pert_type_input_dim=pert_type_input_dim,
                 cell_id_input_dim=cell_id_input_dim, pert_idose_input_dim=pert_idose_input_dim,
                 pert_type_emb_dim=pert_type_emb_dim, cell_id_emb_dim=cell_id_emb_dim, pert_idose_emb_dim=pert_idose_emb_dim, 
                 use_pert_type=use_pert_type, use_cell_id=use_cell_id, use_pert_idose=use_pert_idose)
        self.loss_type = loss_type
        self.initializer = initializer
        self.device = device
        # self.init_weights()

    def init_weights(self):
        print('Initialized deepce\'s weight............')
        if self.initializer is None:
            return
        for name, parameter in self.named_parameters():
            if 'drug_gene_attn' not in name:
                if parameter.dim() == 1:
                    nn.init.constant_(parameter, 10**-7)
                else:
                    self.initializer(parameter)
            # if 'drug_gene_attn' not in name:
            #     if parameter.dim() == 1:
            #         nn.init.constant_(parameter, 10**-7)
            #     else:
            #         self.initializer(parameter)

    def forward(self, input_drug, input_gene, mask, input_pert_type, input_cell_id, input_pert_idose, epoch = 0, linear_only=False):
        # input_drug = {'molecules': molecules, 'atom': node_repr, 'bond': edge_repr}
        # gene_embed = [num_gene * gene_emb_dim]
        # out = [batch * num_gene * hid_dim]
        return self.sub_deepce(input_drug, input_gene, mask, input_pert_type, input_cell_id, input_pert_idose, epoch = epoch, linear_only = linear_only)

    @property
    def loss_type(self):
        return self.__loss_type
    
    @loss_type.setter
    def loss_type(self, loss_type):
        self.__loss_type = loss_type

    def loss(self, label, predict):
        if self.loss_type == 'point_wise_mse':
            loss = point_wise_mse(label, predict)
        elif self.loss_type == 'pair_wise_ranknet':
            loss = pair_wise_ranknet(label, predict, self.device)
        elif self.loss_type == 'list_wise_listnet':
            loss = list_wise_listnet(label, predict)
        elif self.loss_type == 'list_wise_listmle':
            loss = list_wise_listmle(label, predict, self.device)
        elif self.loss_type == 'list_wise_rankcosine':
            loss = list_wise_rankcosine(label, predict)
        elif self.loss_type == 'list_wise_ndcg':
            loss = list_wise_ndcg(label, predict)
        elif self.loss_type == 'combine':
            loss = class_combine_loss(label, predict, self.device)
        else:
            raise ValueError('Unknown loss: %s' % self.loss_type)
            
        return loss

class DeepCEOriginal(DeepCE):


    """
    DeepCE + relu + linear to predict the gene expression profile
    """

    def __init__(self, drug_input_dim, drug_emb_dim, conv_size, degree, gene_input_dim, gene_emb_dim, num_gene,
                 hid_dim, dropout, loss_type, device, initializer=None, pert_type_input_dim=None,
                 cell_id_input_dim=None, pert_idose_input_dim=None,
                 pert_type_emb_dim=None, cell_id_emb_dim=None, pert_idose_emb_dim=None, use_pert_type=False,
                 use_cell_id=False, use_pert_idose=False):
        super(DeepCEOriginal, self).__init__(drug_input_dim, drug_emb_dim, conv_size, degree, gene_input_dim, gene_emb_dim, num_gene,
                 hid_dim, dropout, loss_type, device, initializer=initializer, pert_type_input_dim=pert_type_input_dim,
                 cell_id_input_dim=cell_id_input_dim, pert_idose_input_dim=pert_idose_input_dim,
                 pert_type_emb_dim=pert_type_emb_dim, cell_id_emb_dim=cell_id_emb_dim, pert_idose_emb_dim=pert_idose_emb_dim, 
                 use_pert_type=use_pert_type, use_cell_id=use_cell_id, use_pert_idose=use_pert_idose)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(hid_dim, 1)
        self.init_weights()

    def forward(self, input_drug, input_gene, mask, input_pert_type, input_cell_id,
                input_pert_idose, epoch = 0, linear_only = False):
        out = super().forward(input_drug, input_gene, mask, input_pert_type, input_cell_id,
                              input_pert_idose, epoch = epoch)[0]
        # out = [batch * num_gene * hid_dim]
        out = self.relu(out)
        # out = [batch * num_gene * hid_dim]
        out = self.linear_2(out)
        # out = [batch * num_gene * 1]
        out = out.squeeze(2)
        # out = [batch * num_gene]
        return out
    
    def init_weights(self):
        print('Initialized deepce original\'s weight............')
        super().init_weights()
        print('used original models, no pretraining')
        #print('load old model')
        #self.sub_deepce.load_state_dict(torch.load('best_mode_ehill_storage_'))
        #print('frozen the parameters')
        #for param in self.sub_deepce.parameters():
        #    param.requires_grad = False
    
    def gradual_unfreezing(self, unfreeze_pattern=[True, True, True, True]):
        assert len(unfreeze_pattern) == 4, "length of unfreeze_pattern doesn't match model layers number"
        self.sub_deepce.gradual_unfreezing(unfreeze_pattern[:3])
        if sum(unfreeze_pattern[:3]) or unfreeze_pattern[3]: ### either one of the first three boolean values in unfreeze_pattern is True
            for name, parameter in self.linear_2.named_parameters():
                parameter.requires_grad = True
        else:
            for name, parameter in self.linear_2.named_parameters():
                parameter.requires_grad = False

class DeepCEPretraining(DeepCE):


    """
    DeepCE + two task specific linear to predict auc and ic50, which is used as a pretraining model
    """

    def __init__(self, drug_input_dim, drug_emb_dim, conv_size, degree, gene_input_dim, gene_emb_dim, num_gene,
                 hid_dim, dropout, loss_type, device, initializer=None, pert_type_input_dim=None,
                 cell_id_input_dim=None, pert_idose_input_dim=None,
                 pert_type_emb_dim=None, cell_id_emb_dim=None, pert_idose_emb_dim=None, use_pert_type=False,
                 use_cell_id=False, use_pert_idose=False):
        super(DeepCEPretraining, self).__init__(drug_input_dim, drug_emb_dim, conv_size, degree, gene_input_dim, gene_emb_dim, num_gene,
                 hid_dim, dropout, loss_type, device, initializer=initializer, pert_type_input_dim=pert_type_input_dim,
                 cell_id_input_dim=cell_id_input_dim, pert_idose_input_dim=pert_idose_input_dim,
                 pert_type_emb_dim=pert_type_emb_dim, cell_id_emb_dim=cell_id_emb_dim, pert_idose_emb_dim=pert_idose_emb_dim, 
                 use_pert_type=use_pert_type, use_cell_id=use_cell_id, use_pert_idose=use_pert_idose)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(hid_dim, 2)
        self.task_1_linear = nn.Sequential(nn.Linear(hid_dim, hid_dim//2), 
                                            nn.ReLU(), 
                                            nn.Linear(hid_dim//2, hid_dim//2),
                                            nn.ReLU(),
                                            nn.Linear(hid_dim//2, 1)
                                            )
        self.task_2_linear = nn.Sequential(nn.Linear(hid_dim, hid_dim//2), 
                                            nn.ReLU(), 
                                            nn.Linear(hid_dim//2, hid_dim//2),
                                            nn.ReLU(),
                                            nn.Linear(hid_dim//2, 1)
                                            )
        super().init_weights()

    def forward(self, input_drug, input_gene, mask, input_pert_type, input_cell_id, input_pert_idose):
        out = super().forward(input_drug, input_gene, mask, input_pert_type, input_cell_id, input_pert_idose)[0]
        # out = [batch * num_gene * hid_dim]
        out = self.relu(out)
        # out = [batch * num_gene * hid_dim]
        out = torch.sum(out, dim=1).squeeze(1)
        # out = [batch * 1 * hid_dim] => [batch * hid_dim] 
        out_1 = self.task_1_linear(out)
        # out_1 = [batch * 1] (pic50)
        out_2 = self.task_2_linear(out)
        # out_1 = [batch * 1] (auc)
        out = torch.cat((out_1, out_2), dim=1)
        # out = [batch * 2] (pic50 and auc)
        return out

    def gradual_unfreezing(self, unfreeze_pattern=[True, True, True, True]):
        assert len(unfreeze_pattern) == 4, "length of unfreeze_pattern doesn't match model layers number"
        self.sub_deepce.gradual_unfreezing(unfreeze_pattern[:3])
        for name, parameter in self.named_parameters():
                parameter.requires_grad = True

class DeepCEEhillPretraining(DeepCE):

    """
    DeepCE + task specific linear to predict ehill (whether dosage specific?)
    """

    def __init__(self, drug_input_dim, drug_emb_dim, conv_size, degree, gene_input_dim, gene_emb_dim, num_gene,
                 hid_dim, dropout, loss_type, device, initializer=None, pert_type_input_dim=None,
                 cell_id_input_dim=None, pert_idose_input_dim=None,
                 pert_type_emb_dim=None, cell_id_emb_dim=None, pert_idose_emb_dim=None, use_pert_type=False,
                 use_cell_id=False, use_pert_idose=False):
        super(DeepCEEhillPretraining, self).__init__(drug_input_dim, drug_emb_dim, conv_size, degree, gene_input_dim, gene_emb_dim, num_gene,
                 hid_dim, dropout, loss_type, device, initializer=initializer, pert_type_input_dim=pert_type_input_dim,
                 cell_id_input_dim=cell_id_input_dim, pert_idose_input_dim=pert_idose_input_dim,
                 pert_type_emb_dim=pert_type_emb_dim, cell_id_emb_dim=cell_id_emb_dim, pert_idose_emb_dim=pert_idose_emb_dim, 
                 use_pert_type=use_pert_type, use_cell_id=use_cell_id, use_pert_idose=use_pert_idose)
        self.relu = nn.ReLU()
        self.task_linear = nn.Sequential(nn.Linear(hid_dim, hid_dim//2), 
                                            nn.ReLU(), 
                                            nn.Linear(hid_dim//2, hid_dim//2),
                                            nn.ReLU(),
                                            nn.Linear(hid_dim//2, 1))

        self.genes_linear = nn.Sequential(nn.Linear(num_gene, num_gene//2),
                                            nn.ReLU(),
                                            nn.Linear(num_gene//2, num_gene//2),
                                            nn.ReLU(),
                                            nn.Linear(num_gene//2, 1))
        self.linear_2 = nn.Linear(hid_dim, 1)
        super().init_weights()

    def forward(self, input_drug, input_gene, mask, input_pert_type, input_cell_id,
                input_pert_idose, job_id = 'perturbed', epoch = 0, linear_only = False):

        out, cell_hidden_ = super().forward(input_drug, input_gene, mask, input_pert_type, input_cell_id,
                              input_pert_idose, epoch = epoch, linear_only = linear_only)
        if epoch % 100 == 80:            
            torch.save(input_cell_id, 'input_cell_feature.pt')
        # out = [batch * num_gene * hid_dim]
        out = self.relu(out)
        # out = [batch * num_gene * hid_dim]
        if job_id == 'perturbed':
            out = self.linear_2(out)
            # out = [batch * num_gene * 1]
            out = out.squeeze(2)
            # out = [batch * num_gene]
        else:
            out = self.task_linear(out)
            # out = [batch * num_gene * 1]
            out = out.squeeze(2)
            # out = [batch * num_gene]
            out = self.genes_linear(out)
            # out = [batch*1] (ehill)
        if epoch % 100 == 80:            
            torch.save(out, 'predicted_cell_feature.pt')
        return out, cell_hidden_

    def init_weights(self, pretrained = False):
        print('Initialized deepce original\'s weight............')
        super().init_weights()
        print('used original models, no pretraining')
        for name, parameter in self.named_parameters():
            if 'drug_gene_attn' not in name:
                if parameter.dim() == 1:
                    nn.init.constant_(parameter, 10**-7)
                else:
                    self.initializer(parameter)
        if pretrained:
            print('used pretrained models')
            self.sub_deepce.load_state_dict(torch.load('best_model_ehill_storage_linear_'))

    def gradual_unfreezing(self, unfreeze_pattern=[True, True, True, True]):
        assert len(unfreeze_pattern) == 4, "length of unfreeze_pattern doesn't match model layers number"
        self.sub_deepce.gradual_unfreezing(unfreeze_pattern[:3])
        for name, parameter in self.named_parameters():
                parameter.requires_grad = True

class DeepCE_AE(DeepCE):

    """
    DeepCE + encoder,decoder structure (if job id == 'perturbed", deepce, else, encoder, decoder_only)
    """

    def __init__(self, drug_input_dim, drug_emb_dim, conv_size, degree, gene_input_dim, gene_emb_dim, num_gene,
                 hid_dim, dropout, loss_type, device, initializer=None, pert_type_input_dim=None,
                 cell_id_input_dim=None, pert_idose_input_dim=None,
                 pert_type_emb_dim=None, cell_id_emb_dim=None, cell_decoder_dim = None, pert_idose_emb_dim=None, use_pert_type=False,
                 use_cell_id=False, use_pert_idose=False):
        super(DeepCE_AE, self).__init__(drug_input_dim, drug_emb_dim, conv_size, degree, gene_input_dim, gene_emb_dim, num_gene,
                 hid_dim, dropout, loss_type, device, initializer=initializer, pert_type_input_dim=pert_type_input_dim,
                 cell_id_input_dim=cell_id_input_dim, pert_idose_input_dim=pert_idose_input_dim,
                 pert_type_emb_dim=pert_type_emb_dim, cell_id_emb_dim=cell_id_emb_dim, pert_idose_emb_dim=pert_idose_emb_dim, 
                 use_pert_type=use_pert_type, use_cell_id=use_cell_id, use_pert_idose=use_pert_idose)
        self.guassian_noise = GaussianNoise(device=device)
        self.relu = nn.ReLU()
        self.trans_cell_embed_dim = self.sub_deepce.trans_cell_embed_dim
        self.linear_2 = nn.Linear(hid_dim, 3)
        self.decoder_1 = nn.Linear(self.trans_cell_embed_dim, 1)
        self.decoder_2 = nn.Sequential(nn.Linear(50, 200), nn.Linear(200, cell_decoder_dim))
        self.decoder_linear = nn.Sequential(nn.Linear(50, 200), nn.Linear(200, cell_decoder_dim))
        self.init_weights()

    def forward(self, input_drug, input_gene, mask, input_pert_type, input_cell_id, input_pert_idose,
                job_id = 'perturbed', epoch = 0, linear_only=False):
        if job_id == 'perturbed':
            if epoch % 100 == 80:                
                torch.save(input_cell_id, 'input_cell_feature.pt')
            out, cell_hidden_ = super().forward(input_drug, input_gene, mask, input_pert_type, input_cell_id,
                                                input_pert_idose, epoch = epoch, linear_only = linear_only)
            # out = [batch * num_gene * hid_dim]
            out = self.relu(out)
            # out = [batch * num_gene * hid_dim]
            out = self.linear_2(out)
            # out = [batch * num_gene * 1]
            # out = out.squeeze(2)
            if epoch % 100 == 80:                
                torch.save(out, 'predicted_cell_feature.pt')
            # out = [batch * num_gene]
            return out, cell_hidden_
        else:
            hidden = self.guassian_noise(input_cell_id)
            if linear_only:
                # input_cell_id = [batch * 978]
                cell_hidden_ = self.sub_deepce.cell_id_embed_linear_only(hidden)
                # cell_hidden_ = [batch * cell_id_emb_dim(32)]
                out_2 = self.decoder_linear(cell_hidden_)
            else:
                hidden = self.sub_deepce.cell_id_embed(hidden)
                if epoch % 100 == 80:
                    print('---------------------followings are ae before transformer in ae-------------------------')
                    print(hidden)
                    new_hidden = hidden.clone()
                    new_input_cell_id = input_cell_id.clone()
                    torch.save(new_hidden, 'new_hidden.pt')
                    torch.save(new_input_cell_id, 'new_input_cell_id.pt')

                hidden = hidden.unsqueeze(-1)  # Transformer
                # hidden = [batch * cell_id_emb_dim * 1]
                hidden = hidden.repeat(1,1,self.trans_cell_embed_dim)
                # hidden = self.sub_deepce.cell_id_embed_1(hidden) # Transformer
                # hidden = [batch * cell_id_emb_dim * trans_cell_embed_dim(32))]
                hidden = self.sub_deepce.pos_encoder(hidden)
                hidden = self.sub_deepce.cell_id_transformer(hidden, hidden)
                # hidden = [batch * cell_id_emb_dim * 32]
                cell_hidden_, _ = torch.max(hidden, -1)
                # cell_hidden_ = [batch * cell_id_emb_dim]
                # cell_hidden_ = self.decoder_1(hidden).squeeze(-1)
                # cell_hidden_ = [batch * ]
                out_2 = self.decoder_2(cell_hidden_)
            
            if epoch % 100 == 80:

                print('---------------------followings are ae after transformer/linear embed in ae-------------------------')
                print(out_2)
                new_out_2 = out_2.clone()
                torch.save(new_out_2, 'new_out_2.pt')
            # out_2 = [batch * cell_decoder_dim]
            return out_2, cell_hidden_
    
    def init_weights(self, pretrained = False):
        print('Initialized deepce original\'s weight............')
        super().init_weights()
        print('used original models, no pretraining')
        for name, parameter in self.named_parameters():
            if 'drug_gene_attn' not in name:
                if parameter.dim() == 1:
                    nn.init.constant_(parameter, 10**-7)
                else:
                    self.initializer(parameter)
        if pretrained:
            self.sub_deepce.load_state_dict(torch.load('best_model_ehill_storage_'))
            # if 'attn' not in name:
            #     if parameter.dim() == 1:
            #         nn.init.constant_(parameter, 10**-7)
            #     else:
            #         self.initializer(parameter)
        #print('load old model')
        #self.sub_deepce.load_state_dict(torch.load('best_mode_storage_'))
        #print('frozen the parameters')
        #for param in self.sub_deepce.parameters():
        #    param.requires_grad = False

    def gradual_unfreezing(self, unfreeze_pattern=[True, True, True, True]):
        assert len(unfreeze_pattern) == 4, "length of unfreeze_pattern doesn't match model layers number"
        self.sub_deepce.gradual_unfreezing(unfreeze_pattern[:3])
        for name, parameter in self.named_parameters():
                parameter.requires_grad = True

    # def loss(self, label, predict, hidden = None, cell_type = None):
    #     if self.loss_type == 'mse_and_homophily' and hidden is not None and cell_type is not None:
    #         loss = mse_plus_homophily(label, predict, hidden, cell_type)
    #         return loss
    #     else:
    #         return super.loss()
