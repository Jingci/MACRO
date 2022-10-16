"""
PyTorch BRITS model for the time-series imputation task.
Some part of the code is from https://github.com/caow13/BRITS.
"""
# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GPL-v3

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter

from pypots.utils.metrics import cal_mae
import numpy as np

class AE(nn.Module):
    def __init__(self, infeat, nhid, drop=0.):
        super(AE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(infeat, nhid),
            nn.ReLU(),
            nn.Linear(nhid, nhid),
        )
        self.decoder = nn.Sequential(
            nn.Linear(nhid, nhid),
            nn.ReLU(),
            nn.Linear(nhid, infeat),
        )
        self.drop = nn.Dropout(drop)
    def forward(self,X_value):
        x=self.drop(X_value) #; x=torch.transpose(x,2,1)
        h = self.encoder(x)
        # decode
        x = self.decoder(h) #; x = torch.transpose(x,2,1)
        return x, h

    
def get_ks_graph(g0, k):
    g_all = []
    current_g = g0
    for i in range(k):
        g_all.append( (current_g>0).int() * (1-torch.eye(current_g.shape[0])) )
        current_g = torch.mm(current_g, g0)
    return g_all


def get_graphs(input_size, graph_ids, gk, dist_threshold, DATA, sk):
    # get graph
    graphs = []
    for gid in graph_ids:
        if gid == 0:
            m = torch.zeros(input_size, input_size)
            adj1 = np.load('data/adj_ln_dffcs.npy')
            adj2 = np.load('data/adj_cs_near.npy')
            adj3 = np.load('data/adj_cs_dffcs.npy')
            for i,j in adj1.T:
                m[int(i)][int(j)] = 1; m[int(j)][int(i)] = 1
            for i,j in adj2.T:
                m[int(i)][int(j)] = 1; m[int(j)][int(i)] = 1
            for i,j in adj3.T:
                m[int(i)][int(j)] = 1; m[int(j)][int(i)] = 1
            graphs += get_ks_graph(m, gk)
        elif gid == 1:
            m = torch.zeros(input_size, input_size)
            distance_matrix = torch.from_numpy(np.load('data/distance.npy'))
            m = m + (distance_matrix < dist_threshold).int()
            graphs += get_ks_graph(m, gk)
        elif gid == 2:
            similarity_matrix = torch.from_numpy(np.load(f'data/similarity_graph/adj_similarity_{DATA}_train.npy'))
            similarity_matrix -= torch.eye(input_size)
            for i in range(similarity_matrix.shape[0]):
                k_th_item = torch.topk(similarity_matrix[i], sk)[0][-1]
                similarity_matrix[i] = (similarity_matrix[i] > k_th_item).int()
            graphs += get_ks_graph(similarity_matrix, gk)
    graphs = torch.stack(graphs)
    return graphs


class FullAttention(nn.Module):
    def __init__(self, attention_dropout=0.1):
        super(FullAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = 1. / math.sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        return (V.contiguous(), A)

class AttentionLayer(nn.Module):
    def __init__(self, d_inp, d_model, n_heads):
        super(AttentionLayer, self).__init__()

        d_keys = d_model // n_heads
        d_values = d_model // n_heads
        self.inner_attention = FullAttention(0.1)

        self.query_projection = nn.Linear(d_inp, d_keys * n_heads)
        self.key_projection = nn.Linear(d_inp, d_keys * n_heads)
        self.value_projection = nn.Linear(d_inp, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_inp)
        self.n_heads = n_heads

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        out, attn = self.inner_attention(queries, keys, values)
        out = out.view(B, L, -1)
        #return out
        return self.out_projection(out) #, attn


class GraphFeatureRegression(nn.Module):
    def __init__(self, input_size, num_metag ,gk, graphs, use_attn=False):
        super().__init__()
        self.k = gk
        self.num_metag = num_metag
        self.use_attn = use_attn
        # prior graph
        graphs = graphs
        self.register_buffer('graphs', graphs)
        # parameters
        num_graph = graphs.shape[0]
        self.Ws = nn.Parameter(torch.Tensor(num_graph, input_size, input_size))
        self.bs = nn.Parameter(torch.Tensor(num_graph, input_size))
        self.reset_parameters()
        # reduce
        self.reduce_layers = nn.ModuleList([nn.Linear(self.k, 1) for _ in range(self.num_metag)])
        self.reduce2 = nn.Linear(self.num_metag, 1)
        # attention weights
        self.attn_weights = nn.Linear(input_size, 1)
    def reset_parameters(self):
        for i in range(self.Ws.shape[0]):
            stdv = 1. / math.sqrt(self.Ws[i].size(0))
            self.Ws[i].data.uniform_(-stdv, stdv)
            self.bs[i].data.uniform_(-stdv, stdv)
            
    def forward(self, x):
        outputs = []
        for i in range(self.graphs.shape[0]):
            rst = F.relu( F.linear(x, self.Ws[i] * self.graphs[i], self.bs[i]) )
            outputs.append(rst)
        outputs = torch.stack(outputs, dim=-1)
        reduced = []
        for j in range(self.num_metag):
            current = outputs[:,:,j:j+self.k]
            current = self.reduce_layers[j](current)
            reduced.append(current.squeeze(-1))
        reduced = torch.stack(reduced, dim=-1)
        return self.reduce2(reduced).squeeze(-1)

class FeatureRegression(nn.Module):
    """ The module used to capture the correlation between features for imputation.

    Attributes
    ----------
    W : tensor
        The weights (parameters) of the module.
    b : tensor
        The bias of the module.
    m (buffer) : tensor
        The mask matrix, a squire matrix with diagonal entries all zeroes while left parts all ones.
        It is applied to the weight matrix to mask out the estimation contributions from features themselves.
        It is used to help enhance the imputation performance of the network.

    Parameters
    ----------
    input_size : the feature dimension of the input
    """

    def __init__(self, input_size, graph_params=None):
        super().__init__()
        self.W = Parameter(torch.Tensor(input_size, input_size))
        self.b = Parameter(torch.Tensor(input_size))
        # m = prior_graph(graph_params, input_size) # torch.tensor
        m = torch.ones(input_size, input_size) - torch.eye(input_size, input_size)
        self.register_buffer('m', m)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, x):
        """ Forward processing of the NN module.

        Parameters
        ----------
        x : tensor,
            the input for processing

        Returns
        -------
        output: tensor,
            the processed result containing imputation from feature regression

        """
        
        output = F.linear(x, self.W * self.m, self.b)
        output = F.relu(output)
        return output


class TemporalDecay(nn.Module):
    """ The module used to generate the temporal decay factor gamma in the original paper.

    Attributes
    ----------
    W: tensor,
        The weights (parameters) of the module.
    b: tensor,
        The bias of the module.

    Parameters
    ----------
    input_size : int,
        the feature dimension of the input
    output_size : int,
        the feature dimension of the output
    diag : bool,
        whether to product the weight with an identity matrix before forward processing
    """

    def __init__(self, input_size, output_size, diag=False):
        super().__init__()
        self.diag = diag
        self.W = Parameter(torch.Tensor(output_size, input_size))
        self.b = Parameter(torch.Tensor(output_size))

        if self.diag:
            assert (input_size == output_size)
            m = torch.eye(input_size, input_size)
            self.register_buffer('m', m)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, delta):
        """ Forward processing of the NN module.

        Parameters
        ----------
        delta : tensor, shape [batch size, sequence length, feature number]
            The time gaps.

        Returns
        -------
        gamma : array-like, same shape with parameter `delta`, values in (0,1]
            The temporal decay factor.
        """
        if self.diag:
            gamma = F.relu(F.linear(delta, self.W * Variable(self.m), self.b))
        else:
            gamma = F.relu(F.linear(delta, self.W, self.b))
        gamma = torch.exp(-gamma)
        return gamma


class RITS(nn.Module):
    """ model RITS: Recurrent Imputation for Time Series

    Attributes
    ----------
    n_steps : int,
        sequence length (number of time steps)
    n_features : int,
        number of features (input dimensions)
    rnn_hidden_size : int,
        the hidden size of the RNN cell
    device : str, default=None,
        specify running the model on which device, CPU/GPU
    rnn_cell : torch.nn.module object
        the LSTM cell to model temporal data
    temp_decay_h : torch.nn.module object
        the temporal decay module to decay RNN hidden state
    temp_decay_x : torch.nn.module object
        the temporal decay module to decay data in the raw feature space
    hist_reg : torch.nn.module object
        the temporal-regression module to project RNN hidden state into the raw feature space
    feat_reg : torch.nn.module object
        the feature-regression module
    combining_weight : torch.nn.module object
        the module used to generate the weight to combine history regression and feature regression

    Parameters
    ----------
    n_steps : int,
        sequence length (number of time steps)
    n_features : int,
        number of features (input dimensions)
    rnn_hidden_size : int,
        the hidden size of the RNN cell
    device : str,
        specify running the model on which device, CPU/GPU
    """

    def __init__(self, n_steps, n_features, rnn_hidden_size, graph_params, attn_params, cl_lambda, device=None):
        super().__init__()
        self.n_steps = n_steps
        self.n_features = n_features
        self.rnn_hidden_size = rnn_hidden_size
        self.device = device

        self.rnn_cell = nn.LSTMCell(self.n_features * 2, self.rnn_hidden_size)
        self.temp_decay_h = TemporalDecay(input_size=self.n_features, output_size=self.rnn_hidden_size, diag=False)
        self.temp_decay_x = TemporalDecay(input_size=self.n_features, output_size=self.n_features, diag=True)
        self.hist_reg = nn.Linear(self.rnn_hidden_size, self.n_features)
        if graph_params is None:
            self.feat_reg = None
        else:
            num_metag, gk, graphs = graph_params
            if attn_params is None:
                use_attn, d_model, n_heads = False, None, None
            else:
                use_attn, d_model, n_heads = attn_params

            if num_metag == 0:
                self.feat_reg = FeatureRegression(self.n_features)
            else:
                self.feat_reg = GraphFeatureRegression(input_size=n_features, num_metag=num_metag, gk=gk, graphs=graphs, use_attn=use_attn, d_model=d_model, n_heads=n_heads)
        
        self.combining_weight = nn.Linear(self.n_features * 2, self.n_features)
        # contrastive learning
        self.cl_lambda = cl_lambda
        self.contrastive_weight = nn.Linear(rnn_hidden_size, rnn_hidden_size)
        self.logceloss = nn.CrossEntropyLoss()
        
    def impute(self, inputs, direction):
        """ The imputation function.
        Parameters
        ----------
        inputs : dict,
            Input data, a dictionary includes feature values, missing masks, and time-gap values.
        direction : str, 'forward'/'backward'
            A keyword to extract data from parameter `data`.

        Returns
        -------
        imputed_data : tensor,
            [batch size, sequence length, feature number]
        hidden_states: tensor,
            [batch size, RNN hidden size]
        reconstruction_loss : float tensor,
            reconstruction loss
        """
        values = inputs[direction]['X']  # feature values
        masks = inputs[direction]['missing_mask']  # missing masks
        deltas = inputs[direction]['deltas']  # time-gap values

        #create hidden states and cell states for the lstm cell
        if 'rnn_hidden' not in inputs[direction] or inputs[direction]['rnn_hidden'] is None:
            ae_hidden_vector = None
        else:
            ae_hidden_vector = torch.mean(inputs[direction]['rnn_hidden'],axis=1).squeeze(1)

        if 'rnn_hidden' not in inputs[direction] or inputs[direction]['rnn_hidden'] is None:
            hidden_states = torch.zeros((values.size()[0], self.rnn_hidden_size), device=self.device)
        else:
            #print("with ae")
            hidden_states = ae_hidden_vector.clone()
        
        if 'rnn_hidden' not in inputs[direction] or inputs[direction]['rnn_hidden'] is None:
            #print("without ae")
            cell_states = torch.zeros((values.size()[0], self.rnn_hidden_size), device=self.device)
        else:
            #print("with ae")
            cell_states = ae_hidden_vector.clone()


        estimations = []
        reconstruction_loss = 0.0

        # imputation period
        for t in range(self.n_steps):
            # data shape: [batch, time, features]
            x = values[:, t, :]  # values
            m = masks[:, t, :]  # mask
            d = deltas[:, t, :]  # delta, time gap

            gamma_h = self.temp_decay_h(d)
            gamma_x = self.temp_decay_x(d)

            hidden_states = hidden_states * gamma_h  # decay hidden states
            x_h = self.hist_reg(hidden_states)
            reconstruction_loss += cal_mae(x_h, x, m)

            x_c = m * x + (1 - m) * x_h
            
            if self.feat_reg is None:
                z_h = 0
            else:
                z_h = self.feat_reg(x_c)
                reconstruction_loss += cal_mae(z_h, x, m)

            alpha = torch.sigmoid(self.combining_weight(torch.cat([gamma_x, m], dim=1)))

            c_h =  alpha * z_h + (1 - alpha) * x_h
            reconstruction_loss += cal_mae(c_h, x, m)

            ### contrastive loss ###
            if (self.cl_lambda > 0) and (ae_hidden_vector is not None):
                ae_h = self.contrastive_weight(ae_hidden_vector) # [B, N]
                #ae_h = ae_hidden_vector # [B, N]
                logits = torch.mm(hidden_states, ae_h.transpose(1,0)) # (B,B)
                logits = logits - logits.max(dim=-1,keepdim=True).values
                labels = torch.arange(0,logits.shape[0]).to(self.device).long()
                cl_loss = self.logceloss(logits, labels)
                reconstruction_loss += self.cl_lambda*cl_loss
            ### contrastive loss ###

            c_c = m * x + (1 - m) * c_h
            estimations.append(c_h.unsqueeze(dim=1))

            inputs = torch.cat([c_c, m], dim=1)
            hidden_states, cell_states = self.rnn_cell(inputs, (hidden_states, cell_states))

        estimations = torch.cat(estimations, dim=1)
        imputed_data = masks * values + (1 - masks) * estimations
        return imputed_data, hidden_states, reconstruction_loss

    def forward(self, inputs, direction='forward'):
        """ Forward processing of the NN module.
        Parameters
        ----------
        inputs : dict,
            The input data.

        direction : string, 'forward'/'backward'
            A keyword to extract data from parameter `data`.

        Returns
        -------
        dict,
            A dictionary includes all results.
        """
        imputed_data, hidden_state, reconstruction_loss = self.impute(inputs, direction)
        # for each iteration, reconstruction_loss increases its value for 3 times
        reconstruction_loss /= (self.n_steps * 3)

        ret_dict = {
            'consistency_loss': torch.tensor(0.0, device=self.device),  # single direction, has no consistency loss
            'reconstruction_loss': reconstruction_loss,
            'imputed_data': imputed_data,
            'final_hidden_state': hidden_state
        }
        return ret_dict


class _BRITS(nn.Module):
    """ model BRITS: Bidirectional RITS
    BRITS consists of two RITS, which take time-series data from two directions (forward/backward) respectively.

    Attributes
    ----------
    n_steps : int,
        sequence length (number of time steps)
    n_features : int,
        number of features (input dimensions)
    rnn_hidden_size : int,
        the hidden size of the RNN cell
    rits_f: RITS object
        the forward RITS model
    rits_b: RITS object
        the backward RITS model
    """

    def __init__(self, n_steps, n_features, rnn_hidden_size, graph_params, attn_params, cl_lambda, device=None):
        super().__init__()
        # data settings
        self.n_steps = n_steps
        self.n_features = n_features
        # imputer settings
        self.rnn_hidden_size = rnn_hidden_size
        # create models
        self.rits_f = RITS(n_steps, n_features, rnn_hidden_size, graph_params, attn_params, cl_lambda, device)
        self.rits_b = RITS(n_steps, n_features, rnn_hidden_size, graph_params, attn_params, cl_lambda, device)

    def impute(self, inputs):
        """ Impute the missing data. Only impute, this is for test stage.

        Parameters
        ----------
        inputs : dict,
            A dictionary includes all input data.

        Returns
        -------
        array-like, the same shape with the input feature vectors.
            The feature vectors with missing part imputed.

        """
        imputed_data_f, _, _ = self.rits_f.impute(inputs, 'forward')
        imputed_data_b, _, _ = self.rits_b.impute(inputs, 'backward')
        imputed_data_b = {'imputed_data_b': imputed_data_b}
        imputed_data_b = self.reverse(imputed_data_b)['imputed_data_b']
        imputed_data = (imputed_data_f + imputed_data_b) / 2
        return imputed_data

    @staticmethod
    def get_consistency_loss(pred_f, pred_b):
        """ Calculate the consistency loss between the imputation from two RITS models.

        Parameters
        ----------
        pred_f : array-like,
            The imputation from the forward RITS.
        pred_b : array-like,
            The imputation from the backward RITS (already gets reverted).

        Returns
        -------
        float tensor,
            The consistency loss.
        """
        loss = torch.abs(pred_f - pred_b).mean() * 1e-1
        return loss

    @staticmethod
    def reverse(ret):
        """ Reverse the array values on the time dimension in the given dictionary.

        Parameters
        ----------
        ret : dict

        Returns
        -------
        dict,
            A dictionary contains values reversed on the time dimension from the given dict.
        """

        def reverse_tensor(tensor_):
            if tensor_.dim() <= 1:
                return tensor_
            indices = range(tensor_.size()[1])[::-1]
            indices = torch.tensor(indices, dtype=torch.long, device=tensor_.device, requires_grad=False)
            return tensor_.index_select(1, indices)

        for key in ret:
            ret[key] = reverse_tensor(ret[key])

        return ret

    def merge_ret(self, ret_f, ret_b):
        """ Merge (average) results from two RITS models into one.

        Parameters
        ----------
        ret_f : dict,
            Results from the forward RITS.
        ret_b : dict,
            Results from the backward RITS.

        Returns
        -------
        dict,
            Merged results in a dictionary.
        """
        consistency_loss = self.get_consistency_loss(ret_f['imputed_data'], ret_b['imputed_data'])
        ret_f['imputed_data'] = (ret_f['imputed_data'] + ret_b['imputed_data']) / 2
        ret_f['consistency_loss'] = consistency_loss
        ret_f['loss'] = consistency_loss + \
                        ret_f['reconstruction_loss'] + \
                        ret_b['reconstruction_loss']

        return ret_f

    def forward(self, inputs):
        """ Forward processing of BRITS.

        Parameters
        ----------
        inputs : dict,
            The input data.

        Returns
        -------
        dict, A dictionary includes all results.
        """
        ret_f = self.rits_f(inputs, 'forward')
        ret_b = self.reverse(self.rits_b(inputs, 'backward'))
        ret = self.merge_ret(ret_f, ret_b)
        return ret

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_model = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model, epoch=0):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_model = copy.deepcopy(model) #self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience} with {epoch}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model = copy.deepcopy(model) #self.save_checkpoint(val_loss, model)
            self.counter = 0
