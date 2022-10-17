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
        output = F.linear(x, self.W * self.m, self.b)
        output = F.relu(output)
        return output


class TemporalDecay(nn.Module):
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
        if self.diag:
            gamma = F.relu(F.linear(delta, self.W * Variable(self.m), self.b))
        else:
            gamma = F.relu(F.linear(delta, self.W, self.b))
        gamma = torch.exp(-gamma)
        return gamma


class RITS(nn.Module):
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


class MACRO(nn.Module):
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
        imputed_data_f, _, _ = self.rits_f.impute(inputs, 'forward')
        imputed_data_b, _, _ = self.rits_b.impute(inputs, 'backward')
        imputed_data_b = {'imputed_data_b': imputed_data_b}
        imputed_data_b = self.reverse(imputed_data_b)['imputed_data_b']
        imputed_data = (imputed_data_f + imputed_data_b) / 2
        return imputed_data

    @staticmethod
    def get_consistency_loss(pred_f, pred_b):
        loss = torch.abs(pred_f - pred_b).mean() * 1e-1
        return loss

    @staticmethod
    def reverse(ret):
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
        consistency_loss = self.get_consistency_loss(ret_f['imputed_data'], ret_b['imputed_data'])
        ret_f['imputed_data'] = (ret_f['imputed_data'] + ret_b['imputed_data']) / 2
        ret_f['consistency_loss'] = consistency_loss
        ret_f['loss'] = consistency_loss + \
                        ret_f['reconstruction_loss'] + \
                        ret_b['reconstruction_loss']

        return ret_f

    def forward(self, inputs):
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
