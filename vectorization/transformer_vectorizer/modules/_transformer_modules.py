import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np

__author__ = "Yu-Hsiang Huang"


# github https://github.com/jadore801120/attention-is-all-you-need-pytorch


class Linear(nn.Module):
    ''' Simple Linear layer with xavier init '''

    def __init__(self, d_in, d_out, bias=True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(d_in, d_out, bias=bias)
        init.xavier_normal_(self.linear.weight)

    def forward(self, x):
        return self.linear(x)


class Bottle(nn.Module):
    ''' Perform the reshape routine before and after an operation '''

    def forward(self, input):
        if len(input.size()) <= 2:
            return super(Bottle, self).forward(input)
        size = input.size()[:2]
        out = super(Bottle, self).forward(input.view(size[0] * size[1], -1))
        return out.view(size[0], size[1], -1)


class BottleLinear(Bottle, Linear):
    ''' Perform the reshape routine before and after a linear projection '''
    pass


class BottleSoftmax(Bottle, nn.Softmax):
    ''' Perform the reshape routine before and after a softmax operation'''
    pass


class LayerNormalization(nn.Module):
    ''' Layer normalization module '''

    def __init__(self, d_hid, eps=1e-3):
        super(LayerNormalization, self).__init__()

        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(d_hid), requires_grad=True)
        self.b_2 = nn.Parameter(torch.zeros(d_hid), requires_grad=True)

    def forward(self, z):
        if z.size(1) == 1:
            return z

        mu = torch.mean(z, keepdim=True, dim=-1)
        sigma = torch.std(z, keepdim=True, dim=-1)
        ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
        ln_out = ln_out * self.a_2.expand_as(ln_out) + self.b_2.expand_as(ln_out)

        return ln_out


class BatchBottle(nn.Module):
    ''' Perform the reshape routine before and after an operation '''

    def forward(self, input):
        if len(input.size()) <= 2:
            return super(BatchBottle, self).forward(input)
        size = input.size()[1:]
        out = super(BatchBottle, self).forward(input.view(-1, size[0] * size[1]))
        return out.view(-1, size[0], size[1])


class BottleLayerNormalization(BatchBottle, LayerNormalization):
    ''' Perform the reshape routine before and after a layer normalization'''
    pass


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, d_model, attn_dropout=0.):
        super(ScaledDotProductAttention, self).__init__()
        self.temper = np.power(d_model, 0.5)
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = BottleSoftmax(dim=-1)

    def forward(self, q, k, v, attn_mask=None):
        attn = torch.bmm(q, k.transpose(1, 2)) / self.temper

        if attn_mask is not None:
            assert attn_mask.size() == attn.size(), \
                'Attention mask shape {} mismatch ' \
                'with Attention logit tensor shape ' \
                '{}.'.format(attn_mask.size(), attn.size())

            attn.data.masked_fill_(attn_mask, -float('inf'))

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k=None, d_v=None, d_out=None,
                 use_residual=True, dropout=0.):
        """
        :param n_head: number of parallel attentions (to be concatenated)
        :param d_model: previous layer's last dimension
        :param d_k: size of query and key vectors, defaults to d_model
        :param d_v: size of value vectors, defaults to d_model
        :param d_out: size of output vector, defaults to d_model
        :param use_residual: if True, adds q to output (only works if shapes of q and v are equal)
        """
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_k = d_k = d_k or d_model
        self.d_v = d_v = d_v or d_model
        self.d_out = d_out = d_out or d_model
        self.use_residual = use_residual

        if self.use_residual:
            assert d_model == d_out, "if you want residual, input must be of same size as output"

        self.w_qs = nn.Parameter(torch.FloatTensor(n_head, d_model, d_k))
        self.w_ks = nn.Parameter(torch.FloatTensor(n_head, d_model, d_k))
        self.w_vs = nn.Parameter(torch.FloatTensor(n_head, d_model, d_v))

        self.attention = ScaledDotProductAttention(d_model)
        self.proj = Linear(n_head * d_v, d_out)
        self.layer_norm = LayerNormalization(d_out)

        self.dropout = nn.Dropout(dropout)

        init.xavier_normal_(self.w_qs)
        init.xavier_normal_(self.w_ks)
        init.xavier_normal_(self.w_vs)

    def forward(self, q, k=None, v=None, attn_mask=None, return_attn=False):
        """
        :param q: source of queries (w_qs is applied to this vector), also added to output if residual==True
        :type q: Variable(FloatTensor), shape=[batch, n_inp, hid_size]
        :param k: source of keys (w_ks is applied to this vector), defaults to same as q
        :type k: Variable(FloatTensor), shape=[batch, n_inp, hid_size]
        :param v: source of values (w_vs is applied to this vector), defaults to same as q
        :type v: Variable(FloatTensor), shape=[batch, n_inp, hid_size]
        :param attn_mask: if mask is 0, forbids attention to this ninp
        :type mask: Variable(BoolTensor), shape=[batch, n_inp_q, n_inp_kv]
        """
        k = k if k is not None else q
        v = v if v is not None else q

        d_k, d_v = self.d_k, self.d_v
        n_head = self.n_head

        residual = q

        mb_size, len_q, d_model = q.size()
        mb_size, len_k, d_model = k.size()
        mb_size, len_v, d_model = v.size()

        # treat as a (n_head) size batch
        q_s = q.repeat(n_head, 1, 1).view(n_head, -1, d_model)  # n_head x (mb_size*len_q) x d_model
        k_s = k.repeat(n_head, 1, 1).view(n_head, -1, d_model)  # n_head x (mb_size*len_k) x d_model
        v_s = v.repeat(n_head, 1, 1).view(n_head, -1, d_model)  # n_head x (mb_size*len_v) x d_model

        # treat the result as a (n_head * mb_size) size batch
        q_s = torch.bmm(q_s, self.w_qs).view(-1, len_q, d_k)  # (n_head*mb_size) x len_q x d_k
        k_s = torch.bmm(k_s, self.w_ks).view(-1, len_k, d_k)  # (n_head*mb_size) x len_k x d_k
        v_s = torch.bmm(v_s, self.w_vs).view(-1, len_v, d_v)  # (n_head*mb_size) x len_v x d_v

        # perform attention, result size = (n_head * mb_size) x len_q x d_v
        outputs, attns = self.attention(q_s, k_s, v_s,
                                        attn_mask=None if attn_mask is None else attn_mask.repeat(n_head, 1, 1))

        # back to original mb_size batch, result size = mb_size x len_q x (n_head*d_v)
        outputs = torch.cat(torch.split(outputs, mb_size, dim=0), dim=-1)

        # project back to residual size
        outputs = self.proj(outputs)
        outputs = self.dropout(outputs)
        if self.use_residual:
            outputs = outputs + residual
        outputs = self.layer_norm(outputs)
        return (outputs, attns) if return_attn else outputs


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_hid, d_inner_hid=None, d_out=None, use_residual=True, dropout=0.):
        super(PositionwiseFeedForward, self).__init__()
        d_inner_hid = d_inner_hid or d_hid
        d_out = d_out or d_hid
        self.use_residual = use_residual
        if self.use_residual:
            assert d_hid == d_out, "if you want residual, input must be of same size as output"

        self.w_1 = nn.Conv1d(d_hid, d_inner_hid, 1)  # position-wise
        self.w_2 = nn.Conv1d(d_inner_hid, d_out, 1)  # position-wise
        self.layer_norm = LayerNormalization(d_out)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        output = self.relu(self.w_1(x.transpose(1, 2)))
        output = self.w_2(output).transpose(2, 1)
        output = self.dropout(output)
        if self.use_residual:
            output = output + residual
        return self.layer_norm(output)


class TransformerLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner=None, n_head=8, d_k=None, d_v=None, dropout=0.1):
        super(TransformerLayer, self).__init__()
        d_inner = d_inner or 4 * d_model
        d_k = d_k or d_model
        d_v = d_v or d_model
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, dec_input, enc_output, self_attn_mask=None, dec_enc_attn_mask=None):
        dec_output = self.slf_attn(dec_input, attn_mask=self_attn_mask)
        dec_output = self.enc_attn(dec_output, enc_output, enc_output, attn_mask=dec_enc_attn_mask)
        dec_output = self.pos_ffn(dec_output)
        return dec_output


def get_sinusoid_encoding_table(n_position, d_hid, scale=2.0, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, float(scale) * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)], dtype='float32')
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)
