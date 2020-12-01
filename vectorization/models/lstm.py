import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from vectorization.modules.base import ParameterizedModule
from vectorization.modules._transformer_modules import get_sinusoid_encoding_table
from vectorization.models.common import ConvFeatureExtractor

# TODO @mvkolos, @artonson: derive the models classes from base; parameterize the model with reasonable layer sizes
class LSTMTagger(ParameterizedModule):
    def __init__(self, hidden_dim=128, ffn_dim=512, n_head=8, num_layers=10, input_channels=1, output_dim=5, resnet_count=0, **kwargs):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        #TODO rewrite to new standart
        self.conv = ConvFeatureExtractor(
            in_channels=input_channels, resnet_count=resnet_count, hidden_dim=hidden_dim,  **kwargs)

        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.hidden2tag = nn.Linear(2 * hidden_dim, output_dim)
        self.prefinal_fc = nn.Linear(hidden_dim, hidden_dim * 2)
        self.final_fc = nn.Linear(hidden_dim * 2, output_dim)
        self.final_tanh = nn.Tanh()
        self.final_sigm = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, images, n):
        img_code = self.conv(images)  # [b, c, h, w]
        img_seq = img_code.reshape([img_code.shape[0], img_code.shape[1], -1]).transpose(1, 2)  # [b, h * w , c]

        #         img_seq = img_seq.reshape(img_code.shape[0],-1)
        #         lstm = self.lstm(img_seq)
        # Set initial states
        h0 = torch.zeros(self.num_layers * 2, img_seq.size(0), self.hidden_dim).to(images.device)  # 2 for bidirection
        c0 = torch.zeros(self.num_layers * 2, img_seq.size(0), self.hidden_dim).to(images.device)

        # Forward propagate LSTM

        for it in range(n):
            out, (h0, c0) = self.lstm(img_seq, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
            if it == 0:
                output = self.final_sigm(self.hidden2tag(out[:, -1, :]))
                output = output[:, :, None]
            else:
                output = torch.cat((output, self.final_sigm(self.hidden2tag(out[:, -1, :]))[:, :, None]), dim=2)
        # Decode the hidden state of the last time step
        return output.transpose(1, 2)  # [b, n, output_dim]


class LSTMTagger_normal(ParameterizedModule):
    def __init__(self, hidden_dim=128, ffn_dim=512, n_head=8, num_layers=10, input_channels=1, output_dim=5, resnet_count=0, **kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.conv = ConvFeatureExtractor(
            in_channels=input_channels, resnet_count=resnet_count, hidden_dim=hidden_dim,  **kwargs)
        if resnet_count:
            self.lstm = nn.LSTM(93312, hidden_dim, num_layers, batch_first=True, bidirectional=True)

        else:
            self.lstm = nn.LSTM(21632, hidden_dim, num_layers, batch_first=True, bidirectional=True)

        self.hidden2tag = nn.Linear(2 * hidden_dim, output_dim)
        self.final_fc = nn.Linear(hidden_dim * 2, output_dim)
        self.final_tanh = nn.Tanh()
        self.final_sigm = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, images, n):
        img_code = self.conv(images)  # [b, c, h, w]
        img_seq = img_code.reshape([img_code.shape[0], img_code.shape[1], -1]).transpose(1, 2)  # [b, h * w , c]

        img_seq = img_seq.reshape(img_seq.shape[0], -1)  # [b, h * w * c]
        img_seq = img_seq.unsqueeze(1).expand(img_seq.shape[0], n, img_seq.shape[1])  # [b, n, h * w * c]

        # Set initial states
        h0 = torch.zeros(self.num_layers * 2, img_seq.size(0), self.hidden_dim).to(images.device)  # 2 for bidirection
        c0 = torch.zeros(self.num_layers * 2, img_seq.size(0), self.hidden_dim).to(images.device)

        # Forward propagate LSTM
        out, (h0, c0) = self.lstm(img_seq, (h0, c0))

        fc = self.hidden2tag(self.relu(out))
        coord = (self.final_tanh(fc[:, :, :-1]) + 1.) / 2.  # [b, n, output_dim-1]
        prob = self.final_sigm(fc[:, :, -1]).unsqueeze(-1) # [b, n, 1]
        return torch.cat((coord, prob), dim = -1)  # [b, n, output_dim]



class Attn(nn.Module):
    def __init__(self, method, hidden_size, input_size):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size + input_size, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

    def forward(self, hidden, encoder_outputs, src_len=None):
        '''
        :param hidden:
            previous hidden state of the decoder, in shape (layers*directions,B,H)
        :param encoder_outputs:
            encoder outputs from Encoder, in shape (T,B,H)
        :param src_len:
            used for masking. NoneType or tensor in shape (B) indicating sequence length
        :return
            attention energies in shape (B,T)
        '''
        max_len = encoder_outputs.size(0)
        this_batch_size = encoder_outputs.size(1)

        H = hidden.repeat(max_len, 1, 1).transpose(0, 1)

        encoder_outputs = encoder_outputs.transpose(0, 1)  # [B*T*H]
        attn_energies = self.score(H, encoder_outputs)  # compute attention score

        if src_len is not None:
            mask = []
            for b in range(src_len.size(0)):
                mask.append([0] * src_len[b].item() + [1] * (encoder_outputs.size(1) - src_len[b].item()))
            mask = torch.ByteTensor(mask).unsqueeze(1).to(encoder_outputs.device)  # [B,1,T]
            attn_energies = attn_energies.masked_fill(mask, -1e18)
        return F.softmax(attn_energies, dim=-1).unsqueeze(1)  # normalize with softmax

    def score(self, hidden, encoder_outputs):
        energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2)))  # [B*T*(H + I)] -> [B, T, H]
        energy = energy.transpose(2, 1)  # [B*H*T]
        v = self.v.repeat(encoder_outputs.data.shape[0], 1).unsqueeze(1)  # [B*1*H]
        energy = torch.bmm(v, energy)  # [B*1*T]
        return energy.squeeze(1)  # [B*T]


class ATTNLSTMCell(nn.Module):
    def __init__(self, hidden_size, input_size, encoder_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.attn = Attn('concat', hidden_size, encoder_size)
        self.lstm = nn.LSTMCell(encoder_size + input_size, hidden_size)

    def forward(self, current_input, last_hidden, encoder_outputs):
        # Calculate attention weights and apply to encoder outputs
        last_c, last_h = last_hidden
        attn_weights = self.attn(last_h, encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # (B,1,V)
        context = context[:, 0]
        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat((current_input, context), 1)
        # rnn_input = self.attn_combine(rnn_input) # use it in case your size of rnn_input is different
        hidden = self.lstm(rnn_input, last_hidden)
        return hidden


class ATTNLSTM(nn.Module):
    def __init__(self, hidden_size, input_size, encoder_size):
        super().__init__()
        self.cell = ATTNLSTMCell(hidden_size, input_size, encoder_size)

    def forward(self, input_emb, img_seq, prev_state=None):
        if prev_state is None:
            h0 = torch.zeros(img_seq.size(0), self.cell.hidden_size).to(img_seq.device)
            c0 = torch.zeros(img_seq.size(0), self.cell.hidden_size).to(img_seq.device)
            prev_state = (c0, h0)
        state_seq = []
        for emb_t in input_emb.transpose(0, 1):
            prev_state = self.cell(emb_t, prev_state, img_seq.transpose(0, 1))
            state_seq.append(prev_state)

        cell_seq, hid_seq = zip(*state_seq)

        return torch.stack(cell_seq, dim=1), torch.stack(hid_seq, dim=1)


class BidirectionalATTNLSTM(nn.Module):
    def __init__(self, hidden_size, input_size, encoder_size):
        super().__init__()
        self.lstm_fw = ATTNLSTM(hidden_size, input_size, encoder_size)
        self.lstm_bw = ATTNLSTM(hidden_size, input_size, encoder_size)

    def forward(self, input_emb, img_seq, prev_state_fw=None, prev_state_bw=None):
        cell_seq_fw, hid_seq_fw = self.lstm_fw(input_emb, img_seq, prev_state_fw)  # [batch, seq_len]
        rev_indices = torch.arange(input_emb.shape[1] - 1, -1, -1)
        cell_seq_bw, hid_seq_bw = self.lstm_bw(input_emb[:, rev_indices], img_seq, prev_state_bw)  # [batch, seq_len]
        cell_seq = torch.cat([cell_seq_fw, cell_seq_bw], dim=-1)
        hid_seq = torch.cat([hid_seq_fw, hid_seq_bw], dim=-1)
        return cell_seq, hid_seq


class LSTMTagger_attent(ParameterizedModule):
    def __init__(self, hidden_dim=128, ffn_dim=512, n_head=8, num_layers=10, input_channels=1, output_dim=6, resnet_count=0,
                 device ='cuda:1', **kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.conv = ConvFeatureExtractor(
            in_channels=input_channels, resnet_count=resnet_count, hidden_dim=hidden_dim,  **kwargs)

        self.lstms = [
            BidirectionalATTNLSTM(hidden_dim, hidden_dim * 2, hidden_dim).to(device),
            BidirectionalATTNLSTM(hidden_dim, hidden_dim * 2, hidden_dim).to(device),
        ]
        self.hidden2tag = nn.Linear(2 * hidden_dim, output_dim)
        self.prefinal_fc = nn.Linear(hidden_dim, hidden_dim * 2)
        self.final_fc = nn.Linear(hidden_dim * 2, output_dim)
        self.final_tanh = nn.Tanh()
        self.final_sigm = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, images, n):
        img_code = self.conv(images)  # [b, c, h, w]
        img_seq = img_code.reshape([img_code.shape[0], img_code.shape[1], -1]).transpose(1, 2)  # [b, h * w , c]
        # Forward propagate LSTM
        input_seq = get_sinusoid_encoding_table(n, self.hidden_dim * 2)[None].repeat(img_seq.shape[0], 1,
                                                                                     1)  # [b, n, i]
        input_seq = input_seq.to(images.device)
        for lstm in self.lstms:
            _, input_seq = lstm(input_seq, img_seq)

        # input_seq
        fc = self.final_fc(self.relu(input_seq))
        coord = (self.final_tanh(fc[:, :, :-1]) + 1.) / 2.  # [b, n, output_dim-1]
        prob = self.final_sigm(fc[:, :, -1]).unsqueeze(-1)
        return torch.cat((coord, prob), dim=-1)  # [b, n, output_dim]
