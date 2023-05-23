import os
import logging
import torch
import torch.nn as nn

from transformers import BertTokenizer, BertModel

here = os.path.dirname(os.path.abspath(__file__))


class SentenceRE(nn.Module):

    def __init__(self, hparams):
        super(SentenceRE, self).__init__()

        self.pretrained_model_path = hparams.pretrained_model_path or 'bert-base-chinese'
        self.embedding_dim = hparams.embedding_dim
        self.dropout = hparams.dropout
        self.tagset_size = hparams.tagset_size
        self.max_len = hparams.max_len
        self.hid_size = hparams.hid_size
        self.bert_model = BertModel.from_pretrained(self.pretrained_model_path)
        # 隐藏层
        self.conv_layer = nn.Conv1d(in_channels=self.embedding_dim, out_channels= self.embedding_dim, kernel_size=5)
        self.pool_layer = nn.AdaptiveAvgPool1d(1)
        self.att_weight = nn.Parameter(torch.randn(self.hid_size * 2, 1)) #[self.hidden_size,1]
        self.dense = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.projection = nn.Linear(self.embedding_dim * 2, self.embedding_dim * 2)
        self.att_projection = nn.Linear(self.hid_size * 2, self.embedding_dim * 2)
        self.lstm = nn.LSTM(input_size = self.embedding_dim * 2, hidden_size = self.hid_size, num_layers = 1, batch_first = True, bidirectional=True)
        self.gru = nn.GRU(input_size = self.embedding_dim * 2, hidden_size = self.hid_size, num_layers = 1, batch_first = True, bidirectional=True)
        self.drop = nn.Dropout(self.dropout)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        # 三合一
        self.norm = nn.LayerNorm(self.embedding_dim * 3)
        # 线性层输出预测
        self.hidden2tag = nn.Linear(self.embedding_dim * 3, self.tagset_size)

    def forward(self, token_ids, token_type_ids, attention_mask, e1_mask, e2_mask):
        
        batch_size = e1_mask.shape[0]
        len = e1_mask.shape[1]
        # 对于全局信息应该取平均值mask
        self.global_mask = torch.ones(batch_size, len, device='cuda:0')

        #冻结bert
        self.freeze_bert(self.bert_model, 0, 3)
        
        # 句子的embedding
        sequence_output, cls = self.bert_model(input_ids=token_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, return_dict=False)

        # 实体信息
        e1_avg = self.entity_average(sequence_output, e1_mask)
        e2_avg = self.entity_average(sequence_output, e2_mask)
        e1_avg = self.tanh(self.dense(e1_avg))
        e2_avg = self.tanh(self.dense(e2_avg))
        concat_e = torch.cat([e1_avg, e2_avg], dim = -1)
        concat_e = self.projection(concat_e)
        concat_e = self.relu(concat_e)

        e1_entity = self.get_entity(sequence_output, e1_mask, self.embedding_dim)
        e2_entity = self.get_entity(sequence_output, e2_mask, self.embedding_dim)
        concat_ent = torch.cat([e1_entity, e2_entity], dim = -1)
        #rnn_out, _ = self.gru(concat_ent)
        rnn_out, _ = self.lstm(concat_ent)
        rnn_out = self.drop(rnn_out)
        att_out = self.tanh(self.attn(self, rnn_out))
        att_out = att_out.squeeze(1)
        att_out = self.att_projection(att_out)
        att_out = self.tanh(att_out)

        #全局信息的选取
        conv_pooling = self.conv_layer(sequence_output.transpose(1,2))
        conv_pooling = self.tanh(conv_pooling)
        conv_pooling = self.pool_layer(conv_pooling)
        conv_pooling = conv_pooling.view(batch_size, self.embedding_dim)

        avg_pooling = self.entity_average(sequence_output, self.global_mask)
        global_h = avg_pooling
        #global_h = cls
        #global_h = conv_pooling

        #实体信息的选取
        entity_info = att_out
        #entity_info = concat_e
        #entity_info = torch.cat([e1_avg, e2_avg], dim = -1)

        # 全局信息 + 实体信息
        concat_h = torch.cat([global_h, entity_info], dim = -1)
        #concat_h = torch.cat([global_h, e1_avg, e2_avg], dim = -1)
        concat_h = self.norm(concat_h)

        #预测输出
        logits = self.hidden2tag(self.drop(concat_h))

        return logits
 
    @staticmethod
    def entity_average(hidden_output, e_mask):
        """
        对mask区域的向量组求平均值
        返回值形状: [batch_size, dim]
        """
        #把需要标记的打上mask
        e_mask_unsqueeze = e_mask.unsqueeze(1)  # [b, 1, j-i+1]
        length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1]

        sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)  # [b, 1, j-i+1] * [b, j-i+1, dim] = [b, 1, dim] -> [b, dim]
        avg_vector = sum_vector.float() / length_tensor.float()  
        return avg_vector
    
    @staticmethod
    def get_entity(hidden_output, e_mask, embed):
        e_mask = e_mask.unsqueeze(-1).repeat(1,1,embed)
        entity = e_mask * hidden_output

        return entity
    
    @staticmethod
    def attn(self,H):
        M = torch.tanh(H)  # 非线性变换 size:(batch_size,seq_len, hidden_dim)
        a = nn.functional.softmax(torch.matmul(M, self.att_weight),dim=1)# a.Size : (batch_size,seq_len, 1),注意力权重矩阵
        a = torch.transpose(a,1,2)  # (batch_size,1, seq_len)
        return torch.bmm(a,H)  # (batch_size,1,hidden_dim) #权重矩阵对输入向量进行加权计算
    
    @staticmethod
    def freeze_bert(model, start, end):
        freeze_layers = model.encoder.layer[start : end]
        for layer in freeze_layers:
            for param in layer.parameters():
                param.requires_grad = False