import torch

#Offers pre-built layers like nn.Linear, nn.Conv2d, and nn.ReLU
#Creating, training, and evaluating neural networks in PyTorch.
#It has Loss function nn.CrossEntropyLoss,nn.MSELoss
import torch.nn as nn 
# Module for optimiser
import torch.optim as optim
import math

# Multihead Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
 
        # Size of model dimension
        self.d_model = d_model
        # No. of Attention heads ->  Different Aspects of Relationships
        self.num_heads = num_heads
        # Size of each Dimension Head's of keys,query,values
        self.d_k = d_model // num_heads
 
        # Assertion Error
        assert (d_model % num_heads ==0),"Model dimension needs to be divisible by heads"
        # nn.Linear ->Fully Connected Neural Network
        # value -> actual information that is passed through the network.
        # Value matrix
        self.W_v = nn.Linear(d_model, d_model)
        # key -> position of each word in the sequence.
        # Key matrix
        self.W_k = nn.Linear(d_model, d_model)
        # query -> which parts of the input to focus on.
        # Query matrix
        self.W_q = nn.Linear(d_model, d_model)
        # Output transformation or Linear Transformation or Scale to get back to it original form
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Matrix Multiplication of Query and K Transpose and divding by root of dk
        # (QK^T/√dk)
        a=math.sqrt(self.d_k)
        attention = torch.matmul(Q, K.transpose(-2, -1)) / a
        
        # To assign low weight to attention score
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e20)
        
        # softmax(QK^T/√dk)
        attention_result = torch.softmax(attention, dim=-1)
        
        # Final step to get Attention(Q, K, V ) = softmax(QKT/√dk)V
        output = torch.matmul(attention_result, V)
        return output
    
    def reshape1(self, x):
        # Reshapping the d_model or x.size() into self.heads different pieces
        batch_size, seq_length, d_model = x.size()
        #           x[batch_size, self.heads, seq_length, self.d_k]
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
    
    def attention_head(self, x):
        # Combine the multiple attention heads back to original shape
        batch_size, num_heads, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
    
    def forward(self, Q, K, V, mask=None):
        # Apply linear transformations and split heads
        Q = self.reshape1(self.W_q(Q))
        K = self.reshape1(self.W_k(K))
        V = self.reshape1(self.W_v(V))
        
        # Perform scaled dot-product attention
        attention_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Combine heads and apply output transformation to get back to it original size (batch_size, seq_length, d_model)
        # MultiHead(Q, K, V ) = Concat(head1, ..., headh)WO
        output = self.W_o(self.attention_head(attention_output))
        return output
    
#Feed Forward Network
class PositionWiseFeedForward(nn.Module):
    # d_ff -> dimension of Feed Forward Network
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fcnn1 = nn.Linear(d_model, d_ff)
        self.fcnn2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        # FFN(x) = max(0, xW1 + b1)W2 + b2
        return self.fcnn2(self.relu(self.fcnn1(x)))

#Positional Encoding    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        
        # Intialize all the position encoding tensor values to 0
        self.encoding = torch.zeros(max_seq_length, d_model)
        self.encoding.requires_grad = False
        #          Position indices from 0 to max_seq_length    in 2 dimension Expand 1 dimension
        position = torch.arange(0, max_seq_length).unsqueeze(1).float()

        #                      10000^2i/dmodel
        div_term = (torch.pow(10000.0, torch.arange(0, d_model, 2).float())/ d_model)
        # Even -> sin position encoding
        self.encoding[:, 0::2] = torch.sin(position / div_term)
        # Odd -> cos position encoding
        self.encoding[:, 1::2] = torch.cos(position / div_term)
        
        self.encoding = self.encoding.unsqueeze(0)
        #It is only helpful when we use GPU it easily move it to cuda so we used register_buffer 
        # self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        device = x.device
        #Position Embedding
        # (Batch, max_seq_length, d_model)
        # (max_seq_length, d_model)
        pe = self.encoding.to(device)
        x = x + pe[:, :x.size(1), :]
        return x
    
#Encoder Layer   
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

# Decoder Layer      
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x

#Transformer
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        #Boolean mask where each element value is True else False
        #Adding new dimension at 1 and then at 2
        device = src.device 
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2).to(device)

        #Boolean mask where each element value is True else False
        #Adding new dimension at 1 and then at 3
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3).to(device)
        seq_length = tgt.size(1)
        # torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1) -> Intialize all the tensor values above the diagonal excluding the diagonal to 1
        # torch.triu ( , diagonal=1) -> upper triangular matrix
        # Change the all the values from 1->0 and 0->1
        # Finnally change it to Boolean Values
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length, device=device), diagonal=1)).bool()
        # Both the boolean values get are combine with & to get where the true value is there for both
        tgt_mask = tgt_mask & nopeak_mask
        
        return src_mask, tgt_mask
    
    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output
    