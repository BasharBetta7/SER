from imports import *
from utils import *

# here we will extract a state from the 9th encoder block as it has the best phoneme-level ecnodings [cite from cross corpus wav2vec]
class Wav2vec2Encoder(nn.Module):
    def __init__(self, config="facebook/wav2vec2-base"):
        super().__init__()
        self.wav2vec = Wav2Vec2Model.from_pretrained(config, output_hidden_states=True,                                                   return_dict=True, apply_spec_augment=False)
        self.wav2vec.feature_extractor._freeze_parameters()

    def forward(self, input_audio):
        # input (B, T)
        output = self.wav2vec(input_audio)
        # (B, T, C)
        return output.hidden_states[9]



class CLS_Head(nn.Module):
    def __init__(self, hidden_size, n_labels):
        super().__init__()
        self.repr = Wav2vec2Encoder()
        self.proj = nn.Linear(768, hidden_size)
        self.cls = nn.Linear(hidden_size, n_labels)
        self.act = nn.Tanh()
        

    
    def forward(self, x, targets):
        # x : (B,t), target: (B)
        repr = self.repr(x) # (B, T, 768)
        proj = self.proj(repr) # (B,T,hidden_size)
        proj = self.act(proj) # (B,T,hidden_size)
        logits = self.cls(proj) # (B,T,n_labels)

        # avg pooling layer
        logits = logits.mean(1).squeeze(1) # (B,n_labels)
        
        B,C = logits.shape
        loss = F.cross_entropy(logits, targets)
        return logits, loss
        
        

# we will implement multihead self attention to add it to the model.
class MultiHeadAttention(nn.Module):
    '''takes an input tensor of shape (B,T,input_dim)'''
    def __init__(self, input_dim, embed_dim, n_heads):
        super().__init__()
        assert(embed_dim % n_heads == 0), 'embed_dim does not divide n_heads!'
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.qkv_proj = nn.Linear(input_dim, 3*embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.embed_dim = embed_dim



    def scaled_dot_product(self,query,key,value, mask=None):
        attn = (query @ key.transpose(-1,-2)) * (self.head_dim**(-0.5))# (B,head,T,head_dim) @ (B, head, head_dim, T) --> (B, head, T, T)
        if mask is not None: 
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1) 
        values = attn @ value # (B,head,T,T) @ (B,head, T, head_dim) -->  (B,head, T,head_dim)
        return attn, values
        
    def forward(self, x, return_attention=False, mask=None):
        ''' x : (B,T,input_dim)'''
        qkv = self.qkv_proj(x) #(B,T,3*embed_dim)
        B,T,_ = qkv.shape
        qkv = qkv.reshape(B,T,self.n_heads,3 * self.head_dim) #(B,T,head, 3 * head_dim)
        qkv = qkv.permute(0,2,1,3) # (B,head, T, 3*head_dim)
        q,k,v = qkv.chunk(3, dim=-1) # each (B, head, T, head_dim)
        
        attention, values = self.scaled_dot_product(q,k,v, mask=mask) #(B,head,T,head_dim)
        values = values.permute(0,2,1,3) #(B,T,head, head_dim)
        values = values.reshape(B,T,self.embed_dim)  #(B, T head* head_dim) === (B, T, embed_dim)
        out = self.out_proj(values)
        if return_attention:
            return out, attention
        else:
            return out
        
       
            
class EncoderBlock(nn.Module):
    def __init__(self, input_dim, ff_embed_dim, n_heads, dropout=0.0):
        super().__init__()
        self.attn = MultiHeadAttention(input_dim, input_dim, n_heads)
        self.linear_layers = nn.Sequential(
            nn.Linear(input_dim, ff_embed_dim),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(ff_embed_dim, input_dim)
        )
            
        self.ln1 = nn.LayerNorm(input_dim)
        self.ln2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        '''(B, T, input_dim)'''
        out = x + self.dropout(self.attn(x)) #(B,T, input_dim)
        out = self.ln1(out)
        out = out + self.dropout(self.linear_layers(out))
        out = self.ln2(out)
        return out #(B,T,input_dim)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        '''(B, T, input_dim)'''
        x = x + self.pe[:, :x.size(1)]
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, num_encoders, input_dim, ff_embed_dim, n_heads, dropout=0.0):
        super().__init__()
        self.pos_embed = PositionalEncoding(input_dim)
        self.encoders = nn.ModuleList([EncoderBlock(input_dim=input_dim, ff_embed_dim=ff_embed_dim, n_heads=n_heads, dropout=dropout) for _ in range(num_encoders)])

    def forward(self, x):
        x = self.pos_embed(x)

        for layer in self.encoders:
            x = layer(x)
            
        out = x
        return out



class Wav2vecSelfAttention(nn.Module):
    '''converts raw audio batch into classes'''
    def __init__(self, num_encoder_blocks, input_dim, n_heads, ff_embed_dim, n_labels=4, dropout=0.0, return_attentions=False):
        super().__init__()
        self.feature_extractor = Wav2vec2Encoder()
        self.positional_encoding = PositionalEncoding(input_dim)
        self.transformer = TransformerEncoder(num_encoder_blocks, input_dim, ff_embed_dim, n_heads, dropout)
        self.linear = nn.Linear(input_dim, n_labels)

    def freeze(self):
        for p in self.feature_extractor.parameters():
            p.requires_grad=False

        return sum([p.numel() for p in self.feature_extractor.parameters() if p.requires_grad == False])
        
    def forward(self, x, y):
        ''' x: (B,1,t), y: (B)'''
        x = self.feature_extractor(x) #(B, T, d)
        x = x + self.positional_encoding(x) #(B, T, d)
        x = self.transformer(x) # (B,T,d)
        x = x.mean(1) #(B, d)
        logits = self.linear(x) #(B, n_labels)        
        loss = F.cross_entropy(logits, y)
        return logits, loss


        
        

class TransformerEncoderBLock(nn.Module):
    '''converts raw audio batch into classes'''
    def __init__(self, num_encoder_blocks, input_dim, n_heads, ff_embed_dim, n_labels=4, dropout=0.3, return_attentions=False):
        super().__init__()
        self.positional_encoding = PositionalEncoding(input_dim)
        self.transformer = TransformerEncoder(num_encoder_blocks, input_dim, ff_embed_dim, n_heads, dropout)
        self.linear = nn.Linear(input_dim, n_labels)

    def freeze(self):
        for p in self.feature_extractor.parameters():
            p.requires_grad=False

        return sum([p.numel() for p in self.feature_extractor.parameters() if p.requires_grad == False])
        
    def forward(self, x, y):
        ''' x: (B,1,t), y: (B)'''
        x = self.feature_extractor(x) #(B, T, d)
        x = x + self.positional_encoding(x) #(B, T, d)
        x = self.transformer(x) # (B,T,d)
        x = x.mean(1) #(B, d)
        logits = self.linear(x) #(B, n_labels)        
        loss = F.cross_entropy(logits, y)
        return logits, loss


        
        

# BiLSTM for MFCC :

class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, 
                            hidden_size=hidden_dim, 
                            num_layers=2, 
                            batch_first=True, 
                            dropout=dropout, 
                            bidirectional=True)
    def forward(self, mfcc):
        '''(B, T, n_mfcc)'''
        mfcc = F.normalize(mfcc, dim= -1) # (B, T, n_mfcc) normalized over n_mfcc dimension
        out, _ = self.lstm(mfcc) # (B, T , 2 * hidden_dim)
        return out # (B, 2 * hidden_dim)  


        


# Co-attention block
class CoAttention(nn.Module):
    def __init__(self, input_dim_q, input_dim_kv, embed_dim, out_dim, n_heads):
        super().__init__()
        assert(embed_dim % n_heads == 0), 'embed_dim does not divide n_heads!'
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.embed_dim = embed_dim
        self.query = nn.Linear(input_dim_q, embed_dim)
        self.key = nn.Linear(input_dim_kv, embed_dim)
        self.value = nn.Linear(input_dim_kv, embed_dim)
        self.proj = nn.Linear(embed_dim, out_dim)
      

    def scaled_dot_product(self,query,key,value, mask=None):
        attn = (query @ key.transpose(-1,-2)) * (self.head_dim**(-0.5))# (B,head,T1,head_dim) @ (B, head, head_dim, T2) --> (B, head, T1, T2)
        if mask is not None: 
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1) 
        values = attn @ value # (B,head,T1,T2) @ (B,head, T2, head_dim) -->  (B,head, T1,head_dim)
        return attn, values

    def forward(self, x, y, return_attention=False):
        '''x: (B, T1, input_dim_q), y: (B, T2, input_dim_kv)'''
        query = self.query(x) #(B,T1, embed_dim)
        key = self.key(y) #(B, T2, embed_dim)
        value = self.value(y) #(B,T2, embed_dim)
        # split to multiple heads:
        B,T1,_ = query.shape
        B,T2,_ = key.shape
        query = query.reshape(B, T1, self.n_heads, self.head_dim) # (B,T1,n_heads, head_dim)
        key = key.reshape(B, T2, self.n_heads, self.head_dim) # (B, T2, n_heads, head_dim)
        value = value.reshape(B, T2, self.n_heads, self.head_dim) #(B, T2, n_heads, head_dim) 

        # permute the shape to ( B, n_heads, T, head_dim)
        query = query.permute(0,2,1,3)
        key = key.permute(0,2,1,3)
        value = value.permute(0,2,1,3)

        # calualte scaled_dot_product : 
        attention, values = self.scaled_dot_product(query, key, value) # values : (B, n_heads, T1, head_dim)
        values = values.permute(0,2,1,3) # (B, T1, n_heads, head_dim)
        values = values.reshape(B, T1, self.n_heads * self.head_dim) # (B, T1, embed_dim)
        
        # projection layer :
        values = self.proj(values) #(B, T1, out_dim)
        
        
        if return_attention:
            return values, attention
        else:
            return values
        


            
class CoAttentionEncoderBlock(nn.Module):
    def __init__(self, input_dim_q,  input_dim_kv ,embed_dim, ff_embed_dim, n_heads, dropout=0.0):
        super().__init__()
        self.attn = CoAttention(input_dim_q, input_dim_kv,embed_dim, input_dim_q, n_heads)
        self.linear_layers = nn.Sequential(
            nn.Linear(input_dim_q, ff_embed_dim),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(ff_embed_dim, input_dim_q)
        )
            
        self.ln1 = nn.LayerNorm(input_dim_q)
        self.ln2 = nn.LayerNorm(input_dim_q)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y, return_attention=False):
        '''x : (B, T1, input_dim_q), y: (B, T2, input_dim_kv)'''
        attentions = None
        if return_attention:
            values, attentions = self.attn(x, y, return_attention) #(B, T1, embed_dim) , (B, T1, T2)
        else:
            values = self.attn(x, y)  
        out = x + self.dropout(values) #(B,T1, embed_dim)
       
        out = self.ln1(out)
        out = out + self.dropout(self.linear_layers(out)) #(B,T1, embed_dim)
        out = self.ln2(out)
        if return_attention:
            return out, attentions
        return out #(B,T1, embed_dim)

    
    
    #######################
class WavLMEncoder(nn.Module):
    '''outputs a weighted average of hidden states as audio representation'''
    def __init__(self, config='microsoft/wavlm-base-plus', num_hidden_states=12):
        super().__init__()
        self.wavlm = WavLMModel.from_pretrained(config)
        self.weights =nn.Parameter(torch.ones((num_hidden_states,1)))
        

    def forward(self, input_audio):
        output = self.wavlm(input_audio, output_hidden_states=True).last_hidden_state #(B,T, 768)
        # output_cat = torch.stack(output, dim=-1) #(B,T,d,num_hidden)
        # avg = (output_cat @ self.weights) / self.weights.sum(dim=0) # (B,T,d,num_hidden) @ (num_hidden, 1) = (B,T,d,1)
        # avg = avg.squeeze(dim=-1) #(B,T,d)
        return output


class SSLModule(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.encoder = Wav2vec2Encoder()
        self.activation = nn.Tanh()
        self.linaer = nn.Linear(768, output_dim)
        
    def forward(self, x_wav):
        out = self.encoder(x_wav) #(B, T, 768)
        out = self.activation(out) #(B, T, 768)
        out = self.linaer(out) #(B,T, output_dim)
        return out
        
        
class FeatureEncoderModule(nn.Module):
    def __init__(self, n_mfcc, input_dim_mfcc, dropout=0.1):
        super().__init__()
        self.blstm = BiLSTM(n_mfcc, input_dim_mfcc, 0, dropout=dropout)
        self.linear = nn.Linear(2 * input_dim_mfcc, input_dim_mfcc)
        self.attention = TransformerEncoder(num_encoders=1, input_dim=input_dim_mfcc, ff_embed_dim=1024, n_heads=1)
        
    def forward(self, x_mfcc):
        out = self.blstm(x_mfcc) #(B, T, 2 * input_dim_mfcc)
        out = self.linear(out) #(B,T, input_dim_mfcc)
        out = self.attention(out) #(B,T, input_dim_mfcc) with attention 
        return out
        

class SER2_transformer_block(nn.Module):
    def __init__(self, n_mfcc, input_dim_mfcc, input_dim_wav, n_heads, embed_dim, n_labels=4):
        super().__init__()
        assert input_dim_mfcc == input_dim_wav, 'number of ecnoding dimensions must be the same between mfcc and wav2vec'


        # SSL_module:
        self.ssl_module = SSLModule(input_dim_wav)
        
        #Feature ENcdoer MOdule:
        self.feature_encoder_module = FeatureEncoderModule(n_mfcc, input_dim_mfcc, dropout=0.1)
       
        # mfcc_encoding layer:
        self.mfcc_encoder = BiLSTM(n_mfcc, input_dim_mfcc, 0, dropout=0.1)


        # mfcc linear layer:
        self.mfcc_ff = nn.Linear(2 * input_dim_mfcc, input_dim_mfcc)
        self.mfcc_att = TransformerEncoder(num_encoders=1, input_dim=input_dim_mfcc,ff_embed_dim=1024,n_heads=1)
        # add self attention for mfcc later 
        # self.mfcc_mhsa = MultiHeadAttention(input_dim_mfcc, embed_dim, n_heads)

        # co-attetnion layer between mfcc and wav encodings
        self.coatt = CoAttentionEncoderBlock(input_dim_mfcc, input_dim_wav, embed_dim, 1024, n_heads, 0.3)
        # self.coatt_addnorm = AttentionOutputLayer(embed_dim, dropout=0.0)
        
        # classification head 
        self.cls_head= nn.Linear(input_dim_mfcc, n_labels)

    def forward(self, x_wav, x_mfcc, y):
        '''x_wav: (B, 1, T), x_mfcc:(B, T2, n_mfcc), y:(B)'''
        
       
        wav_enc_lin = self.ssl_module(x_wav) # (B, T1, input_dim_wav)
        mfcc_enc_att = self.feature_encoder_module(x_mfcc) #(B, T2, input_dim_mfcc) with attention 


        mfcc_aware_wav, attention_scores = self.coatt(mfcc_enc_att, wav_enc_lin, return_attention=True) # (B, T2, input_dim_mfcc), (B, T2, T1)
        #mfcc_aware_wav_addnorm = self.coatt_addnorm(mfcc_aware_wav, mfcc_enc_lin) #(B,T2,embed_dim)
        
        
        logits = self.cls_head(mfcc_aware_wav) #(B, T2, n_labels)
      
        logits = logits.mean(1) # (B, n_labels)
        loss = F.cross_entropy(logits, y)

        #print(f"model logs:\n wav_env: {wav_enc.shape}| mfcc_enc: {mfcc_enc.shape}\n wav_enc_lin: {wav_enc_lin.shape}| mfcc_enc_lin: {mfcc_enc_lin.shape}\n mfcc_aware_wav: {mfcc_aware_wav.shape}| attention_scores: {attention_scores.shape}")
        return logits, loss
    
    
    
        