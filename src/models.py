from imports import *


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
            nn.ReLU(inplace=True),
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
        return x



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
    def __init__(self, input_dim_q, input_dim_kv, embed_dim, n_heads):
        super().__init__()
        assert(embed_dim % n_heads == 0), 'embed_dim does not divide n_heads!'
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.embed_dim = embed_dim
        self.query = nn.Linear(input_dim_q, embed_dim)
        self.key = nn.Linear(input_dim_kv, embed_dim)
        self.value = nn.Linear(input_dim_kv, embed_dim)
      

    def scaled_dot_product(self,query,key,value, mask=None):
        attn = (query @ key.transpose(-1,-2)) * (self.head_dim**(-0.5))# (B,head,T1,head_dim) @ (B, head, head_dim, T2) --> (B, head, T1, T2)
        if mask is not None: 
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1) 
        values = attn @ value # (B,head,T1,T2) @ (B,head, T2, head_dim) -->  (B,head, T1,head_dim)
        return attn, values

    def forward(self, x, y, return_attention=False):
        '''x: (B, T1, input_dim_q), y: (B, T2, input_dim_kv)'''
        query = self.query(x)
        key = self.key(y)
        value = self.value(y)
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

        # add & norm 
        query = query.permute(0,2,1,3) # (B, T1, n_heads, head_dim)
        
        
        if return_attention:
            return values, attention
        else:
            return values
        


class AttentionOutputLayer(nn.Module):
    def __init__(self, hidden_dim, dropout=0.0, bias=False):
        '''hidden_dim: dimension of weighted values'''
        super().__init__()
        self.dense = nn.Linear(hidden_dim, hidden_dim, bias=bias) # bias is false because we have layer norm 
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values, input):
        '''values: (B, T, hidden_dim), input:(B, T, hidden_dim)'''
        assert values.shape == input.shape, 'co-attention values dimension does not equal to input dimension!'
        
        hidden_states= self.dense(values)
        output = self.layer_norm(self.dropout(hidden_states) + input) # add & norm layer
        return output
    


class SER(nn.Module):
    def __init__(self, n_mfcc, input_dim_mfcc, input_dim_wav, n_heads, embed_dim, n_labels=4):
        super().__init__()
        assert input_dim_mfcc == input_dim_wav, 'number of ecnoding dimensions must be the same between mfcc and wav2vec'

        #wav2vec enocding layer:
        self.wav2vec_encoder = Wav2vec2Encoder()

        # mfcc_encoding layer:
        self.mfcc_encoder = BiLSTM(n_mfcc, input_dim_mfcc, 0, dropout=0.3)

        # wav2vec linear layer:
        self.wav2vec_ff = nn.Linear(768, input_dim_wav)

        # mfcc linear layer:
        self.mfcc_ff = nn.Linear(2 * input_dim_mfcc, input_dim_mfcc)
        self.mfcc_att = TransformerEncoder(num_encoders=1, input_dim=input_dim_mfcc,ff_embed_dim=1024,n_heads=1)
        # add self attention for mfcc later 
        # self.mfcc_mhsa = MultiHeadAttention(input_dim_mfcc, embed_dim, n_heads)

        # co-attetnion layer between mfcc and wav encodings
        self.coatt = CoAttention(input_dim_mfcc, input_dim_wav,embed_dim, n_head)
        # self.coatt_addnorm = AttentionOutputLayer(embed_dim, dropout=0.0)
        
        # classification head 
        self.cls_head= nn.Linear(embed_dim, n_labels)

    def forward(self, x_wav, x_mfcc, y):
        '''x_wav: (B, 1, T), x_mfcc:(B, T2, n_mfcc), y:(B)'''
        wav_enc = self.wav2vec_encoder(x_wav) # (B, T1, 768)
        mfcc_enc = self.mfcc_encoder(x_mfcc) # (B, T2, input_dim_mfcc)

        wav_enc_lin = self.wav2vec_ff(wav_enc) # (B, T1, input_dim_wav)
        mfcc_enc_lin = self.mfcc_ff(mfcc_enc) # (B, T2, input_dim_mfcc)
        mfcc_enc_att = self.mfcc_att(mfcc_enc_lin) # (B, T2, input_dim_mfcc) with attention 

        mfcc_aware_wav, attention_scores = self.coatt(mfcc_enc_att, wav_enc_lin, return_attention=True) # (B, T2, embed_dim), (B, T2, T1)
        #mfcc_aware_wav_addnorm = self.coatt_addnorm(mfcc_aware_wav, mfcc_enc_lin) #(B,T2,embed_dim)
        
        
        logits = self.cls_head(mfcc_aware_wav) #(B, T2, n_labels)
        logits = logits.mean(1) # (B, n_labels)
        loss = F.cross_entropy(logits, y)

        #print(f"model logs:\n wav_env: {wav_enc.shape}| mfcc_enc: {mfcc_enc.shape}\n wav_enc_lin: {wav_enc_lin.shape}| mfcc_enc_lin: {mfcc_enc_lin.shape}\n mfcc_aware_wav: {mfcc_aware_wav.shape}| attention_scores: {attention_scores.shape}")
        return logits, loss
        



class SER2(nn.Module):
    def __init__(self, n_mfcc, input_dim_mfcc, input_dim_wav, n_heads, embed_dim, n_labels=4):
        super().__init__()
        assert input_dim_mfcc == input_dim_wav, 'number of ecnoding dimensions must be the same between mfcc and wav2vec'

        #wav2vec enocding layer:
        self.wav2vec_encoder = Wav2vec2Encoder()

        # mfcc_encoding layer:
        self.mfcc_encoder = BiLSTM(n_mfcc, input_dim_mfcc, 0, dropout=0.1)

        # wav2vec linear layer:
        self.wav2vec_ff = nn.Linear(768, input_dim_wav)

        # mfcc linear layer:
        self.mfcc_ff = nn.Linear(2 * input_dim_mfcc, input_dim_mfcc)
        self.mfcc_att = TransformerEncoder(num_encoders=1, input_dim=input_dim_mfcc,ff_embed_dim=1024,n_heads=1)
        # add self attention for mfcc later 
        # self.mfcc_mhsa = MultiHeadAttention(input_dim_mfcc, embed_dim, n_heads)

        # co-attetnion layer between mfcc and wav encodings
        self.coatt = CoAttention(input_dim_mfcc, input_dim_wav,embed_dim, n_heads)
        # self.coatt_addnorm = AttentionOutputLayer(embed_dim, dropout=0.0)
        
        # classification head 
        self.cls_head= nn.Linear(embed_dim, n_labels)

    def forward(self, x_wav, x_mfcc, y):
        '''x_wav: (B, 1, T), x_mfcc:(B, T2, n_mfcc), y:(B)'''
        wav_enc = F.tanh(self.wav2vec_encoder(x_wav)) # (B, T1, 768)
        mfcc_enc = self.mfcc_encoder(x_mfcc) # (B, T2, input_dim_mfcc)


       
        wav_enc_lin = self.wav2vec_ff(wav_enc) # (B, T1, input_dim_wav)
        mfcc_enc_lin = self.mfcc_ff(mfcc_enc) # (B, T2, input_dim_mfcc)
        mfcc_enc_att = self.mfcc_att(mfcc_enc_lin) # (B, T2, input_dim_mfcc) with attention 


        mfcc_aware_wav, attention_scores = self.coatt(mfcc_enc_att, wav_enc_lin, return_attention=True) # (B, T2, embed_dim), (B, T2, T1)
        #mfcc_aware_wav_addnorm = self.coatt_addnorm(mfcc_aware_wav, mfcc_enc_lin) #(B,T2,embed_dim)
        
        
        logits = self.cls_head(mfcc_aware_wav) #(B, T2, n_labels)


         ########## visualize output tensors to find abnormality ###############
        
        # legends = 'wav_enc,mfcc_enc,wav_enc_lin,mfcc_enc_lin,mfcc_aware_wav,logits'.split(sep=',')
        # plt.figure(figsize=(20,4))
        # t = wav_enc[0,0,:].detach().cpu()
        # hy,hx = torch.histogram(t, density=True)
        # plt.plot(hx[:-1].detach(), hy.detach())

        # t = mfcc_enc[0,0,:].detach().cpu()
        # hy,hx = torch.histogram(t, density=True)
        # plt.plot(hx[:-1].detach(), hy.detach())

        # t = wav_enc_lin[0,0,:].detach().cpu()
        # hy,hx = torch.histogram(t, density=True)
        # plt.plot(hx[:-1].detach(), hy.detach())

        # t = mfcc_enc_lin[0,0,:].detach().cpu()
        # hy,hx = torch.histogram(t, density=True)
        # plt.plot(hx[:-1].detach(), hy.detach())


        # t = mfcc_aware_wav[0,0,:].detach().cpu()
        # hy,hx = torch.histogram(t, density=True)
        # plt.plot(hx[:-1].detach(), hy.detach())

        # t = logits[0,0,:].detach().cpu()
        # hy,hx = torch.histogram(t, density=True)
        # plt.plot(hx[:-1].detach(), hy.detach())
        # plt.legend(legends)
        plt.show()

        ######################################################################

        logits = logits.mean(1) # (B, n_labels)
        loss = F.cross_entropy(logits, y)

        #print(f"model logs:\n wav_env: {wav_enc.shape}| mfcc_enc: {mfcc_enc.shape}\n wav_enc_lin: {wav_enc_lin.shape}| mfcc_enc_lin: {mfcc_enc_lin.shape}\n mfcc_aware_wav: {mfcc_aware_wav.shape}| attention_scores: {attention_scores.shape}")
        return logits, loss
        
        
        
class SER_MFCCConcat(nn.Module): # avg + avg_mfcc, std+std_mfcc
    def __init__(self, n_mfcc, input_dim_mfcc, input_dim_wav, n_heads, embed_dim, n_labels=4):
        super().__init__()
        assert input_dim_mfcc == input_dim_wav, 'number of ecnoding dimensions must be the same between mfcc and wav2vec'

        #wav2vec enocding layer:
        self.wav2vec_encoder = Wav2vec2Encoder()

        # mfcc_encoding layer:
        self.mfcc_encoder = BiLSTM(n_mfcc, input_dim_mfcc, 0, dropout=0.3)

        # wav2vec linear layer:
        self.wav2vec_ff = nn.Linear(768, input_dim_wav)

        # mfcc linear layer:
        self.mfcc_ff = nn.Linear(2 * input_dim_mfcc, input_dim_mfcc)
        self.mfcc_att = TransformerEncoder(num_encoders=2, input_dim=input_dim_mfcc,ff_embed_dim=1024,n_heads=1)
        # add self attention for mfcc later 
        # self.mfcc_mhsa = MultiHeadAttention(input_dim_mfcc, embed_dim, n_heads)

        # co-attetnion layer between mfcc and wav encodings
        self.coatt = CoAttention(input_dim_mfcc, input_dim_wav,embed_dim, n_heads)
        self.mfcc_lin = nn.Linear(input_dim_mfcc, embed_dim, bias=False)
        
        # self.coatt_addnorm = AttentionOutputLayer(embed_dim, dropout=0.0)
        
        # classification head 
        self.cls_head= nn.Linear(2 * embed_dim, n_labels)

    def forward(self, x_wav, x_mfcc, y):
        '''x_wav: (B, 1, T), x_mfcc:(B, T2, n_mfcc), y:(B)'''
        wav_enc = self.wav2vec_encoder(x_wav) # (B, T1, 768)
        mfcc_enc = self.mfcc_encoder(x_mfcc) # (B, T2, input_dim_mfcc)

        wav_enc_lin = self.wav2vec_ff(wav_enc) # (B, T1, input_dim_wav)
        mfcc_enc_lin = self.mfcc_ff(mfcc_enc) # (B, T2, input_dim_mfcc)
        mfcc_enc_att = self.mfcc_att(mfcc_enc_lin) # (B, T2, input_dim_mfcc) with attention 
        mfcc_enc_att_lin = self.mfcc_lin(mfcc_enc_att) # (B, T2, embed_dim)

        mfcc_aware_wav, attention_scores = self.coatt(mfcc_enc_att, wav_enc_lin, return_attention=True) # (B, T2, embed_dim), (B, T2, T1)
        #mfcc_aware_wav_addnorm = self.coatt_addnorm(mfcc_aware_wav, mfcc_enc_lin) #(B,T2,embed_dim)
        
        logits = torch.cat((mfcc_aware_wav.mean(1), mfcc_enc_att_lin.mean(1)), dim=-1) # (B, 2 * embed_dim)     
        
        logits = self.cls_head(logits) #(B, T2, n_labels)
        loss = F.cross_entropy(logits, y)

        #print(f"model logs:\n wav_env: {wav_enc.shape}| mfcc_enc: {mfcc_enc.shape}\n wav_enc_lin: {wav_enc_lin.shape}| mfcc_enc_lin: {mfcc_enc_lin.shape}\n mfcc_aware_wav: {mfcc_aware_wav.shape}| attention_scores: {attention_scores.shape}")
        return logits, loss
        
        
        
        
        
class SER3(nn.Module):
    def __init__(self, n_mfcc, input_dim_mfcc, input_dim_wav, n_heads, embed_dim, n_labels=4):
        super().__init__()
        assert input_dim_mfcc == input_dim_wav, 'number of ecnoding dimensions must be the same between mfcc and wav2vec'

        #wav2vec enocding layer:
        self.wav2vec_encoder = Wav2vec2Encoder()

        # mfcc_encoding layer:
        self.mfcc_encoder = BiLSTM(n_mfcc, input_dim_mfcc, 0, dropout=0.1)

        # wav2vec linear layer:
        self.wav2vec_ff = nn.Linear(768, input_dim_wav)

        # mfcc linear layer:
        self.mfcc_ff = nn.Linear(2 * input_dim_mfcc, input_dim_mfcc)
        self.mfcc_att = TransformerEncoder(num_encoders=1, input_dim=input_dim_mfcc,ff_embed_dim=1024,n_heads=1)
        # add self attention for mfcc later 
        # self.mfcc_mhsa = MultiHeadAttention(input_dim_mfcc, embed_dim, n_heads)

        # co-attetnion layer between mfcc and wav encodings
        self.coatt = CoAttention(input_dim_mfcc, input_dim_wav,embed_dim, n_heads)
        # self.coatt_addnorm = AttentionOutputLayer(embed_dim, dropout=0.0)
        
        # classification head 
        self.cls_head= nn.Linear(embed_dim, n_labels)

    def forward(self, x_wav, x_mfcc, y):
        '''x_wav: (B, 1, T), x_mfcc:(B, T2, n_mfcc), y:(B)'''
        wav_enc = F.tanh(self.wav2vec_encoder(x_wav)) # (B, T1, 768)
        mfcc_enc = self.mfcc_encoder(x_mfcc) # (B, T2, input_dim_mfcc)

       
        wav_enc_lin = self.wav2vec_ff(wav_enc) # (B, T1, input_dim_wav)
        mfcc_enc_lin = self.mfcc_ff(mfcc_enc) # (B, T2, input_dim_mfcc)
        mfcc_enc_att = self.mfcc_att(mfcc_enc_lin) # (B, T2, input_dim_mfcc) with attention 


        mfcc_aware_wav, attention_scores = self.coatt(mfcc_enc_att, wav_enc_lin, return_attention=True) # (B, T2, embed_dim), (B, T2, T1)
        #mfcc_aware_wav_addnorm = self.coatt_addnorm(mfcc_aware_wav, mfcc_enc_lin) #(B,T2,embed_dim)
        
        
        logits = self.cls_head(mfcc_aware_wav.mean(1)) #(B, n_labels)


         ########## visualize output tensors to find abnormality ###############
        
        # legends = 'wav_enc,mfcc_enc,wav_enc_lin,mfcc_enc_lin,mfcc_aware_wav,logits'.split(sep=',')
        # plt.figure(figsize=(20,4))
        # t = wav_enc[0,0,:].detach().cpu()
        # hy,hx = torch.histogram(t, density=True)
        # plt.plot(hx[:-1].detach(), hy.detach())

        # t = mfcc_enc[0,0,:].detach().cpu()
        # hy,hx = torch.histogram(t, density=True)
        # plt.plot(hx[:-1].detach(), hy.detach())

        # t = wav_enc_lin[0,0,:].detach().cpu()
        # hy,hx = torch.histogram(t, density=True)
        # plt.plot(hx[:-1].detach(), hy.detach())

        # t = mfcc_enc_lin[0,0,:].detach().cpu()
        # hy,hx = torch.histogram(t, density=True)
        # plt.plot(hx[:-1].detach(), hy.detach())


        # t = mfcc_aware_wav[0,0,:].detach().cpu()
        # hy,hx = torch.histogram(t, density=True)
        # plt.plot(hx[:-1].detach(), hy.detach())

        # t = logits[0,0,:].detach().cpu()
        # hy,hx = torch.histogram(t, density=True)
        # plt.plot(hx[:-1].detach(), hy.detach())
        # plt.legend(legends)
       

        ######################################################################

        loss = F.cross_entropy(logits, y)

        #print(f"model logs:\n wav_env: {wav_enc.shape}| mfcc_enc: {mfcc_enc.shape}\n wav_enc_lin: {wav_enc_lin.shape}| mfcc_enc_lin: {mfcc_enc_lin.shape}\n mfcc_aware_wav: {mfcc_aware_wav.shape}| attention_scores: {attention_scores.shape}")
        return logits, loss
    
    
    
class SER_shallow_fusion(nn.Module):
    def __init__(self, n_mfcc, input_dim_mfcc, input_dim_wav, n_heads, embed_dim, n_labels=4):
        super().__init__()
        assert input_dim_mfcc == input_dim_wav, 'number of ecnoding dimensions must be the same between mfcc and wav2vec'

        #wav2vec enocding layer:
        self.wav2vec_encoder = Wav2vec2Encoder()

        # mfcc_encoding layer:
        self.mfcc_encoder = BiLSTM(n_mfcc, input_dim_mfcc, 0, dropout=0.1)

        # wav2vec linear layer:
        self.wav2vec_ff = nn.Linear(768, input_dim_wav)

        # mfcc linear layer:
        self.mfcc_ff = nn.Linear(2 * input_dim_mfcc, input_dim_mfcc)
        self.mfcc_att = TransformerEncoder(num_encoders=1, input_dim=input_dim_mfcc,ff_embed_dim=1024,n_heads=1)
        # add self attention for mfcc later 
        # self.mfcc_mhsa = MultiHeadAttention(input_dim_mfcc, embed_dim, n_heads)

        # co-attetnion layer between mfcc and wav encodings
        self.coatt = CoAttention(input_dim_mfcc, input_dim_wav,embed_dim, n_heads)
        # self.coatt_addnorm = AttentionOutputLayer(embed_dim, dropout=0.0)
        
        # classification head 
        self.cls_head= nn.Linear(embed_dim, n_labels)

    def forward(self, x_wav, x_mfcc, y):
        '''x_wav: (B, 1, T), x_mfcc:(B, T2, n_mfcc), y:(B)'''
        wav_enc = F.tanh(self.wav2vec_encoder(x_wav)) # (B, T1, 768)
        mfcc_enc = self.mfcc_encoder(x_mfcc) # (B, T2, input_dim_mfcc)

       
        wav_enc_lin = self.wav2vec_ff(wav_enc) # (B, T1, input_dim_wav)
        mfcc_enc_lin = self.mfcc_ff(mfcc_enc) # (B, T2, input_dim_mfcc)
        mfcc_enc_att = self.mfcc_att(mfcc_enc_lin) # (B, T2, input_dim_mfcc) with attention 


        mfcc_aware_wav, attention_scores = self.coatt(mfcc_enc_att, wav_enc_lin, return_attention=True) # (B, T2, embed_dim), (B, T2, T1)
        #mfcc_aware_wav_addnorm = self.coatt_addnorm(mfcc_aware_wav, mfcc_enc_lin) #(B,T2,embed_dim)
        
        
        logits = self.cls_head(mfcc_aware_wav.mean(1)) #(B, n_labels)


         ########## visualize output tensors to find abnormality ###############
        
        # legends = 'wav_enc,mfcc_enc,wav_enc_lin,mfcc_enc_lin,mfcc_aware_wav,logits'.split(sep=',')
        # plt.figure(figsize=(20,4))
        # t = wav_enc[0,0,:].detach().cpu()
        # hy,hx = torch.histogram(t, density=True)
        # plt.plot(hx[:-1].detach(), hy.detach())

        # t = mfcc_enc[0,0,:].detach().cpu()
        # hy,hx = torch.histogram(t, density=True)
        # plt.plot(hx[:-1].detach(), hy.detach())

        # t = wav_enc_lin[0,0,:].detach().cpu()
        # hy,hx = torch.histogram(t, density=True)
        # plt.plot(hx[:-1].detach(), hy.detach())

        # t = mfcc_enc_lin[0,0,:].detach().cpu()
        # hy,hx = torch.histogram(t, density=True)
        # plt.plot(hx[:-1].detach(), hy.detach())


        # t = mfcc_aware_wav[0,0,:].detach().cpu()
        # hy,hx = torch.histogram(t, density=True)
        # plt.plot(hx[:-1].detach(), hy.detach())

        # t = logits[0,0,:].detach().cpu()
        # hy,hx = torch.histogram(t, density=True)
        # plt.plot(hx[:-1].detach(), hy.detach())
        # plt.legend(legends)
       

        ######################################################################

        loss = F.cross_entropy(logits, y)

        #print(f"model logs:\n wav_env: {wav_enc.shape}| mfcc_enc: {mfcc_enc.shape}\n wav_enc_lin: {wav_enc_lin.shape}| mfcc_enc_lin: {mfcc_enc_lin.shape}\n mfcc_aware_wav: {mfcc_aware_wav.shape}| attention_scores: {attention_scores.shape}")
        return logits, loss
    
    
    
    
    
    #######################
class WavLMEncoder(nn.Module):
    '''outputs a weighted average of hidden states as audio representation'''
    def __init__(self, config='microsoft/wavlm-base-plus', num_hidden_states=13):
        super().__init__()
        self.wavlm = WavLMModel.from_pretrained(config)
        self.weights =nn.Parameter(torch.ones((num_hidden_states,1)))
        

    def forward(self, input_audio):
        output = self.wavlm(input_audio, output_hidden_states=True).hidden_states
        output_cat = torch.stack(output, dim=-1) #(B,T,d,num_hidden)
        avg = (output_cat @ self.weights) / self.weights.sum(dim=0) # (B,T,d,num_hidden) @ (num_hidden, 1) = (B,T,d,1)
        avg = avg.squeeze(dim=-1) #(B,T,d)
        return avg



class SER_WavLM(nn.Module):
    def __init__(self, n_mfcc, input_dim_mfcc, input_dim_wav, n_heads, embed_dim, n_labels=4):
        super().__init__()
        assert input_dim_mfcc == input_dim_wav, 'number of ecnoding dimensions must be the same between mfcc and wavlm'

        #wavlm enocding layer:
        self.wavlm_encoder = WavLMEncoder()

        # mfcc_encoding layer:
        self.mfcc_encoder = BiLSTM(n_mfcc, input_dim_mfcc, 0, dropout=0.1)

        # wavlm linear layer:
        self.wavlm_ff = nn.Linear(768, input_dim_wav)

        # mfcc linear layer:
        self.mfcc_ff = nn.Linear(2 * input_dim_mfcc, input_dim_mfcc)
        self.mfcc_att = TransformerEncoder(num_encoders=1, input_dim=input_dim_mfcc,ff_embed_dim=1024,n_heads=1)
        # add self attention for mfcc later 
        # self.mfcc_mhsa = MultiHeadAttention(input_dim_mfcc, embed_dim, n_heads)

        # co-attetnion layer between mfcc and wav encodings
        self.coatt = CoAttention(input_dim_mfcc, input_dim_wav,embed_dim, n_heads)
        # self.coatt_addnorm = AttentionOutputLayer(embed_dim, dropout=0.0)
        
        # classification head 
        self.cls_head= nn.Linear(embed_dim, n_labels)

    def forward(self, x_wav, x_mfcc, y):
        '''x_wav: (B, 1, T), x_mfcc:(B, T2, n_mfcc), y:(B)'''
        wav_enc = F.tanh(self.wavlm_encoder(x_wav)) # (B, T1, 768)
        mfcc_enc = self.mfcc_encoder(x_mfcc) # (B, T2, input_dim_mfcc)

       
        wav_enc_lin = self.wavlm_ff(wav_enc) # (B, T1, input_dim_wav)
        mfcc_enc_lin = self.mfcc_ff(mfcc_enc) # (B, T2, input_dim_mfcc)
        mfcc_enc_att = self.mfcc_att(mfcc_enc_lin) # (B, T2, input_dim_mfcc) with attention 


        mfcc_aware_wav, attention_scores = self.coatt(mfcc_enc_att, wav_enc_lin, return_attention=True) # (B, T2, embed_dim), (B, T2, T1)
        #mfcc_aware_wav_addnorm = self.coatt_addnorm(mfcc_aware_wav, mfcc_enc_lin) #(B,T2,embed_dim)
        
        
        logits = self.cls_head(mfcc_aware_wav) #(B,T2, n_labels)
        logits = logits.mean(1) # (B, n_labels)  ## try weighted average layer ## 
        
       
        loss = F.cross_entropy(logits, y)
        #print(f"model logs:\n wav_env: {wav_enc.shape}| mfcc_enc: {mfcc_enc.shape}\n wav_enc_lin: {wav_enc_lin.shape}| mfcc_enc_lin: {mfcc_enc_lin.shape}\n mfcc_aware_wav: {mfcc_aware_wav.shape}| attention_scores: {attention_scores.shape}")
        return logits, loss