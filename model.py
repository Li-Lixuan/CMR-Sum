# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.

import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F

class LabelSmoothingNLLLoss(nn.Module):
    """NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0): # 平滑因子
        """Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingNLLLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, logprobs, target):
        # logprobs = torch.nn.functional.log_softmax(x, dim=-1) # x: (batch size * class数量)，即log(p(k))
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1)) # target: (batch size) 数字标签
        # 相当于取出logprobs中的真实标签的那个位置的logit的负值
        nll_loss = nll_loss.squeeze(1) # (batch size * 1)再squeeze成batch size，即log(p(k))δk,y，δk,y表示除了k=y时该值为1，其余为0
        smooth_loss = -logprobs.mean(dim=-1) # 在class维度取均值，就是对每个样本x的所有类的logprobs取了平均值。
        # smooth_loss = -log(p(k))u(k) = -log(p(k))∗ 1/k

        loss = self.confidence * nll_loss + self.smoothing * smooth_loss # (batch size)
        # loss = (1−ϵ)log(p(k))δk,y + ϵlog(p(k))u(k)
        return loss.mean() # −∑ k=1~K [(1−ϵ)log(p(k))δk,y+ϵlog(p(k))u(k)]


class Seq2Seq(nn.Module):
    """
        Build Seqence-to-Sequence.
        
        Parameters:

        * `encoder`- encoder of seq2seq model. e.g. roberta
        * `decoder`- decoder of seq2seq model. e.g. transformer
        * `config`- configuration of encoder model. 
        * `beam_size`- beam size for beam search. 
        * `max_length`- max length of target for beam search. 
        * `sos_id`- start of symbol ids in target for beam search.
        * `eos_id`- end of symbol ids in target for beam search. 
    """
    def __init__(self, encoder,decoder, config, beam_size=None, max_length=None, sos_id=None, eos_id=None, use_copy=True):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder=decoder
        self.config=config
        self.register_buffer(
            "bias", torch.tril(torch.ones((1024, 1024), dtype=torch.uint8)).view(1,1024, 1024)
        )
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

        self.lm_head = CopyTokenDecoder(config.vocab_size, config.hidden_size, config.intermediate_size, dropout=0.1)
        self.lm_head.output_projection.weight = self.encoder.embeddings.word_embeddings.weight
        self.mem_bias_scale = nn.Parameter(torch.ones(1))
        self.mem_bias_base = nn.Parameter(torch.zeros(1))
        
        self.lsm = nn.LogSoftmax(dim=-1)
        
        self.beam_size = beam_size
        self.max_length = max_length
        self.sos_id = sos_id
        self.eos_id = eos_id       
    
    
    def sentence_emb(self, nl_inputs):
        outputs = self.encoder(nl_inputs,attention_mask=nl_inputs.ne(1))[0]
        outputs = (outputs*nl_inputs.ne(1)[:,:,None]).sum(1)/nl_inputs.ne(1).sum(-1)[:,None]
        return torch.nn.functional.normalize(outputs, p=2, dim=1)
    
    
    def forward(self, source_ids, target_ids=None, 
                retrieval_source_ids=None, 
                retrieval_target_ids=None,
                return_sum_loss=True):   
        temperature = 0.2
        if target_ids is None:
            return self.generate(source_ids, retrieval_source_ids, retrieval_target_ids)
        
        mask = source_ids.ne(1)[:,None,:]*source_ids.ne(1)[:,:,None]
        encoder_output = self.encoder(source_ids,attention_mask=mask,use_cache=True)  
        
        loss_fct = nn.CrossEntropyLoss()
        # code2code loss
        code_feature = (encoder_output.last_hidden_state * source_ids.ne(1)[:,:,None]).sum(1) / source_ids.ne(1).sum(-1)[:,None]
        code_feature = torch.nn.functional.normalize(code_feature, p=2, dim=1)
        code_feature2 = self.sentence_emb(source_ids)
        c2c_scores = torch.einsum("ab,cb->ac", code_feature, code_feature2) 
        code2code_loss = loss_fct(c2c_scores / temperature, torch.arange(target_ids.size(0), device=c2c_scores.device))
        
        # code2nl loss
        nl_feature = self.sentence_emb(target_ids)
        c2nl_scores = torch.einsum("ab,cb->ac", code_feature, nl_feature) 
        code2nl_loss = loss_fct(c2nl_scores / temperature, torch.arange(target_ids.size(0), device=c2nl_scores.device))
        
        # kl divergence
        p_c2c = F.softmax(c2c_scores / temperature, dim=-1)
        p_c2nl = F.softmax(c2nl_scores / temperature, dim=-1)
        # align = (F.kl_div(p_c2nl.log(), p_c2c, reduction="batchmean") + F.kl_div(p_c2c.log(), p_c2nl, reduction="batchmean"))/2
        align = F.kl_div(p_c2nl.log(), p_c2c.detach(), reduction="batchmean")
        
        if return_sum_loss == False:
            return code2code_loss, code2nl_loss, align
        
        ids = torch.cat((source_ids,target_ids),-1)
        mask = self.bias[:,source_ids.size(-1):ids.size(-1),:ids.size(-1)].bool()
        mask = mask & ids[:,None,:].ne(1)
        out = self.decoder(target_ids,attention_mask=mask,past_key_values=encoder_output.past_key_values).last_hidden_state


        # copy retrieval target
        re_mask = retrieval_target_ids.ne(1)
        mem = self.encoder(retrieval_target_ids, attention_mask=re_mask)[0]
        
        attn_bias = torch.diag(c2nl_scores).unsqueeze(-1) * self.mem_bias_scale + self.mem_bias_base
        lm_prob = self.lm_head(out.transpose(0,1), 
                                mem.transpose(0, 1), 
                                re_mask.transpose(0, 1), 
                                attn_bias, 
                                retrieval_target_ids.transpose(0, 1)).transpose(0, 1)
        active_loss = target_ids[..., 1:].ne(1).view(-1)
        shift_logits = lm_prob[..., :-1, :].contiguous()
        shift_labels = target_ids[..., 1:].contiguous()
        
        loss_fct = LabelSmoothingNLLLoss(0.1)
        code_sum_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[active_loss],
                        shift_labels.view(-1)[active_loss])
        
        
        outputs = code_sum_loss,code_sum_loss*active_loss.sum(),active_loss.sum(), code2code_loss, code2nl_loss, align
        return outputs
    
    def generate(self, source_ids, retrieval_source_ids=None, retrieval_target_ids=None):
        mask = source_ids.ne(1)[:,None,:]*source_ids.ne(1)[:,:,None]
        encoder_output = self.encoder(source_ids,attention_mask=mask,use_cache=True)  
        re_out = self.encoder(retrieval_target_ids, attention_mask=retrieval_target_ids.ne(1))[0]  
        
        code_feature = self.sentence_emb(source_ids)
        nl_feature = self.sentence_emb(retrieval_target_ids)
        scores = torch.einsum("ab,cb->ac", code_feature, nl_feature)
        attn_bias = torch.diag(scores).unsqueeze(-1) * self.mem_bias_scale + self.mem_bias_base
 
        preds = []       
        zero = torch.LongTensor(1).fill_(0).to(source_ids.device)
        source_len = list(source_ids.ne(1).sum(-1).cpu().numpy())
        re_target_len = list(retrieval_target_ids.ne(1).sum(-1).cpu().numpy())
        for i in range(source_ids.shape[0]):
            context = [[x[i:i+1,:,:source_len[i]].repeat(self.beam_size,1,1,1) for x in y] 
                     for y in encoder_output.past_key_values]
            re_context = re_out[i:i+1, :re_target_len[i], :].repeat(self.beam_size, 1, 1)
            beam = Beam(self.beam_size,self.sos_id,self.eos_id)
            input_ids = beam.getCurrentState()
            context_ids = source_ids[i:i+1,:source_len[i]].repeat(self.beam_size,1)
            re_context_ids = retrieval_target_ids[i:i+1, :re_target_len[i]].repeat(self.beam_size, 1)
            bias = attn_bias[i:i+1].repeat(self.beam_size, 1)
            for _ in range(self.max_length): 
                if beam.done():
                    break
                
                ids = torch.cat((context_ids,input_ids),-1)
                mask = self.bias[:,context_ids.size(-1):ids.size(-1),:ids.size(-1)].bool()
                mask = mask & ids[:,None,:].ne(1)
                out = self.decoder(input_ids,attention_mask=mask,past_key_values=context).last_hidden_state
                lmprob = self.lm_head(out.transpose(0,1), 
                                        re_context.transpose(0, 1), 
                                        re_context_ids.ne(1).transpose(0, 1), 
                                        bias, 
                                        re_context_ids.transpose(0, 1)).transpose(0, 1)
                out = lmprob[:, -1, :]
                # hidden_states = out[:,-1,:]
                # out = self.lsm(self.lm_head(hidden_states)).data
                
                beam.advance(out)
                input_ids.data.copy_(input_ids.data.index_select(0, beam.getCurrentOrigin()))
                input_ids = torch.cat((input_ids,beam.getCurrentState()),-1)
            hyp = beam.getHyp(beam.getFinal())
            pred = beam.buildTargetTokens(hyp)[:self.beam_size]
            pred = [torch.cat([x.view(-1) for x in p]+[zero]*(self.max_length-len(p))).view(1,-1) for p in pred]
            preds.append(torch.cat(pred,0).unsqueeze(0))

        preds = torch.cat(preds,0)    

        return preds   

class FeedForwardLayer(nn.Module):

    def __init__(self, embed_dim, ff_embed_dim, dropout):
        super(FeedForwardLayer, self).__init__()
        self.fc1 = nn.Linear(embed_dim, ff_embed_dim)
        self.fc2 = nn.Linear(ff_embed_dim, embed_dim)
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.normal_(self.fc2.weight, std=0.02)
        nn.init.constant_(self.fc1.bias, 0.)
        nn.init.constant_(self.fc2.bias, 0.)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        return x


class CopyTokenDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, ff_embed_dim, dropout):
        super(CopyTokenDecoder, self).__init__()
        self.output_projection = nn.Linear(
                embed_dim,
                vocab_size,
                bias=False,
        )
        self.alignment_layer = MultiheadAttention(embed_dim, 1, dropout, weights_dropout=False)
        self.alignment_layer_norm = nn.LayerNorm(embed_dim)
        self.ff_layer = FeedForwardLayer(embed_dim, ff_embed_dim, dropout)
        self.ff_layer_norm = nn.LayerNorm(embed_dim)
        self.diverter = nn.Linear(2*embed_dim, 2)
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.diverter.weight, std=0.02)
        nn.init.constant_(self.diverter.bias, 0.)   

    def forward(self, outs, mem, mem_mask, mem_bias, copy_seq):
        #mem_bias = None
        attn, alignment_weight = self.alignment_layer(outs, mem, mem,
                                                    key_padding_mask=mem_mask,
                                                    need_weights='one',
                                                    attn_bias=mem_bias)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        attn_normalized = self.alignment_layer_norm(attn)

        gates = F.softmax(self.diverter(torch.cat([outs, attn_normalized], -1)), -1)
        gen_gate, copy_gate = gates.chunk(2, dim=-1)

        outs = self.alignment_layer_norm(outs + attn)
        outs = self.ff_layer(outs)
        outs = F.dropout(outs, p=self.dropout, training=self.training)
        outs = self.ff_layer_norm(outs)

        seq_len, bsz, _ = outs.size()
        probs = gen_gate * F.softmax(self.output_projection(outs), -1)

        #copy_seq: src_len x bsz
        #copy_gate: tgt_len x bsz 
        #alignment_weight: tgt_len x bsz x src_len
        #index: tgt_len x bsz
        index = copy_seq.transpose(0, 1).contiguous().view(1, bsz, -1).expand(seq_len, -1, -1)
        # -> tgt_len x bsz x src_len

        copy_probs = (copy_gate * alignment_weight).view(seq_len, bsz, -1)
        # -> tgt_len x bsz x src_len
        probs = probs.scatter_add_(-1, index, copy_probs)
        lprobs = torch.log(probs + 1e-12)
        return lprobs




class MultiheadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout=0., weights_dropout=True):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.weights_dropout = weights_dropout
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.in_proj_weight, std=0.02)
        nn.init.normal_(self.out_proj.weight, std=0.02)
        nn.init.constant_(self.in_proj_bias, 0.)
        nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None, need_weights=None, attn_bias=None):
        """ Input shape: Time x Batch x Channel
            key_padding_mask: Time x batch
            attn_mask:  tgt_len x src_len
            attn_bias:  bs x 1
        """
        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        tgt_len, bsz, embed_dim = query.size()
        assert key.size() == value.size()

        if qkv_same:
            # self-attention
            q, k, v = self.in_proj_qkv(query)
        elif kv_same:
            # encoder-decoder attention
            q = self.in_proj_q(query)
            k, v = self.in_proj_kv(key)
        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        q *= self.scaling


        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        src_len = k.size(1)

        attn_weights = torch.bmm(q, k.transpose(1, 2))

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]
        if attn_bias is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_bias = attn_bias.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights + attn_bias
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if attn_mask is not None:
            attn_weights.masked_fill_(
                ~attn_mask.unsqueeze(0),
                float('-inf')
            )


        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights.masked_fill_(
                ~key_padding_mask.transpose(0, 1).unsqueeze(1).unsqueeze(2),
                float('-inf')
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        attn_weights = F.softmax(attn_weights, dim=-1)
        if self.weights_dropout:
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn = torch.bmm(attn_weights, v)
        if not self.weights_dropout:
            attn = F.dropout(attn, p=self.dropout, training=self.training)

        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        if need_weights:
            # maximum attention weight over heads
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if need_weights == 'max':
                attn_weights, _ = attn_weights.max(dim=1)
            elif need_weights == "one":
                attn_weights = attn_weights[:,0,:,:]
            else:
                assert False, "need weights?"
            attn_weights = attn_weights.transpose(0, 1)
        else:
            attn_weights = None

        return attn, attn_weights

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)

    def in_proj_q(self, query):
        return self._in_proj(query, end=self.embed_dim)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim)

    def _in_proj(self, input, start=0, end=None):
        weight = self.in_proj_weight
        bias = self.in_proj_bias
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)










class Beam(object):
    def __init__(self, size,sos,eos):
        self.size = size
        self.tt = torch.cuda
        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        # The backpointers at each time-step.
        self.prevKs = []
        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size)
                       .fill_(0)]
        self.nextYs[0][0] = sos
        # Has EOS topped the beam yet.
        self._eos = eos
        self.eosTop = False
        # Time and k pair for finished.
        self.finished = []

    def getCurrentState(self):
        "Get the outputs for the current timestep."
        batch = self.tt.LongTensor(self.nextYs[-1]).view(-1, 1)
        return batch

    def getCurrentOrigin(self):
        "Get the backpointers for the current timestep."
        return self.prevKs[-1]

    def advance(self, wordLk):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attnOut`: Compute and update the beam search.

        Parameters:

        * `wordLk`- probs of advancing from the last step (K x words)
        * `attnOut`- attention at the last step

        Returns: True if beam search is complete.
        """
        numWords = wordLk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)

            # Don't let EOS have children.
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] == self._eos:
                    beamLk[i] = -1e20
        else:
            beamLk = wordLk[0]
        flatBeamLk = beamLk.view(-1)
        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)

        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prevK = bestScoresId // numWords
        self.prevKs.append(prevK)
        self.nextYs.append((bestScoresId - prevK * numWords))


        for i in range(self.nextYs[-1].size(0)):
            if self.nextYs[-1][i] == self._eos:
                s = self.scores[i]
                self.finished.append((s, len(self.nextYs) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.nextYs[-1][0] == self._eos:
            self.eosTop = True

    def done(self):
        return self.eosTop and len(self.finished) >=self.size

    def getFinal(self):
        if len(self.finished) == 0:
            self.finished.append((self.scores[0], len(self.nextYs) - 1, 0))
        self.finished.sort(key=lambda a: -a[0])
        if len(self.finished) != self.size:
            unfinished=[]
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] != self._eos:
                    s = self.scores[i]
                    unfinished.append((s, len(self.nextYs) - 1, i)) 
            unfinished.sort(key=lambda a: -a[0])
            self.finished+=unfinished[:self.size-len(self.finished)]
        return self.finished[:self.size]

    def getHyp(self, beam_res):
        """
        Walk back to construct the full hypothesis.
        """
        hyps=[]
        for _,timestep, k in beam_res:
            hyp = []
            for j in range(len(self.prevKs[:timestep]) - 1, -1, -1):
                hyp.append(self.nextYs[j+1][k])
                k = self.prevKs[j][k]
            hyps.append(hyp[::-1])
        return hyps
    
    def buildTargetTokens(self, preds):
        sentence=[]
        for pred in preds:
            tokens = []
            for tok in pred:
                if tok==self._eos:
                    break
                tokens.append(tok)
            sentence.append(tokens)
        return sentence
        

