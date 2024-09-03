from dpr.models.hf_models import (
    HFBertEncoder,
    get_bert_tokenizer,
    BertTensorizer,
    get_optimizer, 
)
from dpr.utils.model_utils import (
    get_schedule_linear,
)
import torch
from torch import nn
from enum import Enum
from torch.cuda.amp import GradScaler
import re
import torch.optim as optim
import torch.nn.functional as F

class ENCODER_TYPE(Enum):
    QUERY_ENC = 0
    CONTEXT_ENC = 1


def init_retriever(cfg,inference_only=False):
    # init bert-based encoder with recommended max_len=512
    query_encoder = HFBertEncoder.init_encoder(
        cfg_name=cfg.pretrained_model_name,
        projection_dim=cfg.projection_dim,
        dropout=cfg.dropout,
        pretrained=True,
        cache_dir=cfg.cache_dir,
    )
    contx_encoder = HFBertEncoder.init_encoder(
        cfg_name=cfg.pretrained_model_name,
        projection_dim=cfg.projection_dim,
        dropout=cfg.dropout,
        pretrained=True,
        cache_dir=cfg.cache_dir,
    )
    retriever = Retriever(query_encoder,contx_encoder,cfg=cfg)

    # get tensorizer
    tokenizer = get_bert_tokenizer(cfg.pretrained_model_name, do_lower_case=True,cache_dir=cfg.cache_dir)
    tensorizer = BertTensorizer(tokenizer,cfg.sequence_length, pad_to_max=False)

    # get prompt_parser
    prompt_parser = PrompParser(cfg)
    if inference_only:
        return retriever,tensorizer,prompt_parser

    #init optimizer
    # optimizer = optim.Adam(retriever.parameters(), lr=cfg.learning_rate)
    optimizer= get_optimizer(
                retriever,
                learning_rate=cfg.learning_rate,
                adam_eps=cfg.adam_eps,
                weight_decay=cfg.weight_decay,
            )

    # init scheduler
    scheduler = get_schedule_linear(
        optimizer,
        cfg.warmup_steps,
        cfg.total_updates,
        steps_shift=0,
    )
    scaler = GradScaler()
    return retriever,tensorizer,optimizer, scheduler, scaler, prompt_parser


class PrompParser(object):
    def __init__(self,cfg) -> None:
        self.cfg=cfg
        self.int_to_letter = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',5:'G'}

    def remove_double_space(self, string):
        return re.sub("[ ]{2,}", " ", string)

    
    def process(self,entry,setup_type,task):
        if setup_type == "qa":
            sent = (
                task.get_question(entry)
                + task.get_answer(entry)
            )
        elif setup_type == "q":
            sent = task.get_question(entry)
        elif setup_type == "a":
            sent = task.get_answer(entry).strip()
        prompt = self.remove_double_space(sent)
        answer_list = [self.remove_double_space(a) for a in task.get_answers(entry)]
        label = task.get_label(entry)
        return prompt,answer_list,label




class Retriever(nn.Module):
    def __init__(self,query_encoder,contx_encoder, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.device = cfg.retriever_device
        self.query_encoder = query_encoder.to(self.device)
        self.contx_encoder = contx_encoder.to(self.device)
        self.hidden_dim = query_encoder.embeddings.word_embeddings.weight.shape[1]
        self.probe_heads = ProbeHeads(cfg.k_shot, self.hidden_dim).to(self.device)

    def query_to_probes(self,query_entry,tensorizer):
        question_ids = tensorizer.text_to_tensor(query_entry["demonstration"],title=query_entry["title"],add_special_tokens=True).to(self.device)
        question_ts = self.get_representation(
            ids_ts = question_ids,
            tensorizer = tensorizer,
            enctype = ENCODER_TYPE.QUERY_ENC
        )
        probes_ts = self.generate_probes(question_ts)
        probes_ts = probes_ts / probes_ts.norm(dim=1, keepdim=True) if self.cfg.norm_mask else probes_ts
        return probes_ts

    def encode_ctxs(self,ctx_entries,tensorizer):
        ctxs_ts = []
        for ctx_entry in ctx_entries:
            ctx_ids=tensorizer.text_to_tensor(ctx_entry["demonstration"],title=ctx_entry["title"],add_special_tokens=True).to(self.device)
            ctxs_ts.append(self.get_representation(
                ids_ts = ctx_ids,
                tensorizer = tensorizer,
                enctype = ENCODER_TYPE.CONTEXT_ENC
            ))
        ctxs_ts = torch.cat(ctxs_ts)
        ctxs_ts = ctxs_ts / ctxs_ts.norm(dim=1, keepdim=True) if self.cfg.norm_mask else ctxs_ts
        return ctxs_ts

    def calculate_similarity(self,query_entry,ctx_entries,tensorizer):
        probes_ts = self.query_to_probes(query_entry,tensorizer) # [k-shot, dim]
        ctxs_ts = self.encode_ctxs(ctx_entries,tensorizer) # [L, dim]
        similarity = probes_ts @ ctxs_ts.T # [k-shot, L]
        return similarity
    
    def calculate_ortho_loss(self,similarity):
        norm_sim = similarity.softmax(dim=-1)
        rsv_eye = 1 - torch.eye(self.cfg.k_shot).to(norm_sim.device)
        ortho_loss = (norm_sim @ norm_sim.T * rsv_eye).sum()/rsv_eye.sum()
        return ortho_loss
    
    def _parse_sim_matrix(self,sim_matrix):
        with torch.no_grad():
            _,candidate_ids = torch.topk(sim_matrix, self.cfg.k_shot,dim=-1)
            indices = []
            for inds in candidate_ids.tolist():
                for ind in inds:
                    if ind not in indices:
                        indices.append(ind)
                        break
            indices = torch.tensor(indices).unsqueeze(-1).to(sim_matrix.device)
        mask = torch.gather(sim_matrix,-1,indices)
        mask,indices = mask.reshape(-1), indices.reshape(-1)
        return mask,indices

    def get_training_mask(self,similarity,ctx_entries,epoch,mask_type):
        if mask_type==0:
            soft_mask,indices = torch.topk(similarity.softmax(dim=-1), 1,dim=-1)
            soft_mask,indices = soft_mask.reshape(-1), indices.reshape(-1)
            selected_ctxs = [ctx_entries[a] for a in indices.tolist()]
            if self.cfg.hard_mask:
                mask = torch.ones_like(soft_mask).to(soft_mask.device) - soft_mask.detach() + soft_mask 
            else:
                mask = soft_mask
        if mask_type==1:
            soft_mask,indices = self._parse_sim_matrix(
                similarity.softmax(dim=-1)
            )
            selected_ctxs = [ctx_entries[a] for a in indices.tolist()]
            if self.cfg.hard_mask:
                mask = torch.ones_like(soft_mask).to(soft_mask.device) - soft_mask.detach() + soft_mask 
            else:
                mask = soft_mask
        elif mask_type==2:
            mask, indices = self._parse_sim_matrix(
                F.gumbel_softmax(similarity, tau=self.cfg.temperature, hard=self.cfg.hard_mask, dim=-1)
            )
            selected_ctxs = [ctx_entries[a] for a in indices.tolist()]
        elif mask_type==3:
            mask, indices = self._parse_sim_matrix(
                F.gumbel_softmax(similarity, tau=self.cfg.temperature/epoch**2, hard=self.cfg.hard_mask, dim=-1)
            )
            selected_ctxs = [ctx_entries[a] for a in indices.tolist()]
        return mask, selected_ctxs
    
    def get_ctrs_loss(self,similarity,pos_idx_list_per_question):
        log_scores, _ = similarity.log_softmax(dim=-1).max(dim=0,keepdim=True) # [1, L]
        ctrs_loss = F.nll_loss(
            log_scores,
            torch.tensor(pos_idx_list_per_question).to(log_scores.device),
            reduction="mean",
        )
        return ctrs_loss

    def get_multi_ctrs_loss(self,similarity,pos_idx_list_per_question):
        log_scores, _ = similarity.log_softmax(dim=-1).max(dim=0) # [L]
        pos_idx_tensor = torch.tensor(pos_idx_list_per_question).to(log_scores.device)
        multi_ctrs_loss = -torch.gather(log_scores, -1, pos_idx_tensor).mean(dim=-1)
        return multi_ctrs_loss

    def get_representation(self,ids_ts,tensorizer, enctype):
        if len(ids_ts.shape)<2:
            ids_ts=ids_ts.unsqueeze(0)
        segments = torch.zeros_like(ids_ts)
        attn_mask = tensorizer.get_attn_mask(ids_ts)

        if enctype == ENCODER_TYPE.QUERY_ENC:
            sequence_output, pooled_output, hidden_states = self.query_encoder(
                ids_ts,
                segments,
                attn_mask,
                representation_token_pos=0,
            )
        else:
            assert enctype == ENCODER_TYPE.CONTEXT_ENC
            sequence_output, pooled_output, hidden_states = self.contx_encoder(
                ids_ts,
                segments,
                attn_mask,
                representation_token_pos=0,
            )
        return pooled_output
    
    def generate_probes(self,question_ts): # TODO(ambyear) # [k-shot, modeldim, modeldim] * [bsz, modeldim]T --> [k-shot, modeldim]
        probes = self.probe_heads(question_ts)
        return probes


class ProbeHeads(nn.Module):
    def __init__(self, k_shot, dim):
        super().__init__()

        self.probeheads = nn.ModuleList(
            [
                ProbeHeadLayer(dim) for _ in range(k_shot)
            ]
        )
    def forward(self, x):
        """x: [bsz, dim]"""
        y = []
        for phead in self.probeheads:
            _y = phead(x)
            y.append(_y.unsqueeze(0))

        y = torch.cat(y, dim=0) # --> [k, bsz, dim]
        y = y.squeeze(1) # TODO(ambyera): temp, just for bsz=1
        return y

class ProbeHeadLayer(nn.Module):
    def __init__(self, dim, actfun = nn.GELU() ):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(dim, 2*dim),
            actfun,
            nn.Linear(2*dim, dim),
        )
        # self.init_weight()
    
    def forward(self, x):
        return self.mlp(x)

    # def init_weight(self):
    #     nn.init.xavier_uniform_(self.weight)
