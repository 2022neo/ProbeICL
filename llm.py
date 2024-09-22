from transformers import AutoModelForCausalLM,AutoTokenizer,GPTNeoForCausalLM
import torch
from torch import nn
from torch import Tensor as T
from pathlib import Path
import torch.nn.functional as F
import random

def get_model_max_length(model_name):
    if 'llama-3' in model_name.lower():
        return 8000
    elif 'gpt-neo' in model_name.lower():
        return 2048
    raise NotImplementedError(model_name)

class CausalLM(nn.Module):
    def __init__(self,config) -> None:
        super().__init__()
        model_name=config.lm_name
        self.max_length = get_model_max_length(model_name)
        if Path(model_name).exists():
            self.model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_name,device_map='auto')
        else:
            self.model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_name, cache_dir=config.cache_dir,device_map='auto')
        self.model = self.model.eval()
        self.device = self.model.device
        self.override_attn()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=config.cache_dir, model_max_length=self.max_length
        )
        self.option_num = config.option_num # option_num == 1:  # text completion question
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.clear_ctx_mask()
        self.generate_max_len = config.generate_max_len
        self.cfg = config

    def set_ctx_mask(self,ctx_mask):
        self.__ctx_mask__ = ctx_mask
    def clear_ctx_mask(self):
        self.__ctx_mask__ = None

    def override_attn(self):
        def new_attn(self, query, key, value, attention_mask=None, head_mask=None):
            query = query.to(torch.float32)
            key = key.to(torch.float32)
            attn_weights = torch.matmul(query, key.transpose(-1, -2))
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length].to(torch.bool)
            mask_value = torch.finfo(attn_weights.dtype).min
            mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
            attn_weights = torch.where(causal_mask, attn_weights, mask_value)

            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask

            attn_weights = nn.functional.softmax(attn_weights, dim=-1)
            attn_weights = attn_weights.to(value.dtype)
            attn_weights = self.attn_dropout(attn_weights)
            if head_mask is not None:
                attn_weights = attn_weights * head_mask
            if self.parent.__ctx_mask__ is not None:
                attn_weights = attn_weights * self.parent.__ctx_mask__[:, None, None, :].to(attn_weights.device)
            attn_output = torch.matmul(attn_weights, value)
            return attn_output, attn_weights
        for name, sub_module in self.model.named_modules():
            if hasattr(sub_module, '_attn'):
                sub_module.parent=self
                sub_module._attn = new_attn.__get__(sub_module, sub_module.__class__)

    def text_to_tensor(self,text,title=None):
        if title:
            tokenized_text = self.tokenizer.encode_plus(
                title,
                text_pair=text,
                truncation=False,
                return_tensors="pt",
                add_special_tokens=False,
            )
        else:
            tokenized_text = self.tokenizer.encode_plus(
                text,
                truncation=False,
                return_tensors="pt",
                add_special_tokens=False,
            )
        ids, attention_mask = tokenized_text.input_ids.to(self.device),tokenized_text.attention_mask.float().to(self.device)
        return ids, attention_mask.requires_grad_(True)
    
    def inference(self,query_entry,ctx_entries,answer_list,test_label,force_pred=False):
        input_ids = []
        for id,ctx_entry in enumerate(ctx_entries):
            ctx_ids,_ = self.text_to_tensor(ctx_entry["demonstration"]+' \n ',title=ctx_entry['title'])
            input_ids+=[ctx_ids]
        q_ids,_ = self.text_to_tensor(query_entry["demonstration"],title=query_entry['title'])
        input_ids+=[q_ids]
        
        loss, pred = None, None
        if self.option_num>1:
            loss, pred =  self.choice_loss(input_ids,None,answer_list,test_label)
        elif self.option_num==1:
            loss = self.completion_logits_loss(input_ids,None,answer_list)
            if force_pred:
                pred = self.get_completion_pred(input_ids)
        else:
            raise NotImplementedError(f'loss for option_num={self.option_num} not implemented')
        return loss, pred

    def aug_inference(self,query_entry,ctx_entries,answer_list,test_label,force_pred=False):
        question = ''
        for ctx_entry in ctx_entries:
            question+=ctx_entry["demonstration"]+' \n '
        question+=query_entry["demonstration"]
        input_ids,_ = self.text_to_tensor(question)
        input_ids = [input_ids]

        loss, pred = None, None
        if self.option_num>1:
            loss, pred = self.choice_loss(input_ids,None,answer_list,test_label)
        elif self.option_num==1:
            loss = self.completion_logits_loss(input_ids,None,answer_list)
            if force_pred:
                pred = self.get_completion_pred(input_ids)
        else:
            raise NotImplementedError(f'loss for option_num={self.option_num} not implemented')
        return loss, pred
        
    def forward(self,query_entry,ctx_entries,answer_list,test_label,mask):
        input_ids = []
        input_ctx_mask = []
        for id,ctx_entry in enumerate(ctx_entries):
            ctx_ids,ctx_attn_mask = self.text_to_tensor(ctx_entry["demonstration"]+' \n ',title=ctx_entry["title"])
            input_ids+=[ctx_ids]
            input_ctx_mask.append(ctx_attn_mask * mask[id])

        q_ids,q_attn_mask = self.text_to_tensor(query_entry["demonstration"],title=query_entry["title"])
        input_ids+=[q_ids]
        input_ctx_mask+=[q_attn_mask] 
        
        loss, pred = None, None
        if self.option_num>1:
            loss, pred = self.choice_loss(input_ids,input_ctx_mask,answer_list,test_label)
        elif self.option_num==1:
            loss = self.completion_logits_loss(input_ids,input_ctx_mask,answer_list)
        else:
            raise NotImplementedError(f'loss for option_num={self.option_num} not implemented')
        return loss, pred
    
    def get_pad_id(self) -> int:
        return self.tokenizer.pad_token_id

    def get_attn_mask(self, tokens_tensor: T) -> T:
        return tokens_tensor != self.get_pad_id()

    def choice_loss(self,input_ids,input_ctx_mask,answer_list,test_label):
        options_losses = []
        for answer in answer_list:
            ans_ids,ans_attn_mask = self.text_to_tensor(answer,title=None)
            option_ids = torch.cat(input_ids+[ans_ids],dim=-1)
            option_ctx_mask = torch.cat(input_ctx_mask+[ans_attn_mask],dim=-1) if input_ctx_mask is not None else None
            self.set_ctx_mask(option_ctx_mask)
            output=self.model(option_ids)
            self.clear_ctx_mask()
            logits=output.logits[:,-ans_ids.shape[-1]-1:-1,:]
            logit_losses= -torch.nn.functional.log_softmax(logits.float(), dim=-1)
            loss= torch.gather(logit_losses, -1, ans_ids.unsqueeze(-1)).squeeze(-1).mean(dim=-1)
            options_losses.append(loss)
        options_losses = torch.cat(options_losses,dim=-1)
        if self.cfg.norm_option:
            options_losses = torch.nn.functional.normalize(options_losses, p=1, dim=-1)
        label_loss = options_losses[test_label]
        pred = torch.argmin(options_losses, dim=-1).item()
        return label_loss,pred


    def completion_logits_loss(self, input_ids,input_ctx_mask,answer_list):
        assert len(answer_list)==1
        ans_ids,ans_attn_mask = self.text_to_tensor(answer_list[0],title=None)
        option_ids = torch.cat(input_ids+[ans_ids],dim=-1)
        option_ctx_mask = torch.cat(input_ctx_mask+[ans_attn_mask],dim=-1) if input_ctx_mask is not None else None
        self.set_ctx_mask(option_ctx_mask)
        output=self.model(option_ids)
        self.clear_ctx_mask()
        logits=output.logits[:,-ans_ids.shape[-1]-1:-1,:]
        logit_losses= -torch.nn.functional.log_softmax(logits.float(), dim=-1)
        loss = torch.gather(logit_losses, -1, ans_ids.unsqueeze(-1)).mean()
        return loss

    def get_completion_pred(self, input_ids):
        input_ids = torch.cat(input_ids,dim=-1)
        answer_start = int(input_ids.shape[-1])
        res = self.model.generate(input_ids=input_ids, #remove the dim for option_num
                                    eos_token_id=self.tokenizer.encode("\n")[0],
                                    pad_token_id=self.tokenizer.pad_token_id,
                                    max_length=min(self.max_length,answer_start+self.generate_max_len),
                                    do_sample=False,
                                    )
        pred_ids=res[:,answer_start:].squeeze(0)
        pred=self.tokenizer.decode(pred_ids,skip_special_tokens=True)
        if '\n' not in pred: pred+='\n' 
        return pred

    def _random_sim2indices(self,size,exclude=[]):
        indices = [a for a in random.choices(list(range(size)),k=self.cfg.k_shot)]
        if set(indices)==set(exclude):
            return self._random_sim2indices(size,exclude)
        return indices

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
        return mask,indices.tolist()

    def _sim2indices(self,sim_matrix,skiplst=[]):
        with torch.no_grad():
            _,candidate_ids = torch.topk(sim_matrix, self.cfg.k_shot,dim=-1)
            indices = []
            for inds in candidate_ids.tolist():
                for ind in inds:
                    if ind not in indices and ind not in skiplst:
                        indices.append(ind)
                        break
        return indices

    def _indices2mask(self,sim_matrix,indices,hard_mask=False):
        indices_tensor = torch.tensor(indices).unsqueeze(-1).to(sim_matrix.device)
        soft_mask = torch.gather(sim_matrix,-1,indices_tensor).reshape(-1)
        if hard_mask:
            mask = torch.ones_like(soft_mask).to(soft_mask.device) - soft_mask.detach() + soft_mask 
        else:
            mask = soft_mask
        return mask

    def get_llm_loss(self,similarity,query_entry,ctx_entries,answer_list,label,epoch):
        soft_similarity = similarity.softmax(dim=-1)
        if self.cfg.mask_type==0:
            parsed_similarity = soft_similarity
        elif self.cfg.mask_type==1:
            parsed_similarity = F.gumbel_softmax(similarity, tau=self.cfg.temperature/(epoch**2), hard=False, dim=-1)
        return self.reward_loss(query_entry,ctx_entries,answer_list,label,parsed_similarity,soft_similarity)

    def _reward_coef(self,flag:bool):
        if flag:
            return 1
        else:
            return -1

    def reward_loss(self,query_entry,ctx_entries,answer_list,label,parsed_similarity,soft_similarity):
        size = soft_similarity.shape[-1]
        if self.cfg.reward_type==0:
            indices = self._sim2indices(parsed_similarity)
            mask = self._indices2mask(parsed_similarity,indices,self.cfg.hard_mask)
            selected_ctxs = [ctx_entries[a] for a in indices]
            label_loss,_ = self.forward(query_entry,selected_ctxs,answer_list,label,mask)
            return label_loss
        
        elif self.cfg.reward_type==1:
            indices1 = self._random_sim2indices(size)
            indices2 = self._random_sim2indices(size,exclude=indices1)
            mask1 = self._indices2mask(parsed_similarity,indices1,self.cfg.hard_mask)
            mask2 = self._indices2mask(parsed_similarity,indices2,self.cfg.hard_mask)
            selected_ctxs1 = [ctx_entries[a] for a in indices1]
            selected_ctxs2 = [ctx_entries[a] for a in indices2]
            label_loss1,pred1 = self.forward(query_entry,selected_ctxs1,answer_list,label,mask1)
            label_loss2,pred2 = self.forward(query_entry,selected_ctxs2,answer_list,label,mask2)
            reward1=-label_loss1
            reward2=-label_loss2
        
        elif self.cfg.reward_type==2:
            indices1 = self._random_sim2indices(size)
            indices2 = self._random_sim2indices(size,exclude=indices1)
            softmask1 = self._indices2mask(soft_similarity,indices1)
            softmask2 = self._indices2mask(soft_similarity,indices2)
            selected_ctxs1 = [ctx_entries[a] for a in indices1]
            selected_ctxs2 = [ctx_entries[a] for a in indices2]
            with torch.no_grad():
                label_loss1,pred1 = self.inference(query_entry,selected_ctxs1,answer_list,label)
                label_loss2,pred2 = self.inference(query_entry,selected_ctxs2,answer_list,label)
            reward1=softmask1.mean()
            reward2=softmask2.mean()
        
        elif self.cfg.reward_type==3:
            indices1 = self._sim2indices(parsed_similarity)
            indices2 = self._random_sim2indices(size,exclude=indices1)
            mask1 = self._indices2mask(parsed_similarity,indices1,self.cfg.hard_mask)
            mask2 = self._indices2mask(parsed_similarity,indices2,self.cfg.hard_mask)
            selected_ctxs1 = [ctx_entries[a] for a in indices1]
            selected_ctxs2 = [ctx_entries[a] for a in indices2]
            label_loss1,pred1 = self.forward(query_entry,selected_ctxs1,answer_list,label,mask1)
            label_loss2,pred2 = self.forward(query_entry,selected_ctxs2,answer_list,label,mask2)
            reward1=-label_loss1
            reward2=-label_loss2
        
        elif self.cfg.reward_type==4:
            indices1 = self._sim2indices(parsed_similarity)
            indices2 = self._random_sim2indices(size,exclude=indices1)
            softmask1 = self._indices2mask(soft_similarity,indices1)
            softmask2 = self._indices2mask(soft_similarity,indices2)
            selected_ctxs1 = [ctx_entries[a] for a in indices1]
            selected_ctxs2 = [ctx_entries[a] for a in indices2]
            with torch.no_grad():
                label_loss1,pred1 = self.inference(query_entry,selected_ctxs1,answer_list,label)
                label_loss2,pred2 = self.inference(query_entry,selected_ctxs2,answer_list,label)
            reward1=softmask1.mean()
            reward2=softmask2.mean()
        
        elif self.cfg.reward_type==5:
            indices1 = self._sim2indices(parsed_similarity)
            indices2 = self._sim2indices(parsed_similarity,skiplst=indices1)
            mask1 = self._indices2mask(parsed_similarity,indices1,self.cfg.hard_mask)
            mask2 = self._indices2mask(parsed_similarity,indices2,self.cfg.hard_mask)
            selected_ctxs1 = [ctx_entries[a] for a in indices1]
            selected_ctxs2 = [ctx_entries[a] for a in indices2]
            label_loss1,pred1 = self.forward(query_entry,selected_ctxs1,answer_list,label,mask1)
            label_loss2,pred2 = self.forward(query_entry,selected_ctxs2,answer_list,label,mask2)
            reward1=-label_loss1
            reward2=-label_loss2

        
        elif self.cfg.reward_type==6:
            indices1 = self._sim2indices(parsed_similarity)
            indices2 = self._sim2indices(parsed_similarity,skiplst=indices1)
            softmask1 = self._indices2mask(soft_similarity,indices1)
            softmask2 = self._indices2mask(soft_similarity,indices2)
            selected_ctxs1 = [ctx_entries[a] for a in indices1]
            selected_ctxs2 = [ctx_entries[a] for a in indices2]
            with torch.no_grad():
                label_loss1,pred1 = self.inference(query_entry,selected_ctxs1,answer_list,label)
                label_loss2,pred2 = self.inference(query_entry,selected_ctxs2,answer_list,label)
            reward1=softmask1.mean()
            reward2=softmask2.mean()

        # return preference loss
        if self.cfg.filter_positive and self.cfg.option_num>1 and (pred1 != label) and (pred2 != label):
            label_loss = -F.logsigmoid((-reward1-reward2)-self.cfg.gamma)
        else:
            label_loss = -F.logsigmoid((reward1-reward2)*self._reward_coef(label_loss1.item()<label_loss2.item())-self.cfg.gamma)
        return label_loss