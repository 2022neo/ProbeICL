from transformers import AutoModelForCausalLM,AutoTokenizer,GPTNeoForCausalLM
import torch
from torch import nn
from torch import Tensor as T
from easydict import EasyDict as edict
from utils.metric import metric_dict
from pathlib import Path

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
            if force_pred:
                pred = self.get_completion_pred(input_ids)
            else:
                loss = self.completion_logits_loss(input_ids,None,answer_list)
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
        normed_losses = torch.nn.functional.normalize(options_losses, p=1, dim=-1)
        label_loss = normed_losses[test_label]
        pred = torch.argmin(normed_losses, dim=-1).item()
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
        # ans_ids,_ = self.text_to_tensor(answer_list[0],title=None)
        # option_ids = torch.cat(input_ids+[ans_ids],dim=-1)
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



    # def completion_losses(self,input_ids,input_atten_mask,labels,task):
    #     with torch.no_grad():
    #         answer_start = int(input_atten_mask.shape[-1]) 
    #         res = self.model.generate(input_ids=input_ids.squeeze(1), #remove the dim for option_num
    #                                     attention_mask=input_atten_mask.squeeze(1),
    #                                     eos_token_id=self.dataset_reader.tokenizer.encode("\n")[0],
    #                                     pad_token_id=self.dataset_reader.tokenizer.pad_token_id,
    #                                     max_length=min(self.max_length,answer_start+self.generate_max_len),
    #                                     do_sample=False)
                        
    #     pred_ids=res[:,answer_start:]
    #     preds=[]
    #     for i in range(len(pred_ids)):
    #         pred=tokenizer.decode(pred_ids[i],skip_special_tokens=True)
    #         # avoid empty prediction to avoid errors when calculating Rouge metric scores
    #         if '\n' not in pred: pred+='\n' 
    #         preds.append(pred)
    #     compute_metric=metric_dict[task.metric]
    #     scores=compute_metric(preds=preds, labels=labels, return_list=True)
    #     return  {
    #             "labels_losses": [1-score for score in scores],
    #             "accurate_list": scores,
    #             "preds": preds
    #             }