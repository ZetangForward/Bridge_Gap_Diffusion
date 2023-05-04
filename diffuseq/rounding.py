import torch
# bert results
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer, default_data_collator, GPT2TokenizerFast
import sys, yaml, os
import json

import numpy as np
import torch.nn.functional as F

# def get_knn(model_emb, text_emb, dist='cos'):
#     if dist == 'cos':
#         adjacency = model_emb @ text_emb.transpose(1, 0).to(model_emb.device)
#     elif dist == 'l2':
#         adjacency = model_emb.unsqueeze(1).expand(-1, text_emb.size(0), -1) - text_emb.unsqueeze(0).expand(
#             model_emb.size(0), -1, -1)
#         adjacency = -torch.norm(adjacency, dim=-1)
#     topk_out = torch.topk(adjacency, k=6, dim=0)
#     return topk_out.values, topk_out.indices

# def enforce_repetition_penalty_(self, lprobs, batch_size, prev_output_tokens, repetition_penalty):
#     """repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858). """
#     for i in range(batch_size):
#         for previous_token in set(prev_output_tokens[i].tolist()):
#             # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
#             if lprobs[i, previous_token] < 0:
#                 lprobs[i, previous_token] *= repetition_penalty
#             else:
#                 lprobs[i, previous_token] /= repetition_penalty

def get_efficient_knn(model_emb, text_emb):
    emb_norm = (model_emb**2).sum(-1).view(-1, 1) # vocab
    text_emb_t = torch.transpose(text_emb.view(-1, text_emb.size(-1)), 0, 1) # d, bsz*seqlen
    arr_norm = (text_emb ** 2).sum(-1).view(-1, 1) # bsz*seqlen, 1
    # print(emb_norm.shape, arr_norm.shape)
    dist = emb_norm + arr_norm.transpose(0, 1) - 2.0 * torch.mm(model_emb, text_emb_t) # (vocab, d) x (d, bsz*seqlen)
    dist = torch.clamp(dist, 0.0, np.inf)
    # print(dist.shape)
    topk_out = torch.topk(-dist, k=1, dim=0)
    return topk_out.values, topk_out.indices

def get_efficient_knn_top_k(model_emb, text_emb, top_k, tau = 1.0):
    emb_norm = (model_emb**2).sum(-1).view(-1, 1) # vocab
    text_emb_t = torch.transpose(text_emb.view(-1, text_emb.size(-1)), 0, 1) # d, bsz*seqlen
    arr_norm = (text_emb ** 2).sum(-1).view(-1, 1) # bsz*seqlen, 1
    # print(emb_norm.shape, arr_norm.shape)
    dist = emb_norm + arr_norm.transpose(0, 1) - 2.0 * torch.mm(model_emb, text_emb_t) # (vocab, d) x (d, bsz*seqlen)
    dist = torch.clamp(dist, 0.0, np.inf)
    # print(dist.shape)
    topk_out = torch.topk(-dist, k=top_k, dim=0)

    sftmx = torch.nn.Softmax(dim=-1)
    indices = topk_out.indices.transpose(0,1)
    values = sftmx(topk_out.values.transpose(0,1)/ tau)
    idx = torch.multinomial(values, 1)
    indices = torch.gather(indices, -1, idx).transpose(0,1)
    values = torch.gather(values, -1, idx).transpose(0,1)

    return values, indices

def get_efficient_knn_top_p(model_emb, text_emb, top_p, tau = 1.0, scale = 1.0):
    emb_norm = (model_emb**2).sum(-1).view(-1, 1) # vocab
    text_emb_t = torch.transpose(text_emb.view(-1, text_emb.size(-1)), 0, 1) # d, bsz*seqlen
    arr_norm = (text_emb ** 2).sum(-1).view(-1, 1) # bsz*seqlen, 1
    # print(emb_norm.shape, arr_norm.shape)
    dist = emb_norm + arr_norm.transpose(0, 1) - 2.0 * torch.mm(model_emb, text_emb_t) # (vocab, d) x (d, bsz*seqlen)
    dist = torch.clamp(dist, 0.0, np.inf)
    # print(dist.shape)
    # topk_out = torch.topk(-dist, k=top_k, dim=0)

    # dist = 
    dist = -dist.transpose(0,1)
    sorted_logits, sorted_indices = torch.sort(dist, descending=True)
    if scale == "last":
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits / tau, dim=-1), dim=-1)
    else:
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    dist[indices_to_remove] = -float("Inf")

    # tau *= scale
    probs = F.softmax(dist / tau, dim=-1)

    idx = torch.multinomial(probs, 1)

    return -dist.transpose(0,1), idx.transpose(0,1)

def get_efficient_cos_top_p(model_emb, text_emb, top_p, tau = 1.0, scale = 1.0):
    dist = model_emb @ text_emb.transpose(1, 0).to(model_emb.device)

    # dist = 
    dist = dist.transpose(0,1)
    sorted_logits, sorted_indices = torch.sort(dist, descending=True)
    if scale == "last":
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits / tau, dim=-1), dim=-1)
    else:
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    dist[indices_to_remove] = -float("Inf")

    # tau *= scale
    probs = F.softmax(dist / tau, dim=-1)

    idx = torch.multinomial(probs, 1)

    return -dist.transpose(0,1), idx.transpose(0,1)


def get_efficient_knn_top_l(model_emb, text_emb, top_p):
    emb_norm = (model_emb**2).sum(-1).view(-1, 1) # vocab
    text_emb_t = torch.transpose(text_emb.view(-1, text_emb.size(-1)), 0, 1) # d, bsz*seqlen
    arr_norm = (text_emb ** 2).sum(-1).view(-1, 1) # bsz*seqlen, 1
    # print(emb_norm.shape, arr_norm.shape)
    dist = emb_norm + arr_norm.transpose(0, 1) - 2.0 * torch.mm(model_emb, text_emb_t) # (vocab, d) x (d, bsz*seqlen)
    dist = torch.clamp(dist, 0.0, np.inf)
    # print(dist.shape)
    # topk_out = torch.topk(-dist, k=top_k, dim=0)

    # dist = 
    sftmx = torch.nn.Softmax(dim=-1)
    indices = topk_out.indices.transpose(0,1)
    values = sftmx(topk_out.values.transpose(0,1))
    idx = torch.multinomial(values, 1)
    indices = torch.gather(indices, -1, idx).transpose(0,1)
    values = torch.gather(values, -1, idx).transpose(0,1)

    return values, indices

def rounding_func(text_emb_lst, model, tokenizer, emb_scale_factor=1.0, scale=1.0):
    decoded_out_lst = []
    
    model_emb = model.weight  # input_embs
    down_proj_emb2 = None

    dist = 'l2'
    
    for text_emb in text_emb_lst:
        import torch
        text_emb = torch.tensor(text_emb)
        # print(text_emb.shape)
        if len(text_emb.shape) > 2:
            text_emb = text_emb.view(-1, text_emb.size(-1))
        else:
            text_emb = text_emb
        val, indices = get_knn((down_proj_emb2 if dist == 'cos' else model_emb),
                                text_emb.to(model_emb.device), dist=dist)
    
        decoded_out_lst.append(tokenizer.decode_token(indices[0]))

    return decoded_out_lst

def compute_logp(args, model, x, input_ids):
    word_emb = model.weight
    sigma = 0.1
    if args.model_arch == '1d-unet':
        x = x.permute(0, 2, 1)

    bsz, seqlen, dim = x.shape

    x_flat = x.reshape(-1, x.size(-1)).unsqueeze(0)  # 1, bsz*sample*seqlen, dim
    word_emb_flat = word_emb.unsqueeze(1)  # vocab, 1,  dim
    diff = (x_flat - word_emb_flat) ** 2  # vocab, seqlen, dim

    logp_expanded = -diff.sum(dim=-1) / (2 * sigma ** 2)  # vocab, seqlen
    logp_expanded = logp_expanded.permute((1, 0))

    ce = torch.nn.CrossEntropyLoss(reduction='none')
    loss = ce(logp_expanded, input_ids.view(-1)).view(bsz, seqlen)

    return loss

def get_weights(model, args):
    if hasattr(model, 'transformer'):
        input_embs = model.transformer.wte  # input_embs
        down_proj = model.down_proj
        model_emb = down_proj(input_embs.weight)
        print(model_emb.shape)
        model = torch.nn.Embedding(model_emb.size(0), model_emb.size(1))
        print(args.emb_scale_factor)
        model.weight.data = model_emb * args.emb_scale_factor

    elif hasattr(model, 'weight'):
        pass
    else:
        assert NotImplementedError
        
    model.weight.requires_grad = False
    return model

def denoised_fn_round(args, model, text_emb, t):
    # print(text_emb.shape) # bsz, seqlen, dim
    model_emb = model.weight  # input_embs
    # print(t)
    if args.clamp_skip == 1:
        if t[0].item() % 2 == 0:
            print(t[0].item(), ": clamp skip")
            return text_emb

    elif args.clamp_skip == 0:
        if t[0].item() % 2 == 1:
            # print(t[0].item(), ": clamp skip")
            return text_emb

    # print(t[0].item(), ": clamp do")

    old_shape = text_emb.shape
    old_device = text_emb.device

    if len(text_emb.shape) > 2:
        text_emb = text_emb.reshape(-1, text_emb.size(-1))
    else:
        text_emb = text_emb
    # val, indices = get_knn(model_emb, text_emb.to(model_emb.device), dist=dist)
    
    if args.top_k != 0:
        # try:
        #     T = (sum(json.loads(args.timestep_respacing)) if args.timestep_respacing else args.step) - 1
        # except:
        #     T = int(args.timestep_respacing[4:])
        # scale = (t[0].item() / T + args.scale_end * (1 - t[0].item() / T))
        scale = 1
        top_k = max(int(args.top_k * scale), 1) 
        val, indices = get_efficient_knn_top_k(model_emb, text_emb.to(model_emb.device), top_k, args.tau)
    elif args.top_p != 0:
        if args.scale_end == "last":
            if t[0].item() == 0:
                print(t[0].item(), ":do topp")
                val, indices = get_efficient_knn_top_p(model_emb, text_emb.to(model_emb.device), args.top_p, args.tau, args.scale_end)
            else:
                print(t[0].item(), ":do greedy")
                val, indices = get_efficient_knn(model_emb, text_emb.to(model_emb.device))
        elif args.scale_end == "odd":
            if t[0].item() % 2 == 1:
                print(t[0].item(), ":do topp")
                val, indices = get_efficient_knn_top_p(model_emb, text_emb.to(model_emb.device), args.top_p, args.tau, args.scale_end)
            else:
                print(t[0].item(), ":do greedy")
                val, indices = get_efficient_knn(model_emb, text_emb.to(model_emb.device))
        elif args.scale_end == "":
            print(t[0].item(), ":do topp")
            val, indices = get_efficient_knn_top_p(model_emb, text_emb.to(model_emb.device), args.top_p, args.tau, args.scale_end)
        elif args.scale_end.startswith('last_'):
            t_topp = int(args.scale_end.split('_')[-1])
            if t[0].item() < t_topp:
                print(t[0].item(), ":do topp")
                val, indices = get_efficient_knn_top_p(model_emb, text_emb.to(model_emb.device), args.top_p, args.tau, args.scale_end)
            else:
                print(t[0].item(), ":do greedy")
                val, indices = get_efficient_knn(model_emb, text_emb.to(model_emb.device))
        elif args.scale_end.startswith('first_'):
            t_topp = int(args.scale_end.split('_')[-1])
            if t[0].item() >= t_topp:
                print(t[0].item(), ":do topp")
                val, indices = get_efficient_knn_top_p(model_emb, text_emb.to(model_emb.device), args.top_p, args.tau, args.scale_end)
            else:
                print(t[0].item(), ":do greedy")
                val, indices = get_efficient_knn(model_emb, text_emb.to(model_emb.device))
        elif args.scale_end == 'cos':
            print(t[0].item(), ":do cos topp")
            val, indices = get_efficient_cos_top_p(model_emb, text_emb.to(model_emb.device), args.top_p, args.tau, args.scale_end)
        else:
            raise NotImplementedError("Unkown args.scale_end:", args.scale_end)


    else:
        val, indices = get_efficient_knn(model_emb, text_emb.to(model_emb.device))
    rounded_tokens = indices[0]
    # print(rounded_tokens.shape)
    new_embeds = model(rounded_tokens).view(old_shape).to(old_device)

    return new_embeds