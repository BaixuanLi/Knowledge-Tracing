import random
import numpy as np
import torch
import string
import torch.nn.functional as F


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_top_k_dict(logits, tokenizer, k):
    probs = F.softmax(logits, dim=-1)
    top_k_probs, top_k_indices = torch.topk(probs, k)
    top_k_tokens = [tokenizer.decode([idx]) for idx in top_k_indices]
    top_k_dict = {token: prob.item() for token, prob in zip(top_k_tokens, top_k_probs)}

    print(top_k_indices[0], top_k_tokens[0])

    return top_k_dict


def compute_effect(logits_1, logits_2, target_token_id):
    assert len(logits_1.shape) == 1 and len(logits_2.shape) == 1

    probs_1 = F.softmax(logits_1, dim=-1)
    probs_2 = F.softmax(logits_2, dim=-1)

    effect = probs_1[target_token_id] - probs_2[target_token_id]

    return effect


def prepare_data(tokenizer, data):
    prompt_inputs = tokenizer(data['prompt'], return_tensors='pt')
    prompt_tokens = tokenizer.convert_ids_to_tokens(prompt_inputs['input_ids'][0], skip_special_tokens=False)

    corrupt_range = [0, 0]
    prompt_split_pos = [-1 for _ in range(len(prompt_tokens))]

    first_flag = False
    current_pos = 0
    for i, t in enumerate(prompt_tokens):
        if t == tokenizer.bos_token:
            continue
        elif not first_flag and 'Ġ' not in t:  # first token
            prompt_split_pos[i] = current_pos
            first_flag = True
        elif first_flag and 'Ġ' not in t:  # subsequent token
            prompt_split_pos[i] = current_pos
        elif 'Ġ' in t:  # not-first but initial
            current_pos += 1
            prompt_split_pos[i] = current_pos

    word_range = [0, 0]
    subject_words = data['subject'].split()
    punctuation = string.punctuation.replace('-', '')
    punctuation = punctuation.replace("'", '')

    for i, word in enumerate(data['prompt'].split()):
        word = word.translate(str.maketrans('', '', punctuation))  # remove punctuations
        if word == subject_words[0]:
            # print("###1", word, subject_words[0], i)
            word_range[0] = i
        if word == subject_words[-1]:
            # print("###2", word, subject_words[-1], i)
            word_range[1] = i
            break

    assert word_range[0] <= word_range[1]
    
    corrupt_range[0] = prompt_split_pos.index(word_range[0])
    corrupt_range[1] = len(prompt_split_pos) - 1 - prompt_split_pos[::-1].index(word_range[1])

    # print(prompt_split_pos)
    # print(prompt_tokens)
    # print(data['prompt'])
    # print(data['subject'])

    # print(corrupt_range)

    return prompt_inputs, corrupt_range