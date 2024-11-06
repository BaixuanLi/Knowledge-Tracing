import torch
import yaml
import tqdm
from transformers import AutoTokenizer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

from utils import set_seed, prepare_data
from dataset import KnownsDataset


def load_causal_all(base_pth, model_name):
    valid_index = torch.load(base_pth+model_name+'-valid_index.pt', weights_only=False)

    total_effect = torch.load(base_pth+model_name+'-te.pt', weights_only=False)
    indirect_effect_wo_server = torch.load(base_pth+model_name+'-ie_wo_sever.pt', weights_only=False)
    indirect_effect_w_attn_server = torch.load(base_pth+model_name+'-ie_w_attn_sever.pt', weights_only=False)
    indirect_effect_w_mlp_server = torch.load(base_pth+model_name+'-ie_w_mlp_sever.pt', weights_only=False)

    total_effect = [total_effect[i] for i in valid_index]
    indirect_effect_wo_server = [indirect_effect_wo_server[i] for i in valid_index]
    indirect_effect_w_attn_server = [indirect_effect_w_attn_server[i] for i in valid_index]
    indirect_effect_w_mlp_server = [indirect_effect_w_mlp_server[i] for i in valid_index]

    return valid_index, total_effect, indirect_effect_wo_server, indirect_effect_w_attn_server, indirect_effect_w_mlp_server


def ie_tracing(
        subject_range, 
        indirect_effect_wo_server, 
        indirect_effect_w_attn_server, 
        indirect_effect_w_mlp_server
    ):
    first_subject_token_ie = {
        'wo_sever': [],  # [layer_num]
        'w_attn_sever': [],
        'w_mlp_sever': [],
    }
    last_subject_token_ie = {
        'wo_sever': [],
        'w_attn_sever': [],
        'w_mlp_sever': [],
    }
    further_tokens_ie = {
        'wo_sever': [],
        'w_attn_sever': [],
        'w_mlp_sever': [],
    }
    last_token_ie = {
        'wo_sever': [],
        'w_attn_sever': [],
        'w_mlp_sever': [],
    }

    for i, r in enumerate(subject_range):
        if r[0] != r[1]:
            first_subject_token_ie['wo_sever'].append(indirect_effect_wo_server[i][:, r[0]])
            first_subject_token_ie['w_attn_sever'].append(indirect_effect_w_attn_server[i][:, r[0]])
            first_subject_token_ie['w_mlp_sever'].append(indirect_effect_w_mlp_server[i][:, r[0]])

        last_subject_token_ie['wo_sever'].append(indirect_effect_wo_server[i][:, r[1]])
        last_subject_token_ie['w_attn_sever'].append(indirect_effect_w_attn_server[i][:, r[1]])
        last_subject_token_ie['w_mlp_sever'].append(indirect_effect_w_mlp_server[i][:, r[1]])

        for j in range(r[1]+1, indirect_effect_wo_server[i].shape[1]-1):
            further_tokens_ie['wo_sever'].append(indirect_effect_wo_server[i][:, j])        
            further_tokens_ie['w_attn_sever'].append(indirect_effect_w_attn_server[i][:, j])
            further_tokens_ie['w_mlp_sever'].append(indirect_effect_w_mlp_server[i][:, j])
            
        last_token_ie['wo_sever'].append(indirect_effect_wo_server[i][:, -1])
        last_token_ie['w_attn_sever'].append(indirect_effect_w_attn_server[i][:, -1])
        last_token_ie['w_mlp_sever'].append(indirect_effect_w_mlp_server[i][:, -1])

    for key in ['wo_sever', 'w_attn_sever', 'w_mlp_sever']:
        first_subject_token_ie[key] = torch.mean(torch.stack(first_subject_token_ie[key]), dim=0)
        last_subject_token_ie[key] = torch.mean(torch.stack(last_subject_token_ie[key]), dim=0)
        further_tokens_ie[key] = torch.mean(torch.stack(further_tokens_ie[key]), dim=0)
        last_token_ie[key] = torch.mean(torch.stack(last_token_ie[key]), dim=0)

    return first_subject_token_ie, last_subject_token_ie, further_tokens_ie, last_token_ie


def ie_heat_map(base_pth, model_name, first_subject_token_ie, last_subject_token_ie, further_tokens_ie, last_token_ie):
    y_labels = ['First subject token', 'Last subject token', 'Further tokens', 'Last token']
    fig, axes = plt.subplots(1, 3, figsize=(40, 6))
    vmin, vmax = 0, 0.6

    ie_wo_sever=[first_subject_token_ie['wo_sever'], last_subject_token_ie['wo_sever'], further_tokens_ie['wo_sever'], last_token_ie['wo_sever']]
    ie_w_attn_sever = [first_subject_token_ie['w_attn_sever'], last_subject_token_ie['w_attn_sever'], further_tokens_ie['w_attn_sever'], last_token_ie['w_attn_sever']]
    ie_w_mlp_sever = [first_subject_token_ie['w_mlp_sever'], last_subject_token_ie['w_mlp_sever'], further_tokens_ie['w_mlp_sever'], last_token_ie['w_mlp_sever']]

    sns.heatmap(ie_wo_sever, cmap='Purples', cbar_kws={'label': 'AIE'}, vmin=vmin, vmax=vmax, ax=axes[0])
    axes[0].set_yticklabels(y_labels, rotation=0)
    axes[0].set_title('AIE of h')
    axes[0].set_xlabel('Layer')

    sns.heatmap(ie_w_attn_sever, cmap='Greens', cbar_kws={'label': 'AIE'}, vmin=vmin, vmax=vmax, ax=axes[1])
    axes[1].set_yticklabels(y_labels, rotation=0)
    axes[1].set_title('AIE of h w/ attn severed')
    axes[1].set_xlabel('Layer')

    sns.heatmap(ie_w_mlp_sever, cmap='Reds', cbar_kws={'label': 'AIE'}, vmin=vmin, vmax=vmax, ax=axes[2])
    axes[2].set_yticklabels(y_labels, rotation=0)
    axes[2].set_title('AIE of h w/ MLP severed')
    axes[2].set_xlabel('Layer')

    plt.tight_layout()
    plt.savefig(base_pth+model_name+'-AIE.pdf')


if __name__ == '__main__':
    set_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True)
    args = parser.parse_args()

    with open(args.config_path, "r") as file:
        config = yaml.safe_load(file)

    tokenizer = AutoTokenizer.from_pretrained(config['model']['pth'])
    model_name = config['model']['name']
    base_pth = config['result_pth']['base_pth']
    dataset = KnownsDataset()

    valid_index, total_effect, indirect_effect_wo_server, indirect_effect_w_attn_server, indirect_effect_w_mlp_server = load_causal_all(base_pth=base_pth, model_name=model_name)

    print('valid_index_num:', len(valid_index))
    print('total_effect:', torch.mean(torch.stack(total_effect)))

    subject_range = []
    bar_format = "{l_bar}{bar} | {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    for i, data in tqdm.tqdm(enumerate(dataset), total=len(dataset), bar_format=bar_format):
        if i in valid_index:
            _, tmp_range = prepare_data(tokenizer=tokenizer, data=data)
            subject_range.append(tmp_range)

    first_subject_token_ie, last_subject_token_ie, further_tokens_ie, last_token_ie = ie_tracing(
        subject_range=subject_range, 
        indirect_effect_wo_server=indirect_effect_wo_server, 
        indirect_effect_w_attn_server=indirect_effect_w_attn_server, 
        indirect_effect_w_mlp_server=indirect_effect_w_mlp_server
    )

    ie_heat_map(base_pth=base_pth, model_name=model_name, first_subject_token_ie=first_subject_token_ie, last_subject_token_ie=last_subject_token_ie, further_tokens_ie=further_tokens_ie, last_token_ie=last_token_ie)