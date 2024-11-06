import torch
import tqdm
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse

from dataset import KnownsDataset
from utils import set_seed


if __name__ == '__main__':
    set_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True)
    args = parser.parse_args()

    with open(args.config_path, "r") as file:
        config = yaml.safe_load(file)

    tokenizer = AutoTokenizer.from_pretrained(config['model']['pth'])
    model = AutoModelForCausalLM.from_pretrained(config['model']['pth'], torch_dtype=torch.bfloat16, device_map='auto')
    dataset = KnownsDataset()

    valid_index = []
    bar_format = "{l_bar}{bar} | {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    for i, data in tqdm.tqdm(enumerate(dataset), total=len(dataset), bar_format=bar_format):
        inputs = tokenizer(data['prompt'], return_tensors='pt')
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        
        generation_args = config['generation_args']
        max_generate_length = generation_args.pop('max_generate_length')
        generation_args['max_length'] = inputs['input_ids'].shape[-1] + max_generate_length
        generation_args['pad_token_id'] = model.config.eos_token_id

        outputs = model.generate(**inputs, **generation_args)

        generation_args['max_generate_length'] = max_generate_length

        generation = ' '.join(tokenizer.decode(outputs.sequences[0, inputs['input_ids'].shape[1]:]).split())
        
        if generation == data['attribute']:
            valid_index.append(i)

    torch.save(valid_index, config['result_pth']['base_pth']+config['model']['name']+'-valid_index.pt')