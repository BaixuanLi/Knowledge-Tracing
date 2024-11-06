import torch
import tqdm
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse

from act_tracer import Tracer
from dataset import KnownsDataset
from utils import set_seed, get_top_k_dict, compute_effect, prepare_data


def intervened_generate(model, inputs, generation_args, hooks):
    '''
    hooks = [
        [module_name_0, hook_fn_0],
        [module_name_1, hook_fn_1],
        ...
    ]
    '''
    hook_handles = []
    for hook in hooks:
        for name, module in model.named_modules():
            if name == hook[0]:
                handle = module.register_forward_hook(hook[1])
                hook_handles.append(handle)

    max_generate_length = generation_args.pop('max_generate_length')

    # generate with hooks
    generation_args['max_length'] = inputs['input_ids'].shape[-1] + max_generate_length
    generation_args['pad_token_id'] = model.config.eos_token_id

    outputs = model.generate(**inputs, **generation_args)

    generation_args['max_generate_length'] = max_generate_length

    for handle in hook_handles:
        handle.remove()

    return {
        'tokens': outputs.sequences,
        'logits': outputs.scores,
    }


if __name__ == '__main__':
    set_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True)
    args = parser.parse_args()

    with open(args.config_path, "r") as file:
        config = yaml.safe_load(file)

    tokenizer = AutoTokenizer.from_pretrained(config['model']['pth'])
    model = AutoModelForCausalLM.from_pretrained(config['model']['pth'], torch_dtype=torch.bfloat16, device_map='auto')
    tracer = Tracer()
    dataset = KnownsDataset()

    noise_ratio = 0.1

    total_effect = []
    indirect_effect_wo_server = []
    indirect_effect_w_attn_server = []
    indirect_effect_w_mlp_server = []
    
    bar_format = "{l_bar}{bar} | {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    for i, data in tqdm.tqdm(enumerate(dataset), total=len(dataset), bar_format=bar_format):
        # prepare data, get subject position
        inputs, corrupt_range = prepare_data(tokenizer=tokenizer, data=data)
        inputs = {key: value.to(model.device) for key, value in inputs.items()}

        target_token_id = tokenizer.encode(' '+data['attribute'])[-1]

        tmp_ie_wo_server = torch.zeros(model.config.num_hidden_layers, inputs['input_ids'].shape[1])
        tmp_ie_w_attn_server = torch.zeros(model.config.num_hidden_layers-1, inputs['input_ids'].shape[1])
        tmp_ie_w_mlp_server = torch.zeros(model.config.num_hidden_layers-1, inputs['input_ids'].shape[1])


        # 1) clean run
        hooks = []
        for i in range(model.config.num_hidden_layers):
            for layer in config['hook_layer']['clean_save']:
                layer_name = layer.replace('[LAYER_NUM]', str(i))
                hook_fn = tracer.clean_act_save_hook(name=layer_name)
                hooks.append([layer_name, hook_fn])

        clean_outputs = intervened_generate(model=model, inputs=inputs, generation_args=config['generation_args'], hooks=hooks)
        

        # 2) corrupted run
        hooks = [[config['hook_layer']['corrupt'], tracer.corrupt_hook(corrupt_range=corrupt_range, noise_ratio=noise_ratio, device=model.device)]]

        for i in range(model.config.num_hidden_layers):
            for layer in config['hook_layer']['corrupted_save']:
                layer_name = layer.replace('[LAYER_NUM]', str(i))
                hook_fn = tracer.corrupted_act_save_hook(name=layer_name)
                hooks.append([layer_name, hook_fn])

        corrupted_outputs = intervened_generate(model=model, inputs=inputs, generation_args=config['generation_args'], hooks=hooks)

        tmp_te = compute_effect(logits_1=clean_outputs['logits'][0][0], logits_2=corrupted_outputs['logits'][0][0], target_token_id=target_token_id).detach().cpu()
        total_effect.append(tmp_te)

        # print(tokenizer.encode(' '+data['attribute'])[-1])
        # print(get_top_k_dict(logits=clean_outputs['logits'][0][0], tokenizer=tokenizer, k=5))
        # print(get_top_k_dict(logits=corrupted_outputs['logits'][0][0], tokenizer=tokenizer, k=5))


        # 3) corrupted run with clean hidden state intervened
        for i in range(model.config.num_hidden_layers):
            for pos in range(inputs['input_ids'].shape[1]):
                hooks = [[config['hook_layer']['corrupt'], tracer.corrupt_hook(corrupt_range=corrupt_range, noise_ratio=noise_ratio, device=model.device)]]

                layer = config['hook_layer']['clean_replace']
                layer_name = layer.replace('[LAYER_NUM]', str(i))
                hook_fn = tracer.clean_replace_hook(name=layer_name, token_pos=pos, device=model.device)
                hooks.append([layer_name, hook_fn])

                clean_intervened_outputs = intervened_generate(model=model, inputs=inputs, generation_args=config['generation_args'], hooks=hooks)

                tmp_ie_wo_server[i, pos] = compute_effect(logits_1=clean_intervened_outputs['logits'][0][0], logits_2=corrupted_outputs['logits'][0][0], target_token_id=target_token_id).detach().cpu()

        indirect_effect_wo_server.append(tmp_ie_wo_server)


        # 4) corrupted run with clean hidden state intervened and attn severed
        for i in range(model.config.num_hidden_layers-1):
            for pos in range(inputs['input_ids'].shape[1]):
                hooks = [[config['hook_layer']['corrupt'], tracer.corrupt_hook(corrupt_range=corrupt_range, noise_ratio=noise_ratio, device=model.device)]]
                
                layer = config['hook_layer']['clean_replace']
                layer_name = layer.replace('[LAYER_NUM]', str(i))
                hook_fn = tracer.clean_replace_hook(name=layer_name, token_pos=pos, device=model.device)
                hooks.append([layer_name, hook_fn])
                
                layer = config['hook_layer']['attn_sever']
                for j in range(i+1, model.config.num_hidden_layers):
                    layer_name = layer.replace('[LAYER_NUM]', str(j))
                    hook_fn = tracer.sever_hook(name=layer_name, token_pos=pos, device=model.device)
                    hooks.append([layer_name, hook_fn])

                clean_intervened_outputs_with_attn_severed = intervened_generate(model=model, inputs=inputs, generation_args=config['generation_args'], hooks=hooks)

                tmp_ie_w_attn_server[i, pos] = compute_effect(logits_1=clean_intervened_outputs_with_attn_severed['logits'][0][0], logits_2=corrupted_outputs['logits'][0][0], target_token_id=target_token_id).detach().cpu()
            
        indirect_effect_w_attn_server.append(tmp_ie_w_attn_server)


        # 5) corrupted run with clean hidden state intervened and mlp severed
        for i in range(model.config.num_hidden_layers-1):
            for pos in range(inputs['input_ids'].shape[1]):
                hooks = [[config['hook_layer']['corrupt'], tracer.corrupt_hook(corrupt_range=corrupt_range, noise_ratio=noise_ratio, device=model.device)]]

                layer = config['hook_layer']['clean_replace']
                layer_name = layer.replace('[LAYER_NUM]', str(i))
                hook_fn = tracer.clean_replace_hook(name=layer_name, token_pos=pos, device=model.device)
                hooks.append([layer_name, hook_fn])
                
                layer = config['hook_layer']['mlp_sever']
                for j in range(i+1, model.config.num_hidden_layers):
                    layer_name = layer.replace('[LAYER_NUM]', str(j))
                    hook_fn = tracer.sever_hook(name=layer_name, token_pos=pos, device=model.device)
                    hooks.append([layer_name, hook_fn])

                clean_intervened_outputs_with_mlp_severed = intervened_generate(model=model, inputs=inputs, generation_args=config['generation_args'], hooks=hooks)

                tmp_ie_w_mlp_server[i, pos] = compute_effect(logits_1=clean_intervened_outputs_with_mlp_severed['logits'][0][0], logits_2=corrupted_outputs['logits'][0][0], target_token_id=target_token_id).detach().cpu()

        indirect_effect_w_mlp_server.append(tmp_ie_w_mlp_server)


        tracer.clear_storage()


    torch.save(total_effect, config['result_pth']['base_pth']+config['model']['name']+'-te.pt')
    torch.save(indirect_effect_wo_server, config['result_pth']['base_pth']+config['model']['name']+'-ie_wo_sever.pt')
    torch.save(indirect_effect_w_attn_server, config['result_pth']['base_pth']+config['model']['name']+'-ie_w_attn_sever.pt')
    torch.save(indirect_effect_w_mlp_server, config['result_pth']['base_pth']+config['model']['name']+'-ie_w_mlp_sever.pt')