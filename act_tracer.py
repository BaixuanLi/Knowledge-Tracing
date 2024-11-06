import torch
import numpy as np


class Tracer:
    def __init__(self):
        self.clean_acts = {}
        self.corrupted_acts = {}

    def clean_replace_hook(self, name, token_pos, device):
        def hook(model, inputs, outputs):
            try:
                if isinstance(outputs, tuple):
                    outputs[0][:, token_pos, :] = self.clean_acts[name][:, token_pos, :].to(device)
                else:
                    outputs[:, token_pos, :] = self.clean_acts[name][:, token_pos, :].to(device)
            except:
                raise ValueError('Acts not exist')

            return outputs
        
        return hook
    
    def sever_hook(self, name, token_pos, device):
        def hook(model, inputs, outputs):
            try:
                if isinstance(outputs, tuple):
                    outputs[0][:, token_pos, :] = self.corrupted_acts[name][:, token_pos, :].to(device)
                else:
                    outputs[:, token_pos, :] = self.corrupted_acts[name][:, token_pos, :].to(device)
            except:
                raise ValueError('Acts not exist')

            return outputs
        
        return hook
    
    def corrupt_hook(self, corrupt_range, noise_ratio, device):
        def hook(model, inputs, outputs):
            start, end = corrupt_range[0], corrupt_range[1]+1

            prng = lambda *shape: np.random.randn(*shape)
            noise_fn = lambda x: noise_ratio * x

            if isinstance(outputs, tuple):
                noise_data = noise_fn(torch.from_numpy(prng(outputs[0].shape[0], end-start, outputs[0].shape[2]))).to(device)
                outputs[0][:, start: end] += noise_data
            else:
                noise_data = noise_fn(torch.from_numpy(prng(outputs.shape[0], end-start, outputs.shape[2]))).to(device)
                outputs[:, start: end] += noise_data

            return outputs
        
        return hook
    
    def clean_act_save_hook(self, name):
        def hook(model, inputs, outputs):
            if isinstance(outputs, tuple):
                self.clean_acts[name] = outputs[0].detach().cpu()
            else:
                self.clean_acts[name] = outputs.detach().cpu()

        return hook
    
    def corrupted_act_save_hook(self, name):
        def hook(model, inputs, outputs):
            if isinstance(outputs, tuple):
                self.corrupted_acts[name] = outputs[0].detach().cpu()
            else:
                self.corrupted_acts[name] = outputs.detach().cpu()

        return hook
    
    def clear_storage(self):
        self.clean_acts = {}
        self.corrupted_acts = {}