import torch
import argparse

def strip_prefix(weights, output_path, prefix):
    if len(prefix) == 0:
        return
    state_dict = torch.load(weights)

    new_state_dict = {}

    m_state_dict = state_dict
    if "model" in state_dict:
        m_state_dict = state_dict["model"]

    for k, v in m_state_dict.items():
        if k.startswith(prefix):
            new_state_dict[k[len(prefix):]] = v
        else:
            new_state_dict[k] = v
    
    torch.save({"model": new_state_dict}, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, help='the weights file path')
    parser.add_argument('--output', type=str, help='the output file path for processed weights file')
    parser.add_argument('--prefix', type=str, help='the prefix to strip')

    opt = parser.parse_args()

    strip_prefix(opt.weights, opt.output, opt.prefix)
