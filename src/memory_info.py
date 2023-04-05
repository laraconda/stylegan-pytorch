"""

def meminfo():
    meminfo = !free -h
    meminfo = meminfo[:2]
    meminfo = '\n'.join(meminfo)
    return meminfo


def gpu_meminfo():
    gpu_info_list = !nvidia-smi --query-gpu=memory.total,memory.free,memory.used --format=csv
    gpu_info = '\n'.join(gpu_info_list)
    if gpu_info.find('failed') >= 0:
        return 'Not connected to a GPU'
    else:
        indices = [i for i, val in enumerate(gpu_info_list[0]) if val == ']']
        vals = gpu_info_list[1].split(',')
        gpu_info_vals = ''
        for i, val in enumerate(vals):
            val.strip()
            if i > 0:
                prev_index = indices[i-1]
            else:
                prev_index = -1
            spaces = max(indices[i] - len(val) - prev_index, 0)
            gpu_info_vals += ' ' * spaces + val
        gpu_info = '\n'.join([gpu_info_list[0], gpu_info_vals])
        return gpu_info


def print_gpu_info():
    gpu_info = !nvidia-smi
    gpu_info = '\n'.join(gpu_info)
    if gpu_info.find('failed') >= 0:
        print('Not connected to a GPU')
    else:
        print(gpu_info)

"""
