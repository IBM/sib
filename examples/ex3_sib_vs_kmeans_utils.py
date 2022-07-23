import platform

import psutil


def populate_kwargs(base_params, params):
    param_sets = [base_params]
    for param, values in params.items():
        new_param_sets = []
        for param_set in param_sets:
            for value in values:
                new_param_set = param_set.copy()
                new_param_set[param] = value
                new_param_sets.append(new_param_set)
        param_sets = new_param_sets
    return param_sets


def str_kwargs(d):
    return '_'.join([(key + '-' + str(value)) for key, value in d.items()])


def get_system_desc():
    return {
        'machine': platform.machine(),
        'version': platform.version(),
        'platform': platform.platform(),
        'uname': platform.uname(),
        'system': platform.system(),
        'processor': platform.processor(),
        'ram': str(round(psutil.virtual_memory().total / (1024.0 ** 3)))+" GB",
        'cpus_physical': psutil.cpu_count(logical=False),
        'cpus_logical': psutil.cpu_count(logical=True)
    }
