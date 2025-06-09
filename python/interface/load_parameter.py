import json
import yaml
import os


# load the configuration from a YAML or JSON file
def load_config(filename):
    if filename.endswith('.yaml') or filename.endswith('.yml'):
        with open(filename, 'r') as f:
            config = yaml.safe_load(f)
    elif filename.endswith('.json'):
        with open(filename, 'r') as f:
            config = json.load(f)
    else:
        raise ValueError(f"Unsupported file format: {filename}")
    return config

def validate_config(config):
    # define the required fields and their types
    required_fields = {
        'mode': str,
        'technology': str,
        'device_type': str,
        'frequency': (float, int),
        'precision_mu': (float, int),
        'precision_sigma': (float, int),
        'dataset': str,
        'model': str,
        'distribution_file': str,
    }

    supported_modes = ['memory', 'cim']
    supported_technologies = ['22', '45', '65']
    supported_device_types = ['SRAM', 'RRAM', 'PCM']
    supported_datasets = ['MNIST', 'CIFAR-10', 'ImageNet']
    supported_models = ['ResNet18', 'ResNet50', 'MLP', 'VGG16']

    print("\n===== Loaded Configuration Parameters =====")
    print(json.dumps(config, indent=4, sort_keys=False))
    print("============================================\n")

    # check if all required fields are present and of the correct type
    for field, field_type in required_fields.items():
        if field not in config:
            raise ValueError(f"Missing required field: {field}")

        if not isinstance(config[field], field_type):
            raise TypeError(f"Field '{field}' should be of type {field_type}, but got {type(config[field])}")

    # Specific value checks
    if config['mode'] not in supported_modes:
        raise ValueError(f"Unsupported mode: {config['mode']}. Supported modes: {supported_modes}")

    if config['technology'] not in supported_technologies:
        raise ValueError(f"Unsupported technology: {config['technology']}. Supported technologies: {supported_technologies}")

    if config['device_type'] not in supported_device_types:
        raise ValueError(f"Unsupported device_type: {config['device_type']}. Supported device_types: {supported_device_types}")

    if config['frequency'] <= 0:
        raise ValueError("frequency must be positive.")

    if config['precision_mu'] < 0:
        raise ValueError("precision_mu must be non-negative.")

    if config['precision_sigma'] < 0:
        raise ValueError("precision_sigma must be non-negative.")

    if config['dataset'] not in supported_datasets:
        raise ValueError(f"Unsupported dataset: {config['dataset']}. Supported datasets: {supported_datasets}")

    if config['model'] not in supported_models:
        raise ValueError(f"Unsupported model: {config['model']}. Supported models: {supported_models}")
    distribution_file = config['distribution_file']
    if not os.path.isfile(distribution_file):
        raise ValueError(f"Distribution file not found: {distribution_file}")

    print("✅ Config validation passed!")



