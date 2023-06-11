import yaml

def load_yaml(file_path):
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def get_bottle_settings(input_file_path):
    quality_checks = load_yaml(input_file_path)
    return quality_checks
