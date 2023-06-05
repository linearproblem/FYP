import yaml


# Load yaml file
def load_yaml(file_path):
    with open(file_path, 'r') as f:
        yaml_data = yaml.safe_load(f)
    return yaml_data


# Check if the feature is turned on for a specific camera
def update_camera_features(features, feature_details, camera):
    # Return a dictionary that indicates if each feature is turned on for the specified camera
    return {feature: 'location' in feature_details and feature_details['location'].get(camera, True) for feature in
            features}


# Get settings for the features that are in use
def get_feature_settings(camera_features, bottle_settings):
    feature_settings = {}
    for feature, key in camera_features.items():
        if key is True:
            feature_settings[feature] = {}
            # If the feature has a 'value', store it in the settings
            if 'value' in data[feature]:
                feature_settings[feature]['value'] = bottle_settings[feature]['value']
            # If the feature has a 'region', store it in the settings
            if 'region' in bottle_settings[feature]:
                region = data[feature]['region']
                feature_settings[feature].update({k: region.get(k) for k in ('x0', 'y0', 'x1', 'y1')})
            # If the feature has 'dimensions', store them in the settings
            if 'dimensions' in bottle_settings[feature]:
                dimensions = bottle_settings[feature]['dimensions']
                feature_settings[feature].update({k: dimensions.get(k) for k in ('width', 'height')})
    return feature_settings


# Main function to process the settings from a yaml file
def get_bottle_settings(input_file_path):
    # Load the data from yaml file
    data = load_yaml(input_file_path)

    # Determine which features are active for the front and rear cameras
    front_camera_features = update_camera_features(data.keys(), data, 'front')
    rear_camera_features = update_camera_features(data.keys(), data, 'rear')

    # Get the settings for the active features for both front and rear cameras
    enabled_front_feature_settings = get_feature_settings(front_camera_features, data)
    enabled_rear_feature_settings = get_feature_settings(rear_camera_features, data)

    # Return features and their settings for both cameras
    return front_camera_features, rear_camera_features, enabled_front_feature_settings, enabled_rear_feature_settings
