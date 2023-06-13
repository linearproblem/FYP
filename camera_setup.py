import depthai as dai
import logging
import json

# Configure the logging
logging.basicConfig(filename='error.log', level=logging.ERROR, format='%(asctime)s - %(levelname)s: %(message)s')


def apply_camera_settings(device_id, camera):
    """
    Sets camera focus, exposure, iso and some other image manipulations before the image reaches the host
    :param device_id: this is the device serial number that is used to differentiate between the two cameras
    :param camera: the colour camera node, that the settings will be applied to
    :return:
    """
    default_settings = {
        'lens_position': 100,
        'iso': 1500,
        'exposure': 13500,
        'contrast': -1,
        'brightness': -1,
        'saturation': 8,
        'sharpness': 3,
        'manual_white_balance': 4000,
        'luma_denoise': 3,
        'chroma_denoise': 4
    }

    if device_id == "19443010B118F81200":  # Front (Oak-D Lite)
        default_settings['lens_position'] = 67
        default_settings['exposure'] = 13500
        default_settings['iso'] = 1500
        default_settings['manual_white_balance'] = 3600

    elif device_id == "19443010F19B281300":  # Rear (Oak-1)
        default_settings['lens_position'] = 141
        default_settings['exposure'] = 9000  # 3000
        default_settings['iso'] = 1500
        default_settings['manual_white_balance'] = 6000

    camera.initialControl.setManualFocus(default_settings['lens_position'])
    camera.initialControl.setAutoExposureLock(True)
    camera.initialControl.setManualExposure(default_settings['exposure'], default_settings['iso'])
    camera.initialControl.setManualWhiteBalance(default_settings['manual_white_balance'])
    camera.initialControl.setBrightness(default_settings['brightness'])
    camera.initialControl.setContrast(default_settings['contrast'])
    camera.initialControl.setSaturation(default_settings['saturation'])
    camera.initialControl.setSharpness(default_settings['sharpness'])
    camera.initialControl.setLumaDenoise(default_settings['luma_denoise'])
    camera.initialControl.setChromaDenoise(default_settings['chroma_denoise'])


def setup_object_detection(neural_network=None):
    blob_path = 'model/model.blob'
    config_path = 'model/config.json'

    # Load JSON configuration data
    with open(config_path, 'r') as file:
        config_data = json.load(file)

    # Extract YOLO parameters
    yolo_params = config_data['nn_config']['NN_specific_metadata']
    classes = yolo_params['classes']
    coordinates = yolo_params['coordinates']
    anchors = yolo_params['anchors']
    anchor_masks = yolo_params['anchor_masks']
    iou_threshold = yolo_params['iou_threshold']
    confidence_threshold = yolo_params['confidence_threshold']
    confidence_threshold = 0.5  # Override confidence setting

    # Set up YOLO detection network
    if neural_network is not None:
        neural_network.setBlobPath(blob_path)
        neural_network.setConfidenceThreshold(confidence_threshold)
        neural_network.setNumClasses(classes)
        neural_network.setCoordinateSize(coordinates)
        neural_network.setAnchors(anchors)
        neural_network.setAnchorMasks(anchor_masks)
        neural_network.setIouThreshold(iou_threshold)
        neural_network.input.setBlocking(False)

    # Extract and return labels
    labels = config_data['mappings']['labels'][:]
    return labels


def create_pipeline(device_id):
    pipeline = dai.Pipeline()

    # Define the camera node and its properties
    colour_camera = pipeline.create(dai.node.ColorCamera)
    colour_camera.setPreviewSize(416, 416)
    colour_camera.setBoardSocket(dai.CameraBoardSocket.RGB)
    colour_camera.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
    colour_camera.setInterleaved(False)
    #colour_camera.setFps(5)
    colour_camera.setFps(30)
    # Apply the camera settings for this device
    apply_camera_settings(device_id, colour_camera)

    # This can be used to increase NN size and therefore the fov
    # colour_camera.setPreviewKeepAspectRatio(False)

    # Define the neural network node and its properties
    neural_network = pipeline.create(dai.node.YoloDetectionNetwork)
    labels = setup_object_detection(neural_network)  # Set up the object detection node

    # Create the output node for the camera's preview to be fed into the NN
    x_out_colour = pipeline.create(dai.node.XLinkOut)
    x_out_colour.setStreamName("rgb")
    colour_camera.preview.link(neural_network.input)  # Links 416x416 preview to NN

    # Create output node for detections
    x_out_detection = pipeline.create(dai.node.XLinkOut)
    x_out_detection.setStreamName("detection")
    neural_network.out.link(x_out_detection.input)

    # Pass 4k Video out - Note this is if skipping the cropping that happens below
    # colour_camera.preview.link(x_out_colour.input) # 416x416 direct to queue
    # neural_network.passthrough.link(x_out_colour.input)# 416x416 to queue via NN passthrough
    # colour_camera.video.link(x_out_colour.input)

    # Create ImageManip node to crop the camera output - just another step on the camera
    manip = pipeline.create(dai.node.ImageManip)
    manip.initialConfig.setCropRect(0.21875, 0, 0.78125, 1)  # Adjust these parameters to define the crop area
    colour_camera.video.link(manip.inputImage)
    manip.setMaxOutputFrameSize(7000000)  # Adjust this value based on your needs, image approx 6.7MB
    manip.out.link(x_out_colour.input)

    return pipeline


def setup_camera(device_id):
    try:
        pipeline = create_pipeline(device_id)
        device = dai.Device(device_id)
        # device = dai.Device(pipeline, maxUsbSpeed=dai.UsbSpeed.SUPER)
        device.startPipeline(pipeline)
        device.setLogLevel(dai.LogLevel.DEBUG)
        device.setLogOutputLevel(dai.LogLevel.DEBUG)

        q_rgb = device.getOutputQueue(name="rgb", maxSize=1, blocking=False)
        q_detection = device.getOutputQueue(name="detection", maxSize=1, blocking=False)
        return device, q_rgb, q_detection
    except Exception as e:
        camera_location = "front" if device_id == "19443010B118F81200" else (
            "rear" if device_id == "19443010F19B281300" else None)
        # Write an error message to the log
        logging.error(f"Warning: Could not connect to device with ID {device_id} ({camera_location})."
                      f"Please check if the device is connected. Exception:{e}.")
        return None, None, None
