import depthai as dai
import logging

# Configure the logging
logging.basicConfig(filename='error.log', level=logging.ERROR, format='%(asctime)s - %(levelname)s: %(message)s')


def create_pipeline():
    pipeline = dai.Pipeline()
    colour_camera = pipeline.create(dai.node.ColorCamera)
    colour_camera.setPreviewSize(416, 416)
    colour_camera.setBoardSocket(dai.CameraBoardSocket.RGB)
    colour_camera.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
    colour_camera.setInterleaved(False)
    x_out_colour = pipeline.create(dai.node.XLinkOut)
    x_out_colour.setStreamName("rgb")
    colour_camera.preview.link(x_out_colour.input)
    return pipeline


def setup_camera(device_id):
    try:
        pipeline = create_pipeline()
        device = dai.Device(device_id)
        device.startPipeline(pipeline)
        q_rgb = device.getOutputQueue(name="rgb", maxSize=1, blocking=True)
        return device, q_rgb
    except RuntimeError as e:
        camera_location = "front" if device_id == "19443010B118F81200" else (
            "rear" if device_id == "19443010F19B281300" else None)
        # Write an error message to the log
        logging.error(f"Warning: Could not connect to device with ID {device_id} ({camera_location})."
                      f"Please check if the device is connected.")
        return None, None
