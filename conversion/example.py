import cv2
import depthai
import numpy as np
import blobconverter
import os
import shutil
from export import export_test
import subprocess


def main():
    clear_exports()
    export_test()
    optimize_model("test")
    convert_blob("test")
    pipeline = setup_pipeline()
    run_pipeline(pipeline)


def clear_exports():
    try:
        shutil.rmtree("./exports")
    except FileNotFoundError:
        pass
    os.mkdir("./exports")


def optimize_model(name):
    optimizer_path = "/opt/intel/openvino_2021/" + \
                     "deployment_tools/model_optimizer/mo.py"
    subprocess.run(["python3", optimizer_path,
                    "--input_model", f"./exports/{name}.onnx",
                    "--data_type", "FP16"])
    os.rename(name + ".bin", os.path.join("./exports", name + ".bin"))
    os.rename(name + ".xml", os.path.join("./exports", name + ".xml"))
    os.rename(name + ".mapping", os.path.join("./exports", name + ".mapping"))


def convert_blob(name):
    xmlfile = os.path.join("./exports", name) + ".xml"
    binfile = os.path.join("./exports", name) + ".bin"

    blob_path = blobconverter.from_openvino(
        xml = xmlfile,
        bin = binfile,
        data_type = "FP16",
        shaves = 5,
        output_dir = "./exports"
    )


def setup_pipeline():
    pipeline = depthai.Pipeline()

    cam_rgb = pipeline.createColorCamera()
    cam_rgb.setPreviewSize(256, 256)
    cam_rgb.setInterleaved(False)

    detection_nn = pipeline.createNeuralNetwork()
    detection_nn.setBlobPath("./exports/test_openvino_2021.4_5shave.blob")
    cam_rgb.preview.link(detection_nn.input)

    xout_nn = pipeline.createXLinkOut()
    xout_nn.setStreamName("nn")
    detection_nn.out.link(xout_nn.input)

    return pipeline


def run_pipeline(pipeline):
    with depthai.Device(pipeline) as device:
        q_nn = device.getOutputQueue("nn", maxSize=4, blocking=False)
        while True:
            in_nn = q_nn.tryGet()
            if in_nn is not None:
                # get data
                output = in_nn.getAllLayerNames()[-1]
                data = np.array(in_nn.getLayerFp16(output))

                # format data as image
                data = data.reshape(3, 256, 256).transpose(1, 2, 0).astype(np.uint8)
                data = cv2.resize(data, (1024, 1020))

                # show image
                cv2.imshow("preview", data)

            # quit if user presses q
            if cv2.waitKey(1) == ord('q'):
                break


if __name__ == "__main__":
    main()
