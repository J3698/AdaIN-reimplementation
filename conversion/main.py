import cv2
import depthai
import numpy as np
import blobconverter
import os
import shutil
from export import export_test, export_stylizer
import subprocess
from termcolor import colored

def main():
    clear_exports()
    print(colored('Cleared Exports', 'green'))
    export_stylizer()
    print(colored('Exported stylizer', 'green'))
    optimize_model("stylizer")
    print(colored('Optimized model', 'green'))
    compile_to_blob("stylizer")
    print(colored('Compiled to blob', 'green'))
    pipeline = setup_pipeline("stylizer")
    print(colored('Setup pipeline', 'green'))
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


def compile_to_blob(name):
    compiler_path = "/opt/intel/openvino_2021/deployment_tools/tools/compile_tool/compile_tool"
    subprocess.run([compiler_path, "-m", f"./exports/{name}.xml",
                    "-ip", "U8", "-d", "MYRIAD",
                    "-VPU_NUMBER_OF_SHAVES", "7",
                    "-VPU_NUMBER_OF_CMX_SLICES", "7",
                    "-o", f"exports/{name}.blob"])


def setup_pipeline(name):
    pipeline = depthai.Pipeline()

    cam_rgb = pipeline.createColorCamera()
    cam_rgb.setPreviewSize(256, 256)
    cam_rgb.setInterleaved(False)

    detection_nn = pipeline.createNeuralNetwork()
    detection_nn.setBlobPath(f"./exports/{name}.blob")
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
