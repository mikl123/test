import os
import shutil
import onnxruntime
import numpy as np
from fire import Fire
import cv2
import pandas as pd
import time
from alike_step2 import ALike, configs
import copy
def preprocess_image(image):
    image = image.copy() / 255.0
    image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    image = np.expand_dims(image, axis=0)
    image = np.transpose(image, (0, 3, 1, 2))
    return image.astype(np.float32)


def compare_float_3d_arrays(arr1, arr2, error_margin=0.01):
    total_elements = arr1.size
    matching_elements = np.sum(np.abs(arr1 - arr2) <= error_margin)

    similarity = (matching_elements / total_elements) * 100
    print(f"The arrays are {similarity:.2f}% similar.")


def relative_difference_for_arrays(arr1, arr2):
    rel_diff = np.mean(np.abs(arr1 - arr2) / np.abs(arr1)) * 100
    print(f"Relative difference: {rel_diff:.2f}%")

class SimpleTracker(object):
    def __init__(self):
        self.pts_prev = None
        self.desc_prev = None

    def update(self, img, pts, desc):
        N_matches = 0
        if self.pts_prev is None:
            self.pts_prev = pts
            self.desc_prev = desc

            out = copy.deepcopy(img)
            for pt1 in pts:
                p1 = (int(round(pt1[0])), int(round(pt1[1])))
                cv2.circle(out, p1, 1, (0, 0, 255), -1, lineType=16)
        else:
            matches = self.mnn_mather(self.desc_prev, desc)
            mpts1, mpts2 = self.pts_prev[matches[:, 0]], pts[matches[:, 1]]
            N_matches = len(matches)

            out = copy.deepcopy(img)
            for pt1, pt2 in zip(mpts1, mpts2):
                p1 = (int(round(pt1[0])), int(round(pt1[1])))
                p2 = (int(round(pt2[0])), int(round(pt2[1])))
                cv2.line(out, p1, p2, (0, 255, 0), lineType=16)
                cv2.circle(out, p2, 1, (0, 0, 255), -1, lineType=16)

            self.pts_prev = pts
            self.desc_prev = desc

        return out, N_matches

    def mnn_mather(self, desc1, desc2):
        sim = desc1 @ desc2.transpose()
        sim[sim < 0.9] = 0
        nn12 = np.argmax(sim, axis=1)
        nn21 = np.argmax(sim, axis=0)
        ids1 = np.arange(0, sim.shape[0])
        mask = (ids1 == nn21[nn12])
        matches = np.stack([ids1[mask], nn12[mask]])
        return matches.transpose()

class SubGraphCompiler:
    def __init__(self, onnx_path, artifacts_folder):
        self.onnx_path = onnx_path
        self.artifacts_folder = artifacts_folder

        self.input_node = "img"
        self.input_shape = (1 ,480, 640, 3)
        self.output_shape = (1, 1, 65, 480, 640)
        self.calibration_frames = 5
        self.calibration_iterations = 5
        self.data = self.get_calibration_tensors()

    def get_compilation_options(self):
        return {
            "tidl_tools_path": os.environ.get("TIDL_TOOLS_PATH", "/home/workdir/tidl_tools"),
            "artifacts_folder": self.artifacts_folder,
            "accuracy_level": 1,
            "debug_level": 7,
            "tensor_bits": 8,
            "advanced_options:calibration_frames": self.calibration_frames,
            "advanced_options:calibration_iterations": self.calibration_iterations,
            "advanced_options:add_data_convert_ops": 1,
            "debugTraceLevel": 2,
        }

    def get_inference_options(self):
        return {
            "tidl_tools_path": os.environ.get("TIDL_TOOLS_PATH", "/home/workdir/tidl_tools"),
            "artifacts_folder": self.artifacts_folder,
            "debug_level": 7,
        }

    def generate_fixed_seed_arrays(self, seed=47):
        np.random.seed(seed)
        return [
            np.random.rand(*self.input_shape).astype(np.float32)
            for _ in range(self.calibration_frames)
        ]

    def get_calibration_tensors(self):
        calibration_dataset_path = "/home/workdir/assets/calibration_120823"
        data = pd.read_csv(os.path.join(calibration_dataset_path, "data.csv"))
        data["image_path"] = (
            f"{calibration_dataset_path}/"
            + data["image_folder_path"]
            + "/"
            + data["filename"]
        )

        preprocessed_images = []
        for row_index, row in data.iterrows():
            if row_index == self.calibration_frames:
                break
            image = cv2.cvtColor(cv2.imread(row["image_path"]), cv2.COLOR_BGR2RGB)
            preprocessed_images.append(preprocess_image(image.astype(np.float32)))
        
        return preprocessed_images
    def predict(self, step_1, step_2, data):
        start1 = time.time()
        output = step_1.run(None, {self.input_node: data.astype(np.float32)})
        print(str(time.time() - start1) + "step1")
        start1 = time.time()
        res = step_2(output[0],output[1])
        print(str(time.time() - start1) + "step2")
        return res
    def compile(self):
        if os.path.exists(self.artifacts_folder):
            shutil.rmtree(self.artifacts_folder)
        os.makedirs(self.artifacts_folder)
        compilation_session = onnxruntime.InferenceSession(
            self.onnx_path,
            providers=["TIDLCompilationProvider"],
            provider_options=[self.get_compilation_options()],
            sess_options=onnxruntime.SessionOptions(),
        )
        for _ in range(self.calibration_iterations):
            for inputs in self.data:
                print(inputs.shape)
                ort_inputs = {self.input_node: inputs}
                _ = compilation_session.run(None, ort_inputs)

    def inference(self):
        args = {"model":"alike-t",
        "input":r".\assets\tum",
        "device":"cpu",
        "top_k":-1,
        "scores_th":0.2,
        "n_limit":5000,
        "no_display":False
        }
        tracker = SimpleTracker()
        model = ALike(**configs[args["model"]],
                  device=args["device"],
                  top_k=args["top_k"],
                  scores_th=args["scores_th"],
                  n_limit=args["n_limit"])
        inference_tidl_session = onnxruntime.InferenceSession(
            self.onnx_path,
            providers=["TIDLExecutionProvider"],
            provider_options=[self.get_inference_options()],
            sess_options=onnxruntime.SessionOptions(),
        )
        for inputs in self.data[:5]:
            pred = self.predict(inference_tidl_session, model, inputs)
            kpts = pred["keypoints"]
            desc = pred["descriptors"]
            out, N_matches = tracker.update(inputs, kpts, desc)
            print(N_matches)
            if not args["no_display"]:
                cv2.imshow(args["model"], out)
                if cv2.waitKey(1) == ord('q'):
                    break

if __name__ == "__main__":
    Fire(SubGraphCompiler)
    print("Done")
