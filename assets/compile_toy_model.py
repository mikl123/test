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
from copy import deepcopy
from torchvision.transforms import ToTensor
from tqdm import tqdm
import torch
import numpy as np
from pathlib import Path
import torch.utils.data as data

def generate_read_function(method, extension='ppm'):
    def read_function(seq_name, im_idx):
        aux = np.load(os.path.join(dataset_path, seq_name, '%d.%s.%s' % (im_idx, extension, method)))
        if top_k is None:
            return aux['keypoints'], aux['descriptors']
        else:
            assert ('scores' in aux)
            ids = np.argsort(aux['scores'])[-top_k:]
            return aux['keypoints'][ids, :], aux['descriptors'][ids, :]

    return read_function


def mnn_matcher(descriptors_a, descriptors_b):
    device = descriptors_a.device
    sim = descriptors_a @ descriptors_b.t()
    nn12 = torch.max(sim, dim=1)[1]
    nn21 = torch.max(sim, dim=0)[1]
    ids1 = torch.arange(0, sim.shape[0], device=device)
    mask = (ids1 == nn21[nn12])
    matches = torch.stack([ids1[mask], nn12[mask]])
    return matches.t().data.cpu().numpy()


def homo_trans(coord, H):
    kpt_num = coord.shape[0]
    homo_coord = np.concatenate((coord, np.ones((kpt_num, 1))), axis=-1)
    proj_coord = np.matmul(H, homo_coord.T).T
    proj_coord = proj_coord / proj_coord[:, 2][..., None]
    proj_coord = proj_coord[:, 0:2]
    return proj_coord


def benchmark_features(read_feats):
    lim = [1, 5]
    rng = np.arange(lim[0], lim[1] + 1)

    seq_names = sorted(os.listdir(dataset_path))

    n_feats = []
    n_matches = []
    seq_type = []
    i_err = {thr: 0 for thr in rng}
    v_err = {thr: 0 for thr in rng}

    i_err_homo = {thr: 0 for thr in rng}
    v_err_homo = {thr: 0 for thr in rng}

    for seq_idx, seq_name in tqdm(enumerate(seq_names), total=len(seq_names)):
        keypoints_a, descriptors_a = read_feats(seq_name, 1)
        n_feats.append(keypoints_a.shape[0])

        # =========== compute homography
        ref_img = cv2.imread(os.path.join(dataset_path, seq_name, '1.ppm'))
        ref_img_shape = ref_img.shape

        for im_idx in range(2, 7):
            keypoints_b, descriptors_b = read_feats(seq_name, im_idx)
            n_feats.append(keypoints_b.shape[0])

            matches = mnn_matcher(
                torch.from_numpy(descriptors_a).to(device=device),
                torch.from_numpy(descriptors_b).to(device=device)
            )

            homography = np.loadtxt(os.path.join(dataset_path, seq_name, "H_1_" + str(im_idx)))

            pos_a = keypoints_a[matches[:, 0], : 2]
            pos_a_h = np.concatenate([pos_a, np.ones([matches.shape[0], 1])], axis=1)
            pos_b_proj_h = np.transpose(np.dot(homography, np.transpose(pos_a_h)))
            pos_b_proj = pos_b_proj_h[:, : 2] / pos_b_proj_h[:, 2:]

            pos_b = keypoints_b[matches[:, 1], : 2]

            dist = np.sqrt(np.sum((pos_b - pos_b_proj) ** 2, axis=1))

            n_matches.append(matches.shape[0])
            seq_type.append(seq_name[0])

            if dist.shape[0] == 0:
                dist = np.array([float("inf")])

            for thr in rng:
                if seq_name[0] == 'i':
                    i_err[thr] += np.mean(dist <= thr)
                else:
                    v_err[thr] += np.mean(dist <= thr)

            # =========== compute homography
            gt_homo = homography
            pred_homo, _ = cv2.findHomography(keypoints_a[matches[:, 0], : 2], keypoints_b[matches[:, 1], : 2],
                                              cv2.RANSAC)
            if pred_homo is None:
                homo_dist = np.array([float("inf")])
            else:
                corners = np.array([[0, 0],
                                    [ref_img_shape[1] - 1, 0],
                                    [0, ref_img_shape[0] - 1],
                                    [ref_img_shape[1] - 1, ref_img_shape[0] - 1]])
                real_warped_corners = homo_trans(corners, gt_homo)
                warped_corners = homo_trans(corners, pred_homo)
                homo_dist = np.mean(np.linalg.norm(real_warped_corners - warped_corners, axis=1))

            for thr in rng:
                if seq_name[0] == 'i':
                    i_err_homo[thr] += np.mean(homo_dist <= thr)
                else:
                    v_err_homo[thr] += np.mean(homo_dist <= thr)

    seq_type = np.array(seq_type)
    n_feats = np.array(n_feats)
    n_matches = np.array(n_matches)

    return i_err, v_err, i_err_homo, v_err_homo, [seq_type, n_feats, n_matches]


do_testing = True


# Testing ------------------------------------------------------------------
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

methods = ['new']
names = ['new']

top_k = None
n_i = 20
n_v = 5
cache_dir = 'hseq/cache'
dataset_path = 'hseq/hpatches-sequences-release'
# Testing ------------------------------------------------------------------





# Extraction ----------------------------------------------------------------------------


class HPatchesDataset(data.Dataset):
    def __init__(self, root: str = dataset_path, alteration: str = 'all'):
        """
        Args:
            root: dataset root path
            alteration: # 'all', 'i' for illumination or 'v' for viewpoint
        """
        assert (Path(root).exists()), f"Dataset root path {root} dose not exist!"
        self.root = root

        # get all image file name
        self.image0_list = []
        self.image1_list = []
        self.homographies = []
        folders = [x for x in Path(self.root).iterdir() if x.is_dir()]
        self.seqs = []
        for folder in folders:
            if alteration == 'i' and folder.stem[0] != 'i':
                continue
            if alteration == 'v' and folder.stem[0] != 'v':
                continue

            self.seqs.append(folder)

        self.len = len(self.seqs)
        assert (self.len > 0), f'Can not find PatchDataset in path {self.root}'

    def __getitem__(self, item):
        folder = self.seqs[item]

        imgs = []
        homos = []
        for i in range(1, 7):
            img = cv2.imread(str(folder / f'{i}.ppm'), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # HxWxC
            imgs.append(img)

            if i != 1:
                homo = np.loadtxt(str(folder / f'H_1_{i}')).astype('float32')
                homos.append(homo)

        return imgs, homos, folder.stem

    def __len__(self):
        return self.len

    def name(self):
        return self.__class__


def extract_multiscale(self, model, img, scale_f=2 ** 0.5,
                       min_scale=1., max_scale=1.,
                       min_size=0., max_size=99999.,
                       image_size_max=99999,
                       n_k=0, sort=False):
    H_, W_, three = img.shape
    assert three == 3, "input image shape should be [HxWx3]"

    old_bm = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = False  # speedup

    # ==================== image size constraint
    image = deepcopy(img)
    max_hw = max(H_, W_)
    if max_hw > image_size_max:
        ratio = float(image_size_max / max_hw)
        image = cv2.resize(image, dsize=None, fx=ratio, fy=ratio)

    # ==================== convert image to tensor
    H, W, three = image.shape
    image = ToTensor()(image).unsqueeze(0)
    image = image.to(device)

    s = 1.0  # current scale factor
    keypoints, descriptors, scores, scores_maps, descriptor_maps = [], [], [], [], []
    while s + 0.001 >= max(min_scale, min_size / max(H, W)):
        if s - 0.001 <= min(max_scale, max_size / max(H, W)):
            nh, nw = image.shape[2:]

            # extract descriptors
            with torch.no_grad():
                # descriptor_map, scores_map = model.extract_dense_map(image)
                # keypoints_, descriptors_, scores_, _ = model.dkd(scores_map, descriptor_map)

                args = {"model":"alike-t",
                "input":r".\assets\tum",
                "device":"cpu",
                "top_k":-1,
                "scores_th":0.2,
                "n_limit":5000,
                "no_display":False
                }
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
                pred = self.predict(inference_tidl_session, model, image)
                keypoints_ = pred["keypoints"]
                descriptors_ = pred["descriptors"]
                scores_ = pred["scores"]
































            keypoints.append(keypoints_[0])
            descriptors.append(descriptors_[0])
            scores.append(scores_[0])

        s /= scale_f

        # down-scale the image for next iteration
        nh, nw = round(H * s), round(W * s)
        image = torch.nn.functional.interpolate(image, (nh, nw), mode='bilinear', align_corners=False)

    # restore value
    torch.backends.cudnn.benchmark = old_bm

    keypoints = torch.cat(keypoints)
    descriptors = torch.cat(descriptors)
    scores = torch.cat(scores)
    keypoints = (keypoints + 1) / 2 * keypoints.new_tensor([[W_ - 1, H_ - 1]])

    if sort or 0 < n_k < len(keypoints):
        indices = torch.argsort(scores, descending=True)
        keypoints = keypoints[indices]
        descriptors = descriptors[indices]
        scores = scores[indices]

    if 0 < n_k < len(keypoints):
        keypoints = keypoints[0:n_k]
        descriptors = descriptors[0:n_k]
        scores = scores[0:n_k]

    return {'keypoints': keypoints, 'descriptors': descriptors, 'scores': scores}


def extract_method(self, m):
    hpatches = HPatchesDataset(root=dataset_path, alteration='all')
    model = m[:7]
    min_scale = 0.3 if m[8:] == 'ms' else 1.0

    model = ALike(**configs[model], device=device, top_k=0, scores_th=0.2, n_limit=5000)

    progbar = tqdm(hpatches, desc='Extracting for {}'.format(m))
    for imgs, homos, seq_name in progbar:
        for i in range(1, 7):
            img = imgs[i - 1]
            pred = extract_multiscale(model, img, min_scale=min_scale, max_scale=1, sort=False, n_k=5000)
            kpts, descs, scores = pred['keypoints'], pred['descriptors'], pred['scores']

            with open(os.path.join(dataset_path, seq_name, f'{i}.ppm.{m}'), 'wb') as f:
                np.savez(f, keypoints=kpts.cpu().numpy(),
                         scores=scores.cpu().numpy(),
                         descriptors=descs.cpu().numpy())



# Extraction ---------------------------------------------------------------------------------------------------






time_total = 0
time_step1 = 0
time_step2 = 0
time_dkd = 0
time_normalize = 0
time_convertion_tensor = 0

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
            "debug_level": 0,
            "tensor_bits": 8,
            "advanced_options:calibration_frames": self.calibration_frames,
            "advanced_options:calibration_iterations": self.calibration_iterations,
            "advanced_options:add_data_convert_ops": 1,
            "debugTraceLevel": 0,
        }

    def get_inference_options(self):
        return {
            "tidl_tools_path": os.environ.get("TIDL_TOOLS_PATH", "/home/workdir/tidl_tools"),
            "artifacts_folder": self.artifacts_folder,
            "debug_level": 0,
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

        start_total = time.time()
        output = step_1.run(None, {self.input_node: data.astype(np.float32)})
        time_step1 += time.time() - start_total

        start_step2 = time.time()
        res = step_2(output[0],output[1])
        time_step2 += time.time() - start_step2
        
        time_total += time.time() - start_total

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


            # out, N_matches = tracker.update(inputs, kpts, desc)
            # print(N_matches)
            # if not args["no_display"]:
            #     cv2.imshow(args["model"], out)
            #     if cv2.waitKey(1) == ord('q'):
            #         break
        print("Averaged total time: ", str(time_total / 5))
        print("Averaged step1 time: ", str(time_step1 / 5))
        print("Averaged step2 time: ", str(time_step2 / 5))

        print ("----------------------------Detailed STEP 2---------------------------------")
        print("Average convertion", str(model.time_convertion / 5))
        print("Average normalize", str(model.time_normalize / 5))
        print("Average DKD", str(model.time_dkd / 5))


    # errors = {}
    # for method in methods:
    #     output_file = os.path.join(cache_dir, method + '.npy')
    #     read_function = generate_read_function(method)
    #     if os.path.exists(output_file):
    #         errors[method] = np.load(output_file, allow_pickle=True)
    #     else:
    #         extract_method(method)
    #         errors[method] = benchmark_features(read_function)
    #         np.save(output_file, errors[method])

    # for name, method in zip(names, methods):
    #     i_err, v_err, i_err_hom, v_err_hom, _ = errors[method]

    #     print(f"====={name}=====")
    #     print(f"MMA@1 MMA@2 MMA@3 MHA@1 MHA@2 MHA@3: ", end='')
    #     for thr in range(1, 4):
    #         err = (i_err[thr] + v_err[thr]) / ((n_i + n_v) * 5)
    #         print(f"{err * 100:.2f}%", end=' ')
    #     for thr in range(1, 4):
    #         err_hom = (i_err_hom[thr] + v_err_hom[thr]) / ((n_i + n_v) * 5)
    #         print(f"{err_hom * 100:.2f}%", end=' ')
    #     print('')




    errors = {}
    for method in methods:
        output_file = os.path.join(cache_dir, method + '.npy')
        read_function = generate_read_function(method)
        if os.path.exists(output_file):
            errors[method] = np.load(output_file, allow_pickle=True)
        else:
            extract_method(method)
            errors[method] = benchmark_features(read_function)
            np.save(output_file, errors[method])

    for name, method in zip(names, methods):
        i_err, v_err, i_err_hom, v_err_hom, _ = errors[method]

        print(f"====={name}=====")
        print(f"MMA@1 MMA@2 MMA@3 MHA@1 MHA@2 MHA@3: ", end='')
        for thr in range(1, 4):
            err = (i_err[thr] + v_err[thr]) / ((n_i + n_v) * 5)
            print(f"{err * 100:.2f}%", end=' ')
        for thr in range(1, 4):
            err_hom = (i_err_hom[thr] + v_err_hom[thr]) / ((n_i + n_v) * 5)
            print(f"{err_hom * 100:.2f}%", end=' ')
        print('')


if __name__ == "__main__":
    Fire(SubGraphCompiler)
    print("Done")
