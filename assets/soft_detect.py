import torch
from torch import nn
import torch.nn.functional as F
import time
import multiprocessing as mp
import numpy as np
import threading
import onnxruntime as ort
# coordinates system
#  ------------------------------>  [ x: range=-1.0~1.0; w: range=0~W ]
#  | -----------------------------
#  | |                           |
#  | |                           |
#  | |                           |
#  | |         image             |
#  | |                           |
#  | |                           |
#  | |                           |
#  | |---------------------------|
#  v
# [ y: range=-1.0~1.0; h: range=0~H ]
lock = threading.Lock()
def max_pool(x,nms_radius,res,number):
        computed = torch.nn.functional.max_pool2d(x, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius)
        lock.acquire()
        try:
            res.append((number,computed))
        finally:
            lock.release()
def max_pool_parallel(x,nms_radius):
    res = []
    threads = []
    for i in range(0, x[0][0].size()[0],x[0][0].size()[0]//2):
        thread = threading.Thread(target=max_pool, args=(x[:,:,i:i+x[0][0].size()[0]//2,:],nms_radius,res,i))
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()
    res = sorted(res, key=lambda x: x[0])
    joined = torch.cat((res[0][1], res[1][1]), dim=2)
    return joined

def simple_nms(scores, nms_radius: int):
    """ Fast Non-maximum suppression to remove nearby points """
    assert (nms_radius >= 0)
    ort_session = ort.InferenceSession("max_pool_model.onnx")
    def max_pool(x):
        ort_inputs = {'input': x.numpy()}
        ort_outputs = ort_session.run(None, ort_inputs)
        output_tensor = torch.tensor(ort_outputs[0])
        return output_tensor
    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


def simple_nms1(scores, nms_radius: int):
    """ Fast Non-maximum suppression to remove nearby points """
    assert (nms_radius >= 0)

    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius)

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)

    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


def nms_fast(scores, dist_thresh = 3):
    """
    Run a faster approximate Non-Max-Suppression on numpy corners shaped:
      3xN [x_i,y_i,conf_i]^T
  
    Algo summary: Create a grid sized HxW. Assign each corner location a 1, rest
    are zeros. Iterate through all the 1's and convert them either to -1 or 0.
    Suppress points by setting nearby values to 0.
  
    Grid Value Legend:
    -1 : Kept.
     0 : Empty or suppressed.
     1 : To be processed (converted to either kept or supressed).
  
    NOTE: The NMS first rounds points to integers, so NMS distance might not
    be exactly dist_thresh. It also assumes points are within image boundaries.
  
    Inputs
      in_corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
      H - Image height.
      W - Image width.
      dist_thresh - Distance to suppress, measured as an infinty norm distance.
    Returns
      nmsed_corners - 3xN numpy matrix with surviving corners.
      nmsed_inds - N length numpy vector with surviving corner indices.
    """

    time1 = time.time()
    
    zeros = torch.zeros_like(scores)
    W, H = np.array(scores).shape[2:]

    in_corners = []
    for i, row in enumerate(scores[0][0]):
      for j, element in enumerate(row):
          in_corners.append([i, j, element])
    in_corners = np.array(in_corners).T
    print("Nms 1")
    print(time.time() - time1)

    time2 = time.time()
    grid = np.zeros((H, W)).astype(int) # Track NMS data.
    inds = np.zeros((H, W)).astype(int) # Store indices of points.
    # Sort by confidence and round to nearest int.
    inds1 = np.argsort(-in_corners[2,:])
    corners = in_corners[:,inds1]
    rcorners = corners[:2,:].round().astype(int) # Rounded corners.
    # Check for edge case of 0 or 1 corners.
    if rcorners.shape[1] == 0:
      return np.zeros((3,0)).astype(int), np.zeros(0).astype(int)
    if rcorners.shape[1] == 1:
      out = np.vstack((rcorners, in_corners[2])).reshape(3,1)
      return out, np.zeros((1)).astype(int)
    # Initialize the grid.
    for i, rc in enumerate(rcorners.T):
      grid[rcorners[1,i], rcorners[0,i]] = 1
      inds[rcorners[1,i], rcorners[0,i]] = i
    # Pad the border of the grid, so that we can NMS points near the border.
    pad = dist_thresh
    grid = np.pad(grid, ((pad,pad), (pad,pad)), mode='constant')
    # Iterate through points, highest to lowest conf, suppress neighborhood.
    count = 0
    for i, rc in enumerate(rcorners.T):
      # Account for top and left padding.
      pt = (rc[0]+pad, rc[1]+pad)
      if grid[pt[1], pt[0]] == 1: # If not yet suppressed.
        grid[pt[1]-pad:pt[1]+pad+1, pt[0]-pad:pt[0]+pad+1] = 0
        grid[pt[1], pt[0]] = -1
        count += 1
    # Get all surviving -1's and return sorted array of remaining corners.
    keepy, keepx = np.where(grid==-1)
    keepy, keepx = keepy - pad, keepx - pad
    inds_keep = inds[keepy, keepx]
    out = corners[:, inds_keep]
    values = out[-1, :]
    inds2 = np.argsort(-values)
    out = out[:, inds2]
    print("Nms 2")
    print(time.time() - time2)
    out_inds = inds1[inds_keep[inds2]]
    # return out, out_inds
    # zeros = torch.zeros_like(torch.tensor(scores))
    for point in out.T:
    #   print(point)
      zeros[:, :, int(point[0]), int(point[1])] = point[2]
    # out_inds = inds1[inds_keep[inds2]]

    return zeros

def sample_descriptor(descriptor_map, kpts, bilinear_interp=False):
    """
    :param descriptor_map: BxCxHxW
    :param kpts: list, len=B, each is Nx2 (keypoints) [h,w]
    :param bilinear_interp: bool, whether to use bilinear interpolation
    :return: descriptors: list, len=B, each is NxD
    """
    batch_size, channel, height, width = descriptor_map.shape

    descriptors = []
    for index in range(batch_size):
        kptsi = kpts[index]  # Nx2,(x,y)

        if bilinear_interp:
            descriptors_ = torch.nn.functional.grid_sample(descriptor_map[index].unsqueeze(0), kptsi.view(1, 1, -1, 2),
                                                           mode='bilinear', align_corners=True)[0, :, 0, :]  # CxN
        else:
            kptsi = (kptsi + 1) / 2 * kptsi.new_tensor([[width - 1, height - 1]])
            kptsi = kptsi.long()
            descriptors_ = descriptor_map[index, :, kptsi[:, 1], kptsi[:, 0]]  # CxN

        descriptors_ = torch.nn.functional.normalize(descriptors_, p=2, dim=0)
        descriptors.append(descriptors_.t())

    return descriptors


class DKD(nn.Module):
    def __init__(self, radius=2, top_k=0, scores_th=0.2, n_limit=20000):
        """
        Args:
            radius: soft detection radius, kernel size is (2 * radius + 1)
            top_k: top_k > 0: return top k keypoints
            scores_th: top_k <= 0 threshold mode:  scores_th > 0: return keypoints with scores>scores_th
                                                   else: return keypoints with scores > scores.mean()
            n_limit: max number of keypoint in threshold mode
        """
        super().__init__()
        self.radius = radius
        self.top_k = top_k
        self.scores_th = scores_th
        self.n_limit = n_limit
        self.kernel_size = 2 * self.radius + 1
        self.temperature = 0.1  # tuned temperature
        self.unfold = nn.Unfold(kernel_size=self.kernel_size, padding=self.radius)
        # Time measure
        self.average_all = []
        self.average_detect_keypoints = []
        self.average_sample_descriptors = []
        self.average_nms = [] 
        self.average_keypoint_other = []

        
        # local xy grid
        x = torch.linspace(-self.radius, self.radius, self.kernel_size)
        # (kernel_size*kernel_size) x 2 : (w,h)
        self.hw_grid = torch.stack(torch.meshgrid([x, x])).view(2, -1).t()[:, [1, 0]]

    def detect_keypoints(self, scores_map, sub_pixel=True):
        
        b, c, h, w = scores_map.shape
        scores_nograd = scores_map.detach()
        print("================")
        # nms_scores = simple_nms(scores_nograd, self.radius)
        start_time_NMS = time.time()
        nms_scores = simple_nms(scores_nograd, 3)
        self.average_nms.append(time.time() - start_time_NMS)
        
        time_start = time.time()
        # remove border
        nms_scores[:, :, :self.radius + 1, :] = 0
        nms_scores[:, :, :, :self.radius + 1] = 0
        nms_scores[:, :, h - self.radius:, :] = 0
        nms_scores[:, :, :, w - self.radius:] = 0
        # detect keypoints without grad


        if self.top_k > 0:
            topk = torch.topk(nms_scores.view(b, -1), self.top_k)
            indices_keypoints = topk.indices  # B x top_k
        else:
            if self.scores_th > 0:
                masks = nms_scores > self.scores_th
                if np.sum(np.array(masks)) == 0:
                    th = scores_nograd.reshape(b, -1).mean(dim=1)  # th = self.scores_th
                    masks = nms_scores > th.reshape(b, 1, 1, 1)
            else:
                th = scores_nograd.reshape(b, -1).mean(dim=1)  # th = self.scores_th
                masks = nms_scores > th.reshape(b, 1, 1, 1)
            masks = masks.reshape(b, -1)

            indices_keypoints = []  # list, B x (any size)
            scores_view = scores_nograd.reshape(b, -1)
            for mask, scores in zip(masks, scores_view):
                indices = mask.nonzero(as_tuple=False)[:, 0]
                if indices.size()[0] > self.n_limit:
                    kpts_sc = scores[indices]
                    sort_idx = kpts_sc.sort(descending=True)[1]
                    sel_idx = sort_idx[:self.n_limit]
                    indices = indices[sel_idx]
                indices_keypoints.append(indices)

        keypoints = []
        scoredispersitys = []
        kptscores = []
        
        if sub_pixel:
            # detect soft keypoints with grad backpropagation
            patches = self.unfold(scores_map)  # B x (kernel**2) x (H*W)
            self.hw_grid = self.hw_grid.to(patches)  # to device
            for b_idx in range(b):
                patch = patches[b_idx].t()  # (H*W) x (kernel**2)
                indices_kpt = indices_keypoints[b_idx]  # one dimension vector, say its size is M
                patch_scores = patch[indices_kpt]  # M x (kernel**2)

                # max is detached to prevent undesired backprop loops in the graph
                max_v = patch_scores.max(dim=1).values.detach()[:, None]
                x_exp = ((patch_scores - max_v) / self.temperature).exp()  # M * (kernel**2), in [0, 1]

                # \frac{ \sum{(i,j) \times \exp(x/T)} }{ \sum{\exp(x/T)} }
                xy_residual = x_exp @ self.hw_grid / x_exp.sum(dim=1)[:, None]  # Soft-argmax, Mx2

                hw_grid_dist2 = torch.norm((self.hw_grid[None, :, :] - xy_residual[:, None, :]) / self.radius,
                                           dim=-1) ** 2
                scoredispersity = (x_exp * hw_grid_dist2).sum(dim=1) / x_exp.sum(dim=1)

                # compute result keypoints
                keypoints_xy_nms = torch.stack([indices_kpt % w, indices_kpt // w], dim=1)  # Mx2
                keypoints_xy = keypoints_xy_nms + xy_residual
                keypoints_xy = keypoints_xy / keypoints_xy.new_tensor(
                    [w - 1, h - 1]) * 2 - 1  # (w,h) -> (-1~1,-1~1)

                kptscore = torch.nn.functional.grid_sample(scores_map[b_idx].unsqueeze(0),
                                                           keypoints_xy.view(1, 1, -1, 2),
                                                           mode='bilinear', align_corners=True)[0, 0, 0, :]  # CxN

                keypoints.append(keypoints_xy)
                scoredispersitys.append(scoredispersity)
                kptscores.append(kptscore)
        else:
            for b_idx in range(b):
                indices_kpt = indices_keypoints[b_idx]  # one dimension vector, say its size is M
                keypoints_xy_nms = torch.stack([indices_kpt % w, indices_kpt // w], dim=1)  # Mx2
                keypoints_xy = keypoints_xy_nms / keypoints_xy_nms.new_tensor(
                    [w - 1, h - 1]) * 2 - 1  # (w,h) -> (-1~1,-1~1)
                kptscore = torch.nn.functional.grid_sample(scores_map[b_idx].unsqueeze(0),
                                                           keypoints_xy.view(1, 1, -1, 2),
                                                           mode='bilinear', align_corners=True)[0, 0, 0, :]  # CxN
                keypoints.append(keypoints_xy)
                scoredispersitys.append(None)
                kptscores.append(kptscore)
        self.average_keypoint_other.append(time.time() - time_start)
        return keypoints, scoredispersitys, kptscores

    def forward(self, scores_map, descriptor_map, sub_pixel=False):
        """
        :param scores_map:  Bx1xHxW
        :param descriptor_map: BxCxHxW
        :param sub_pixel: whether to use sub-pixel keypoint detection
        :return: kpts: list[Nx2,...]; kptscores: list[N,....] normalised position: -1.0 ~ 1.0
        """

        time_start = time.time()
        time_detection_keypoint_start = time.time()
        keypoints, scoredispersitys, kptscores = self.detect_keypoints(scores_map,
                                                                       sub_pixel)
        self.average_detect_keypoints.append(time.time() - time_detection_keypoint_start)
        
        time_descriptor_start = time.time()
        descriptors = sample_descriptor(descriptor_map, keypoints, sub_pixel)
        
        self.average_sample_descriptors.append(time.time() - time_descriptor_start)
        self.average_all.append(time.time() - time_start)
       
        # keypoints: B M 2
        # descriptors: B M D
        # scoredispersitys:
        return keypoints, descriptors, kptscores, scoredispersitys
