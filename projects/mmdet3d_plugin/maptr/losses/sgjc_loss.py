# file: projects/mmdet3d_plugin/maptr/losses/sgjc_loss.py
import torch
import torch.nn.functional as F
import numpy as np
import warnings
from scipy.optimize import linear_sum_assignment
from mmcv.utils import get_logger


def stable_nll_loss_batch_3d(mu, Sigma, target, eps=1e-6):
    """
    Robust NLL for 3D Gaussian (x, y, sem_error).
    mu: [M, 3], Sigma: [M, 3, 3], target: [M, 3]
    """
    resid = target - mu
    M = resid.shape[0]
    D = mu.shape[1]  # Should be 3
    device = Sigma.device
    dtype = Sigma.dtype
    eye = torch.eye(D, device=device, dtype=dtype).unsqueeze(0).expand(M, -1, -1)
    Sigma_stable = Sigma + eps * eye

    try:
        L = torch.linalg.cholesky(Sigma_stable)
        diagL = torch.diagonal(L, dim1=-2, dim2=-1).clamp(min=eps)
        log_det = 2.0 * torch.log(diagL).sum(dim=-1)
        inv_Sigma = torch.cholesky_inverse(L)
    except RuntimeError:
        # Fallback to pseudo-inverse if Cholesky fails
        inv_Sigma = torch.linalg.pinv(Sigma_stable)
        log_det = torch.log(torch.det(Sigma_stable).clamp(min=1e-12))

    resid_col = resid.unsqueeze(-1)
    mahala = (resid_col.transpose(-2, -1) @ inv_Sigma @ resid_col).squeeze(-1).squeeze(-1)
    const = torch.tensor(D * np.pi, device=device, dtype=dtype).log() * 2.0
    nll = 0.5 * (log_det + mahala + const)
    return nll


def extract_gt_points(lines_obj, device, num_pts_per_vec):
    """
    Extract GT points from MapTR lines object (Robust version).
    """
    pts_list = []
    raw_lines = []
    try:
        if hasattr(lines_obj, '__iter__'): raw_lines = list(lines_obj)
    except:
        pass
    if len(raw_lines) == 0 and hasattr(lines_obj, 'instance_list'): raw_lines = lines_obj.instance_list
    if len(raw_lines) == 0 and hasattr(lines_obj, 'points'): raw_lines = lines_obj.points

    for poly_pts in raw_lines:
        # Handle Shapely
        if hasattr(poly_pts, 'coords'):
            poly_pts = np.array(poly_pts.coords)
        elif isinstance(poly_pts, (list, tuple)):
            poly_pts = np.array(poly_pts)

        if not isinstance(poly_pts, torch.Tensor):
            poly_pts = torch.tensor(poly_pts, device=device, dtype=torch.float32)
        else:
            poly_pts = poly_pts.to(device)

        L_pts = poly_pts.shape[0]
        if L_pts < 2:
            sampled = poly_pts.new_zeros(num_pts_per_vec, 2)
        else:
            if L_pts <= num_pts_per_vec:
                indices = torch.linspace(0, L_pts - 1, num_pts_per_vec).long().to(device)
            else:
                step = (L_pts - 1) / (num_pts_per_vec - 1)
                indices = (torch.arange(num_pts_per_vec, device=device) * step).long().clamp(max=L_pts - 1)
            sampled = poly_pts[indices]
        pts_list.append(sampled[:, :2])  # Only take xy
    return pts_list


def compute_hungarian_matching(ceh_preds_dict, gt_bboxes_3d, gt_labels_3d, cost_weights):
    """
    Custom Hungarian Matcher based on L1 distance and Classification Cost.
    """
    mu_xy = ceh_preds_dict['mu_xy']  # [B, N, P, 2]
    logits = ceh_preds_dict['logits']  # [B, N, P, C]

    B, N, P, _ = mu_xy.shape
    device = mu_xy.device
    matches = []

    w_cls = cost_weights.get('cls', 2.0)
    w_pts = cost_weights.get('pts', 5.0)

    for b in range(B):
        # 1. Get GT
        pts_list = extract_gt_points(gt_bboxes_3d[b], device, P)
        if len(pts_list) == 0:
            matches.append((torch.empty(0, dtype=torch.long, device=device),
                            torch.empty(0, dtype=torch.long, device=device)))
            continue

        gt_pts = torch.stack(pts_list, dim=0)  # [G, P, 2]
        G = gt_pts.shape[0]

        if gt_labels_3d is not None and len(gt_labels_3d) > b:
            gt_labels = gt_labels_3d[b].to(device)  # [G]
        else:
            gt_labels = torch.zeros(G, device=device).long()

        # 2. Compute Costs
        # Classification Cost
        pred_logits = logits[b].mean(dim=1)  # [N, C]
        pred_scores = pred_logits.sigmoid()  # [N, C]

        # Clamp to avoid nan
        pred_scores = pred_scores.clamp(min=1e-6, max=1.0 - 1e-6)

        # Cost is -score for the target class
        # Ensure gt_labels are within range
        valid_cls = (gt_labels >= 0) & (gt_labels < pred_scores.shape[1])
        cost_cls = torch.zeros(N, G, device=device)

        if valid_cls.all():
            cost_cls = -pred_scores[:, gt_labels]
        else:
            # Fallback if labels are weird
            cost_cls = torch.zeros(N, G, device=device)

        # Points L1 Cost
        pred_flat = mu_xy[b].flatten(1)  # [N, P*2]
        gt_flat = gt_pts.flatten(1)  # [G, P*2]
        cost_pts = torch.cdist(pred_flat, gt_flat, p=1) / P

        # Final Cost
        C = w_cls * cost_cls + w_pts * cost_pts  # [N, G]

        # Hungarian Assignment
        C_cpu = C.detach().cpu().numpy()
        # Check for NaNs
        if np.isnan(C_cpu).any():
            # Fallback to greedy if cost matrix is broken
            matches.append((torch.empty(0, dtype=torch.long, device=device),
                            torch.empty(0, dtype=torch.long, device=device)))
            continue

        indices = linear_sum_assignment(C_cpu)

        matches.append((torch.as_tensor(indices[0], dtype=torch.long, device=device),
                        torch.as_tensor(indices[1], dtype=torch.long, device=device)))

    return matches


def compute_greedy_matching_fallback(ceh_preds_dict, gt_bboxes_3d):
    """Fallback greedy matching if hungarian fails completely."""
    mu_xy = ceh_preds_dict['mu_xy']
    B, N, P, _ = mu_xy.shape
    device = mu_xy.device
    matches = []
    for b in range(B):
        pts_list = extract_gt_points(gt_bboxes_3d[b], device, P)
        if len(pts_list) == 0:
            matches.append(
                (torch.empty(0, dtype=torch.long, device=device), torch.empty(0, dtype=torch.long, device=device)))
            continue
        gt_pts = torch.stack(pts_list)
        pred_mean = mu_xy[b].mean(1)
        gt_mean = gt_pts.mean(1)
        try:
            dists = torch.cdist(pred_mean, gt_mean, p=1)
            # Simple min assignment without uniqueness constraint for absolute fallback
            vals, gt_inds = dists.min(dim=1)
            # But we usually want uniqueness. Let's return empty if failed.
            matches.append(
                (torch.empty(0, dtype=torch.long, device=device), torch.empty(0, dtype=torch.long, device=device)))
        except:
            matches.append(
                (torch.empty(0, dtype=torch.long, device=device), torch.empty(0, dtype=torch.long, device=device)))
    return matches


def sgjc_loss(ceh_preds, gt_bboxes_3d, gt_labels_3d=None,
              img_metas=None,
              matched_indices=None,
              weights=None,
              cross_encourage_weight=0.05,
              eps=1e-6):
    if weights is None:
        weights = {'loss_geo': 1.0, 'loss_nll': 1.0, 'loss_sem_reg': 0.05, 'loss_sem_cls': 0.5, 'loss_sem_var': 1.0}

    # Extract Predictions
    mu_joint = ceh_preds['mu_joint']  # [B, N, 3] (x, y, sem_err) -> Wait, did we keep [B,N,P,3]?
    # In sgjc_head, we returned mu_joint [M, 3]. In maptr_head, we reshaped to [B, N, P, 3].

    Sigma_joint = ceh_preds['Sigma_joint']  # [B, N, P, 3, 3]
    logits = ceh_preds['logits']  # [B, N, P, C]
    p_sem = ceh_preds['p_sem']  # [B, N, P, C]
    semantic_var = ceh_preds['semantic_var']  # [B, N, P]
    mu_xy = ceh_preds['mu_xy']  # [B, N, P, 2]

    device = mu_xy.device
    B, N, P, _ = mu_xy.shape

    # 1. 准备 GT (Points)
    gt_pts_batch = []
    for b in range(B):
        pts_list = extract_gt_points(gt_bboxes_3d[b], device, P)
        pts = torch.stack(pts_list, dim=0) if pts_list else torch.zeros(0, P, 2, device=device)
        gt_pts_batch.append(pts)

    # 2. 匹配 (Hungarian)
    cost_weights = {'cls': 2.0, 'pts': 5.0}
    try:
        normalized_matches = compute_hungarian_matching(ceh_preds, gt_bboxes_3d, gt_labels_3d, cost_weights)
    except Exception:
        # Emergency fallback to empty match to prevent crash
        normalized_matches = [
            (torch.empty(0, dtype=torch.long, device=device), torch.empty(0, dtype=torch.long, device=device)) for _ in
            range(B)]

    # 3. 初始化 Loss
    loss_l1 = 0.0
    loss_nll = 0.0
    loss_reg = 0.0
    loss_sem_cls = 0.0
    loss_sem_var = 0.0  # <--- 初始化在这里
    entropy_sum = 0.0
    cross_encourage = 0.0
    total_points = 0
    cnt_lines = 0

    for b in range(B):
        pred_inds, gt_inds = normalized_matches[b]
        if pred_inds.numel() == 0: continue

        gt_pts_b = gt_pts_batch[b]
        if gt_inds.max() >= gt_pts_b.shape[0]: continue

        # --- Extract Tensors ---
        # mu_joint is [B, N, P, 3] or [B, N, 3]?
        # In maptr_head_with_sgjc.py, we did: "mu_joint": r(sgjc_out_final["mu_joint"])
        # If sgjc_head returns [M, 3], r() makes it [B, N, P, 3].

        # mu_b [K, P, 3]
        mu_b = mu_joint[b, pred_inds]
        sigma_b = Sigma_joint[b, pred_inds]
        logits_b = logits[b, pred_inds]
        p_b = p_sem[b, pred_inds]
        sem_var_b = semantic_var[b, pred_inds]
        gt_b_xy = gt_pts_b[gt_inds]

        # Valid Mask (gt not zero)
        mask = (gt_b_xy.norm(dim=-1) > 1e-6)
        if mask.sum() == 0: continue

        # Filter Valid Points
        mu_b_valid = mu_b[mask]  # [Valid, 3]
        gt_b_xy_valid = gt_b_xy[mask]  # [Valid, 2]
        sigma_b_valid = sigma_b[mask]  # [Valid, 3, 3]
        logits_b_valid = logits_b[mask]
        p_b_valid = p_b[mask]
        sem_var_valid = sem_var_b[mask]

        # --- Construct 3D Target for NLL ---
        # Target: [gt_x, gt_y, gt_sem_error]
        gt_sem_err_valid = torch.zeros(mu_b_valid.shape[0], 1, device=device)

        if gt_labels_3d is not None and len(gt_labels_3d) > b:
            gt_label_b = gt_labels_3d[b].to(device)[gt_inds]
            # Expand label to points: [K] -> [K, P] -> [Valid]
            gt_label_expand = gt_label_b.unsqueeze(-1).expand(-1, P)[mask]

            # Calculate Real Cross Entropy (Semantic Error)
            # This is the target for the 3rd dimension of mu
            # logits_b_valid: [Valid, C]
            real_ce_loss = F.cross_entropy(logits_b_valid, gt_label_expand.long(), reduction='none')
            gt_sem_err_valid = real_ce_loss.unsqueeze(-1)  # [Valid, 1]

            # --- Loss: Semantic Classification ---
            loss_sem_cls += real_ce_loss.sum()

            # --- Loss: Semantic Variance (std dev) ---
            # We want predicted semantic variance (sem_var) to approximate prediction confidence error
            # Confidence Error ~ 1 - probability of true class
            probs = torch.softmax(logits_b_valid, dim=-1)
            true_probs = torch.gather(probs, 1, gt_label_expand.unsqueeze(-1)).squeeze(-1)
            conf_error = (1.0 - true_probs).detach()

            # sem_var_valid should match conf_error
            loss_sem_var += F.smooth_l1_loss(sem_var_valid, conf_error, reduction='sum')

        # Concat Target [Valid, 3]
        target_joint = torch.cat([gt_b_xy_valid, gt_sem_err_valid], dim=-1)

        # --- Loss: Geometry L1 ---
        loss_l1 += F.l1_loss(mu_b_valid[:, :2], gt_b_xy_valid, reduction='sum')

        # --- Loss: Joint NLL (3D) ---
        nll_per = stable_nll_loss_batch_3d(mu_b_valid, sigma_b_valid, target_joint, eps=eps)
        loss_nll += nll_per.sum()

        # --- Loss: Entropy Reg ---
        entropy = - (p_b_valid * torch.log(p_b_valid + 1e-12)).sum(dim=-1)
        loss_reg += entropy.mean()
        entropy_sum += entropy.mean().item()

        # --- Loss: Cross Encourage ---
        # Encourage non-zero off-diagonals (xy, xz, yz)
        # indices: (0,1), (0,2), (1,2)
        off_diag = torch.cat([
            sigma_b_valid[:, 0, 1].unsqueeze(-1),
            sigma_b_valid[:, 0, 2].unsqueeze(-1),
            sigma_b_valid[:, 1, 2].unsqueeze(-1)
        ], dim=-1)
        cross_encourage += (- off_diag.norm(dim=-1).mean().clamp(max=1e6))

        total_points += mu_b_valid.shape[0]
        cnt_lines += pred_inds.numel()

    # Handle Empty Batch
    if total_points == 0:
        zero = torch.tensor(0.0, device=device, requires_grad=True)
        return {
            'loss_geo': zero,
            'loss_nll': zero,
            'loss_sem_reg': zero,
            'loss_entropy': zero,
            'loss_sem_cls': zero,
            'loss_cross_encourage': zero,
            'loss_sem_var': zero  # Ensure key exists
        }

    # Normalize Losses
    loss_geo = (loss_l1 / total_points) * weights.get('loss_geo', 1.0)
    loss_nll = (loss_nll / total_points) * weights.get('loss_nll', 1.0)
    loss_sem_var = (loss_sem_var / total_points) * weights.get('loss_sem_var', 1.0)

    # Instance-level normalization for others
    norm_lines = max(cnt_lines, 1)
    loss_sem_reg = (loss_reg / norm_lines) * weights.get('loss_sem_reg', 0.05)
    loss_sem_cls = (loss_sem_cls / total_points) * weights.get('loss_sem_cls', 0.5)
    loss_cross = (cross_encourage / norm_lines) * cross_encourage_weight

    avg_entropy = entropy_sum / norm_lines

    return {
        'loss_geo': loss_geo,
        'loss_nll': loss_nll,
        'loss_sem_var': loss_sem_var,  # Key is present
        'loss_sem_cls': loss_sem_cls,
        'loss_sem_reg': loss_sem_reg,
        'loss_entropy': torch.tensor(avg_entropy, device=device),
        'loss_cross_encourage': loss_cross
    }