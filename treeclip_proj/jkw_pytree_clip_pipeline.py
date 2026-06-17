#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orchard tree clipping — Canopy-first v5
- Chunked classification (memory-tamed)
- FIX: boolean mask assignment in chunked row assignment
- Bounded connectivity pruning
- Full-resolution output by default (--cc-map-back now ON by default when --cc-ds>0)
- NEW: --output-voxel for optional lightweight output downsampling (default 0 = disabled)
- NEW: --clean-artefacts for post-cluster tripod/sphere/ground removal

OUTPUT DENSITY BEHAVIOUR:
  Downsampling is used ONLY for intermediate analysis steps (ground plane fitting,
  canopy row/tree detection, connectivity pruning). The saved per-tree .ply files
  are always at the ORIGINAL input point density unless you explicitly opt in to
  output downsampling with --output-voxel.

When --cc-prune is enabled and --cc-ds>0:
  * --cc-map-back is TRUE by default: DBSCAN runs on the downsampled copy to decide
    which components to keep, then those decisions are mapped back to the ORIGINAL
    full-resolution points via KDTree. The mapping radius defaults to 0.9*cc_ds.
  * Pass --no-cc-map-back to revert to the old behaviour (output at cc-ds voxel
    resolution) — useful if RAM is very tight.

Optional output downsampling (disabled by default):
  * --output-voxel 0.01   apply a final voxel downsample to each saved tree (e.g. 1 cm)
  * --output-voxel 0      no output downsampling (default)
"""
import os
import argparse
import logging
import numpy as np
import open3d as o3d
from sklearn.cluster import KMeans

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# ---------------- IO ----------------

def _sniff_xyz_like_format(path, max_lines=4000):
    cols = None
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for _ in range(max_lines):
            s = f.readline()
            if not s:
                break
            s = s.strip()
            if not s or s.startswith('#'):
                continue
            s = s.replace(',', ' ')
            parts = [p for p in s.split() if p]
            if len(parts) >= 3:
                nnum = 0
                for p in parts:
                    try:
                        float(p); nnum += 1
                    except Exception:
                        pass
                if nnum >= 3:
                    cols = nnum
                    break
    return cols

def _load_xyz_or_txt(path):
    pts = []
    cols = _sniff_xyz_like_format(path)
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            s = s.replace(',', ' ')
            parts = [p for p in s.split() if p]
            if len(parts) < 3:
                continue
            try:
                x, y, z = map(float, parts[:3])
            except Exception:
                continue
            if cols and cols >= 6:
                try:
                    r, g, b = map(float, parts[3:6])
                    pts.append((x, y, z, r, g, b))
                except Exception:
                    pts.append((x, y, z))
            else:
                pts.append((x, y, z))
    A = np.asarray(pts, dtype=float)
    if A.shape[1] >= 6:
        P = A[:, :3]
        C = A[:, 3:6]
        if C.max() > 1.0:
            C = C / 255.0
    else:
        P = A[:, :3]
        C = None
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(P)
    if C is not None:
        pcd.colors = o3d.utility.Vector3dVector(C)
    return pcd

def read_point_cloud_any(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.xyz', '.txt'):
        return _load_xyz_or_txt(path)
    pcd = o3d.io.read_point_cloud(path)
    if pcd is None or len(pcd.points) == 0:
        raise RuntimeError('Failed to read point cloud or empty: %s' % path)
    return pcd

# ---------------- Ground plane ----------------

def fit_ground_plane(pcd, voxel=0.08, dist_thresh=0.12, ransac_n=3, iters=2000):
    ds = pcd.voxel_down_sample(voxel)
    if len(ds.points) == 0:
        raise RuntimeError('Empty after downsample')
    model, _ = ds.segment_plane(distance_threshold=dist_thresh, ransac_n=ransac_n, num_iterations=iters)
    a, b, c, d = [float(x) for x in model]
    n = np.array([a, b, c], dtype=float)
    n /= (np.linalg.norm(n) + 1e-12)
    return n[0], n[1], n[2], d

def orient_plane_up(plane):
    a, b, c, d = plane
    n = np.array([a, b, c], dtype=float)
    return (a, b, c, d) if n[2] >= 0 else (-a, -b, -c, -d)

def plane_signed_distance(P, plane):
    a, b, c, d = plane
    return P[:,0]*a + P[:,1]*b + P[:,2]*c + d

def adjust_plane_bias(pcd, plane, q=2.0):
    P = np.asarray(pcd.points)
    H = plane_signed_distance(P, plane)
    bias = float(np.percentile(H, q))
    a,b,c,d = plane
    return (a,b,c,d - bias), bias

# ---------------- Row axis from canopy ----------------

def estimate_row_axis_from_canopy(P_canopy):
    XY = P_canopy[:, :2]
    mu = XY.mean(axis=0)
    Xc = XY - mu
    _, _, VT = np.linalg.svd(Xc, full_matrices=False)
    u = VT[0]
    u = u / (np.linalg.norm(u) + 1e-12)
    v = np.array([-u[1], u[0]], dtype=float)
    return mu.astype(np.float32), u.astype(np.float32), v.astype(np.float32)

# ---------------- Canopy selection ----------------

def select_canopy_points(P, H, min_h=None, q_top=None):
    if min_h is not None and min_h > 0:
        mask = H >= float(min_h)
    elif q_top is not None and 0.0 < q_top < 1.0:
        thr = np.quantile(H, q_top)
        mask = H >= thr
    else:
        thr = np.quantile(H, 0.65)
        mask = H >= thr
    idx = np.where(mask)[0]
    return idx

# ---------------- Canopy clustering per row ----------------

def cluster_canopy_per_row(P_canopy, mu, u, v, rows=1, expected_per_row=None,
                           q_s=0.98, m_s=0.20, q_t=0.98, m_t=0.60):
    XY = P_canopy[:, :2]
    t = (XY - mu) @ u
    s = (XY - mu) @ v

    if rows > 1:
        km_rows = KMeans(n_clusters=rows, n_init=10, random_state=42)
        row_raw = km_rows.fit_predict(s.reshape(-1, 1))
        row_median_s = np.array([
            np.median(s[row_raw == r]) if np.any(row_raw == r) else np.nan
            for r in range(rows)
        ], dtype=float)
        order = np.argsort(row_median_s)
        remap = {old: new for new, old in enumerate(order)}
        row_ids = np.vectorize(lambda x: remap.get(x, rows-1))(row_raw)
        row_centers_s = row_median_s[order]
    else:
        row_ids = np.zeros_like(s, dtype=int)
        row_centers_s = np.array([np.median(s)], dtype=float)

    if rows == 1:
        row_bounds_s = np.array([row_centers_s[0] - 10.0, row_centers_s[0] + 10.0], dtype=float)
    else:
        mids = (row_centers_s[1:] + row_centers_s[:-1]) * 0.5
        left  = row_centers_s[0]  - (mids[0] - row_centers_s[0])
        right = row_centers_s[-1] + (row_centers_s[-1] - mids[-1])
        row_bounds_s = np.concatenate([[left], mids, [right]])

    per_row, centers_t, t_bounds, s_limits = [], [], [], []
    t_centers, s_centers = [], []
    shadow_rects = []

    for r in range(rows):
        mask_r = (row_ids == r)
        if not np.any(mask_r):
            per_row.append(0)
            centers_t.append([])
            t_bounds.append(np.zeros(0))
            s_limits.append([])
            t_centers.append([])
            s_centers.append([])
            continue

        t_r = t[mask_r]
        s_r = s[mask_r]

        if expected_per_row and len(expected_per_row) > r and expected_per_row[r] > 0:
            k = int(expected_per_row[r])
        else:
            length = float(t_r.max() - t_r.min()) if t_r.size else 0.0
            k = max(1, int(round(length / 3.0)))

        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        km.fit(t_r.reshape(-1, 1))
        c = np.sort(km.cluster_centers_.ravel())

        per_row.append(k)
        centers_t.append(c)

        if k == 1:
            half = 1.0
            bounds = np.array([c[0] - half, c[0] + half], dtype=float)
        else:
            mids = (c[1:] + c[:-1]) * 0.5
            left = c[0] - (c[1] - c[0]) * 0.5
            right = c[-1] + (c[-1] - c[-2]) * 0.5
            bounds = np.concatenate([[left], mids, [right]])
        t_bounds.append(bounds)

        bin_idx = np.searchsorted(bounds, t_r, side='right') - 1

        s_half_list, t_center_list, s_center_list = [], [], []
        for i in range(k):
            mask_i = (bin_idx == i)
            t_i = t_r[mask_i]
            s_i = s_r[mask_i]

            if t_i.size == 0:
                t_center = c[i]
                s_center = float(np.median(s_r)) if s_r.size else 0.0
                t_half = 0.8
                s_half = 0.6
            else:
                t_center = float(np.median(t_i))
                s_center = float(np.median(s_i))
                t_half = float(np.quantile(np.abs(t_i - t_center), q_t)) + m_t
                s_half = float(np.quantile(np.abs(s_i - s_center), q_s)) + m_s
                t_half = max(t_half, 0.6)
                s_half = max(s_half, 0.5)

            s_half_list.append(s_half)
            t_center_list.append(t_center)
            s_center_list.append(s_center)
            shadow_rects.append((r, t_center, s_center, t_half, s_half))

        s_limits.append(s_half_list)
        t_centers.append(t_center_list)
        s_centers.append(s_center_list)

    return per_row, centers_t, t_bounds, s_limits, shadow_rects, row_centers_s, row_bounds_s, t_centers, s_centers

# ---------------- Local ground plane helper ----------------

def local_ground_cut(sub_pcd, clean_local_floor=0.12, voxel_ds=0.06, lowest_frac=0.15,
                     global_plane=None):
    P = np.asarray(sub_pcd.points)
    if P.shape[0] == 0:
        return np.zeros(0, dtype=bool)

    ds = sub_pcd.voxel_down_sample(voxel_ds) if voxel_ds and voxel_ds > 0 else sub_pcd
    Pd = np.asarray(ds.points)
    if Pd.shape[0] < 50:
        return np.ones(P.shape[0], dtype=bool)

    if global_plane is not None:
        a,b,c,d = global_plane
        Hglob_ds = Pd[:,0]*a + Pd[:,1]*b + Pd[:,2]*c + d
        thr = np.percentile(Hglob_ds, lowest_frac*100.0)
        low_mask = (Hglob_ds <= thr)
    else:
        z_thr = np.percentile(Pd[:,2], lowest_frac*100.0)
        low_mask = (Pd[:,2] <= z_thr)
    if not np.any(low_mask):
        low_mask = np.ones(Pd.shape[0], dtype=bool)

    ds_low = o3d.geometry.PointCloud()
    ds_low.points = o3d.utility.Vector3dVector(Pd[low_mask])

    try:
        model, _ = ds_low.segment_plane(distance_threshold=0.10, ransac_n=3, num_iterations=1000)
        a,b,c,d = [float(x) for x in model]
        if c < 0: a,b,c,d = -a,-b,-c,-d
        Hloc = P[:,0]*a + P[:,1]*b + P[:,2]*c + d
        keep = Hloc > float(clean_local_floor)
        return keep
    except Exception:
        return np.ones(P.shape[0], dtype=bool)

# ---------------- Connectivity pruning (downsampled space) ----------------

def _prune_components_on_pcd(work_pcd, radius=0.25, min_size=300, trunk_center=None, trunk_touch_radius=0.35):
    """Run DBSCAN on work_pcd and return a boolean mask (len=work_pcd) of points to keep.
    Keeps largest component; also keeps any component touching trunk_center within trunk_touch_radius,
    and any component with size >= min_size.
    """
    if len(work_pcd.points) == 0:
        return np.zeros(0, dtype=bool)
    labels = np.array(work_pcd.cluster_dbscan(eps=radius, min_points=20, print_progress=False))
    if labels.size == 0:
        return np.ones(len(work_pcd.points), dtype=bool)  # nothing clustered -> keep all
    uniq, counts = np.unique(labels[labels >= 0], return_counts=True)
    if uniq.size == 0:
        return np.ones(len(work_pcd.points), dtype=bool)
    main_label = int(uniq[np.argmax(counts)])
    keep = (labels == main_label)
    if trunk_center is not None:
        P = np.asarray(work_pcd.points)
        d2 = np.sum((P - np.asarray(trunk_center))**2, axis=1)
        near_trunk = d2 <= (trunk_touch_radius**2)
        comp_labels_near = np.unique(labels[near_trunk & (labels >= 0)])
        for lb in comp_labels_near:
            keep |= (labels == lb)
    for lb, sz in zip(uniq, counts):
        if sz >= min_size:
            keep |= (labels == lb)
    return keep

# ---------------- Post-cluster artefact removal ----------------

def remove_artefacts(pcd, 
                     stat_nb=20, stat_std=2.0,
                     min_component_pts=0,
                     cc_radius=0.26,
                     height_sphere_max=None,
                     height_sphere_min=None):
    """Remove residual ground patches, tripods, and registration spheres from a
    per-tree point cloud after segmentation.

    Steps applied in order:
      1. Statistical outlier removal  -- eliminates isolated sphere surface pts.
      2. Small-component pruning      -- drops any DBSCAN component smaller than
                                        min_component_pts (catches tripod legs etc.)
      3. Optional height-band filter  -- removes points in a height band that is
                                        characteristic of sphere artefacts (e.g.
                                        0.8–1.4 m) that are spatially separated
                                        from the tree trunk centroid.

    Returns a boolean keep-mask aligned to the INPUT pcd points.
    """
    if len(pcd.points) == 0:
        return np.ones(0, dtype=bool)

    N = len(pcd.points)
    overall_keep = np.ones(N, dtype=bool)

    # --- Step 1: statistical outlier removal ---
    if stat_nb > 0 and stat_std > 0:
        _, ind = pcd.remove_statistical_outlier(nb_neighbors=int(stat_nb),
                                                std_ratio=float(stat_std))
        mask_stat = np.zeros(N, dtype=bool)
        mask_stat[np.asarray(ind)] = True
        overall_keep &= mask_stat
        logging.debug('  artefact removal: statistical outlier removed %d pts', N - mask_stat.sum())

    # --- Step 2: small-component pruning ---
    if min_component_pts > 0 and overall_keep.sum() > 0:
        sub_idx = np.where(overall_keep)[0]
        sub = pcd.select_by_index(sub_idx)
        lbl = np.array(sub.cluster_dbscan(eps=float(cc_radius),
                                           min_points=5,
                                           print_progress=False))
        keep_comp = np.zeros(len(sub_idx), dtype=bool)
        if lbl.size > 0:
            uniq, counts = np.unique(lbl[lbl >= 0], return_counts=True)
            for lb, sz in zip(uniq, counts):
                if sz >= int(min_component_pts):
                    keep_comp |= (lbl == lb)
        overall_keep[sub_idx] = keep_comp
        logging.debug('  artefact removal: small-component pass removed %d pts',
                      sub_idx.size - keep_comp.sum())

    return overall_keep

# ---------------- Helpers ----------------

def ts_to_xy(mu, u, v, t_val, s_val):
    return mu + t_val*u + s_val*v

# ---------------- Chunked classification ----------------

def classify_by_prisms_chunked(P, mu, u, v,
                               rows, per_row, t_bounds, s_limits,
                               row_centers_s, t_centers, s_centers,
                               chunk_size_points=5_000_000):
    N = P.shape[0]
    K = int(sum(per_row))
    label_dtype = np.int16 if K < 32768 else np.int32
    labels = np.full(N, -1, dtype=label_dtype)
    cluster_parts = [ [] for _ in range(K) ]

    per_row_arr = np.asarray(per_row, dtype=int)
    row_offsets = np.concatenate([[0], np.cumsum(per_row_arr[:-1])]).astype(int)

    mu32 = mu.astype(np.float32)
    u32 = u.astype(np.float32)
    v32 = v.astype(np.float32)

    for start in range(0, N, chunk_size_points):
        end = min(N, start + chunk_size_points)
        XY = P[start:end, :2].astype(np.float32, copy=False)
        t = (XY - mu32) @ u32
        s = (XY - mu32) @ v32

        # Row assignment without NxR broadcast (HOTFIX applied)
        best_r = np.zeros(end-start, dtype=np.int32)
        min_d = np.full(end-start, np.inf, dtype=np.float32)
        for r, s_center in enumerate(row_centers_s):
            d = np.abs(s - float(s_center))
            better = d < min_d
            best_r[better] = r
            min_d[better] = d[better]

        lbl_chunk = np.full(end-start, -1, dtype=label_dtype)
        for r in range(rows):
            mask_r = (best_r == r)
            if not np.any(mask_r):
                continue
            bounds = np.asarray(t_bounds[r], dtype=np.float32)
            if bounds.size < 2:
                continue
            t_r = t[mask_r]
            s_r = s[mask_r]
            idx_local = np.searchsorted(bounds, t_r, side='right') - 1
            k = int(per_row_arr[r])
            valid = (idx_local >= 0) & (idx_local < k)
            if not np.any(valid):
                continue
            s_half_arr = np.asarray(s_limits[r], dtype=np.float32)
            s_center_arr = np.asarray(s_centers[r], dtype=np.float32)
            idx_clip = np.clip(idx_local, 0, k-1)
            valid &= (np.abs(s_r - s_center_arr[idx_clip]) <= s_half_arr[idx_clip])
            loc = np.where(mask_r)[0]
            loc_v = loc[valid]
            if loc_v.size:
                gl = (row_offsets[r] + idx_local[valid]).astype(label_dtype)
                lbl_chunk[loc_v] = gl
        labels[start:end] = lbl_chunk

        ulabels = np.unique(lbl_chunk)
        for g in ulabels:
            if g < 0:
                continue
            part_idx = np.flatnonzero(lbl_chunk == g) + start
            cluster_parts[int(g)].append(part_idx)
        logging.info('Classify chunk %d:%d / %d done', start, end, N)

    cluster_indices = [ (np.concatenate(parts) if parts else np.empty(0, dtype=np.int64))
                        for parts in cluster_parts ]
    return labels, cluster_indices

# ---------------- Save with cleanup ----------------

def save_clusters_with_cleanup(input_path, pcd, plane, labels, K, save_ground_dist,
                               mu, u, v, shadow_rects,
                               cluster_indices,
                               clean_ground=False,
                               clean_local_floor=0.12,
                               clean_hull_margin_t=0.50,
                               clean_hull_margin_s=0.12,
                               clean_shadow_floor=0.20,
                               cc_prune=False, cc_radius=0.25, cc_min_size=300,
                               cc_trunk_touch=False, cc_trunk_radius=0.35,
                               cc_ds=0.0, cc_max_points=None,
                               cc_map_back=False, cc_map_back_radius=0.0,
                               cc_map_back_chunk=2_000_000,
                               clean_artefacts=False,
                               artefact_stat_nb=20, artefact_stat_std=2.0,
                               artefact_min_component=500,
                               artefact_cc_radius=0.26,
                               output_voxel=0.0):
    base_dir = os.path.dirname(os.path.abspath(input_path))
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    out_dir = os.path.join(base_dir, f"{base_name}_canopy_clusters")
    os.makedirs(out_dir, exist_ok=True)

    P = np.asarray(pcd.points)
    C = np.asarray(pcd.colors) if pcd.has_colors() else None
    XY = P[:, :2].astype(np.float32, copy=False)
    mu32 = mu.astype(np.float32); u32 = u.astype(np.float32); v32 = v.astype(np.float32)

    t_all = (XY - mu32) @ u32
    s_all = (XY - mu32) @ v32
    H_global = plane_signed_distance(P, plane).astype(np.float32)

    saved = 0
    rects = shadow_rects[:K]

    for t_idx in range(K):
        idx = cluster_indices[t_idx]
        if idx.size == 0:
            continue

        sub = o3d.geometry.PointCloud()
        sub.points = o3d.utility.Vector3dVector(P[idx])
        if C is not None:
            sub.colors = o3d.utility.Vector3dVector(C[idx])

        keep_mask = np.ones(idx.size, dtype=bool)

        if plane is not None and save_ground_dist and save_ground_dist > 0:
            Hg = H_global[idx]
            keep_mask &= (Hg > float(save_ground_dist))

        if clean_ground and np.any(keep_mask):
            sub_keep = local_ground_cut(
                sub.select_by_index(np.where(keep_mask)[0]),
                clean_local_floor=clean_local_floor,
                voxel_ds=0.06,
                lowest_frac=0.15,
                global_plane=plane
            )
            km = keep_mask.copy()
            km_idx = np.where(keep_mask)[0]
            km[km_idx] = sub_keep
            keep_mask = km

            if t_idx < len(rects):
                _, t_c, s_c, t_half0, s_half0 = rects[t_idx]
                t_half = float(t_half0 + clean_hull_margin_t*0.0)
                s_half = float(s_half0 + clean_hull_margin_s*0.0)
                ta = t_all[idx]
                sa = s_all[idx]
                Hg = H_global[idx]
                outside_shadow = (np.abs(ta - t_c) > t_half) | (np.abs(sa - s_c) > s_half)
                low_near_ground = Hg <= float(clean_shadow_floor)
                keep_mask &= ~(outside_shadow & low_near_ground)

        # Apply pruning (after ground/shadow) on remaining points
        if cc_prune and np.any(keep_mask):
            # Build a working subcloud on kept points
            kept_indices_local = np.where(keep_mask)[0]
            sub_kept = sub.select_by_index(kept_indices_local)

            N_sub = len(sub_kept.points)
            if cc_max_points is not None and N_sub > cc_max_points:
                # Skip pruning on overly large clusters
                pruned_local_mask = np.ones(N_sub, dtype=bool)
            else:
                # Prepare trunk center estimate
                trunk_center_xyz = None
                if t_idx < len(rects):
                    _, t_c, s_c, _, _ = rects[t_idx]
                    xy_center = ts_to_xy(mu32, u32, v32, t_c, s_c)
                    P_subk = np.asarray(sub_kept.points)
                    near_xy = np.hypot(P_subk[:,0]-xy_center[0], P_subk[:,1]-xy_center[1]) <= 0.40
                    if np.any(near_xy):
                        z_med = float(np.median(P_subk[near_xy,2]))
                    else:
                        z_med = float(np.median(P_subk[:,2]))
                    trunk_center_xyz = np.array([xy_center[0], xy_center[1], z_med], dtype=float)

                if cc_ds and cc_ds > 0:
                    # Downsample for pruning
                    work = sub_kept.voxel_down_sample(cc_ds)
                    if len(work.points) == 0:
                        pruned_local_mask = np.ones(N_sub, dtype=bool)
                    else:
                        keep_ds = _prune_components_on_pcd(work, radius=float(cc_radius), min_size=int(cc_min_size),
                                                           trunk_center=trunk_center_xyz if cc_trunk_touch else None,
                                                           trunk_touch_radius=float(cc_trunk_radius))
                        kept_ds = work.select_by_index(np.where(keep_ds)[0])

                        if cc_map_back:
                            # Map back kept_ds -> original sub_kept using KDTree (k=1), in chunks
                            r_map = float(cc_map_back_radius)
                            if r_map <= 0:
                                r_map = float(cc_ds) * 0.9
                            r2 = r_map * r_map
                            tree = o3d.geometry.KDTreeFlann(kept_ds)
                            P_subk = np.asarray(sub_kept.points)
                            pruned_local_mask = np.zeros(N_sub, dtype=bool)

                            # Chunked nearest neighbor queries
                            step = int(cc_map_back_chunk)
                            for s0 in range(0, N_sub, step):
                                s1 = min(N_sub, s0 + step)
                                for i in range(s0, s1):
                                    # k=1 nearest
                                    k, idxs, d2 = tree.search_knn_vector_3d(P_subk[i], 1)
                                    if k > 0 and d2[0] <= r2:
                                        pruned_local_mask[i] = True
                        else:
                            # Not mapping back: return the downsampled kept set by projecting to nearest original.
                            # Build KDTree on sub_kept and mark originals close to any kept_ds point
                            if len(kept_ds.points) == 0:
                                pruned_local_mask = np.zeros(N_sub, dtype=bool)
                            else:
                                tree_full = o3d.geometry.KDTreeFlann(sub_kept)
                                pruned_local_mask = np.zeros(N_sub, dtype=bool)
                                r_map = float(cc_ds) * 0.9
                                r2 = r_map * r_map
                                for p in np.asarray(kept_ds.points):
                                    k, idxs, d2 = tree_full.search_radius_vector_3d(p, r_map)
                                    if k > 0:
                                        pruned_local_mask[idxs] = True
                else:
                    # No pre-DS: run DBSCAN directly on original sub_kept
                    keep_work = _prune_components_on_pcd(sub_kept, radius=float(cc_radius), min_size=int(cc_min_size),
                                                         trunk_center=trunk_center_xyz if cc_trunk_touch else None,
                                                         trunk_touch_radius=float(cc_trunk_radius))
                    pruned_local_mask = keep_work

            # Merge back pruned_local_mask into keep_mask
            km = keep_mask.copy()
            km_idx = kept_indices_local
            km[km_idx] = pruned_local_mask
            keep_mask = km

        # Finalize selection
        if not np.any(keep_mask):
            continue
        sub = sub.select_by_index(np.where(keep_mask)[0])
        if len(sub.points) == 0:
            continue

        # --- Post-cluster artefact removal (tripods, spheres, ground patches) ---
        if clean_artefacts:
            art_keep = remove_artefacts(
                sub,
                stat_nb=int(artefact_stat_nb),
                stat_std=float(artefact_stat_std),
                min_component_pts=int(artefact_min_component),
                cc_radius=float(artefact_cc_radius),
            )
            n_before = len(sub.points)
            sub = sub.select_by_index(np.where(art_keep)[0])
            logging.info('  tree%d artefact removal: %d -> %d pts (removed %d)',
                         t_idx, n_before, len(sub.points), n_before - len(sub.points))
            if len(sub.points) == 0:
                logging.warning('  tree%d: all points removed by artefact filter — skipping', t_idx)
                continue

        # --- Optional output downsampling (disabled by default) ---
        if output_voxel and output_voxel > 0:
            n_before = len(sub.points)
            sub = sub.voxel_down_sample(float(output_voxel))
            logging.info('  tree%d output voxel DS %.4fm: %d -> %d pts',
                         t_idx, output_voxel, n_before, len(sub.points))

        out = os.path.join(out_dir, f"{base_name}_tree{t_idx}.ply")
        o3d.io.write_point_cloud(out, sub, write_ascii=False)
        saved += 1

    logging.info('Saved %d clusters to %s', saved, out_dir)

# ---------------- CLI ----------------

def parse_args():
    ap = argparse.ArgumentParser(description='Canopy-first tree clipping v3b-hotfix2 (chunked + map-back pruning)')
    ap.add_argument('input_path')

    # canopy selection
    ap.add_argument('--canopy-min-h', type=float, default=1.2)
    ap.add_argument('--canopy-quantile', type=float, default=-1.0)

    # rows & seeding
    ap.add_argument('--rows', type=int, default=1)
    ap.add_argument('--expected-per-row', type=str, default='')

    # classification footprint controls
    ap.add_argument('--classify-s-quantile', type=float, default=0.99)
    ap.add_argument('--classify-s-margin', type=float, default=0.28)
    ap.add_argument('--classify-t-quantile', type=float, default=0.98)
    ap.add_argument('--classify-t-margin', type=float, default=0.60)

    # chunking
    ap.add_argument('--classify-chunk-m', type=float, default=5.0,
                    help='Chunk size in millions of points for classification (e.g., 5 = 5M).')

    # saving
    ap.add_argument('--save-ground-dist', type=float, default=0.12)

    # cleanup options
    ap.add_argument('--clean-ground', action='store_true')
    ap.add_argument('--clean-local-floor', type=float, default=0.08)
    ap.add_argument('--clean-hull-margin-t', type=float, default=0.50)
    ap.add_argument('--clean-hull-margin-s', type=float, default=0.16)
    ap.add_argument('--clean-shadow-floor', type=float, default=0.20)

    # canopy analysis voxel
    ap.add_argument('--voxel-canopy', type=float, default=0.06)

    # connectivity pruning
    ap.add_argument('--cc-prune', action='store_true')
    ap.add_argument('--cc-radius', type=float, default=0.26)
    ap.add_argument('--cc-min-size', type=int, default=250)
    ap.add_argument('--cc-trunk-touch', action='store_true')
    ap.add_argument('--cc-trunk-radius', type=float, default=0.35)
    ap.add_argument('--cc-ds', type=float, default=0.02, help='Voxel DS used only for pruning stage.')
    ap.add_argument('--cc-max-points', type=int, default=2_000_000, help='Skip pruning if cluster has more points.')
    ap.add_argument('--cc-map-back', action='store_true',
                    help='[DEPRECATED — map-back is now ON by default] Kept for backwards compatibility.')
    ap.add_argument('--no-cc-map-back', action='store_true',
                    help='Disable map-back: output trees at cc-ds voxel resolution instead of full resolution. '
                         'Use when RAM is very tight.')
    ap.add_argument('--cc-map-back-radius', type=float, default=0.0, help='Override map-back radius (m). If <=0, uses 0.9*cc-ds.')
    ap.add_argument('--cc-map-back-chunk', type=int, default=2_000_000, help='Points per chunk for KDTree map-back.')

    # output downsampling (disabled by default — output is full resolution)
    ap.add_argument('--output-voxel', type=float, default=0.0,
                    help='Optional voxel size (m) for output downsampling of saved tree clusters. '
                         '0 = disabled (full resolution, default). Example: --output-voxel 0.01')

    # artefact removal (tripods, spheres, residual ground)
    ap.add_argument('--clean-artefacts', action='store_true',
                    help='Enable post-cluster artefact removal (tripods, registration spheres, ground patches).')
    ap.add_argument('--artefact-stat-nb', type=int, default=20,
                    help='Statistical outlier removal: number of neighbours. Set 0 to disable.')
    ap.add_argument('--artefact-stat-std', type=float, default=2.0,
                    help='Statistical outlier removal: std-ratio threshold.')
    ap.add_argument('--artefact-min-component', type=int, default=500,
                    help='Drop DBSCAN components with fewer points than this (removes tripods/spheres).')
    ap.add_argument('--artefact-cc-radius', type=float, default=0.26,
                    help='DBSCAN radius (m) used for the artefact small-component pass.')

    # presets
    ap.add_argument('--preset', type=str, default='')

    return ap.parse_args()

# ---------------- Main ----------------

def main():
    args = parse_args()

    # Apply preset overrides (orchard_loose remains forgiving; map-back is opt-in)
    if (args.preset or '').lower() == 'orchard_loose':
        args.classify_s_quantile = 0.99
        args.classify_s_margin   = 0.28
        args.classify_t_quantile = 0.98
        args.classify_t_margin   = 0.60
        args.clean_local_floor   = 0.08
        args.cc_prune            = True
        args.cc_radius           = 0.26
        args.cc_min_size         = 250
        args.cc_trunk_touch      = True
        args.cc_trunk_radius     = 0.35
        args.cc_ds               = 0.02
        args.cc_max_points       = 2_000_000
        if args.clean_hull_margin_s < 0.16:
            args.clean_hull_margin_s = 0.16
        # Artefact removal is enabled by default in orchard_loose
        args.clean_artefacts          = True
        args.artefact_stat_nb         = 20
        args.artefact_stat_std        = 2.0
        args.artefact_min_component   = 500
        args.artefact_cc_radius       = 0.26
        # map-back is on by default; output is full resolution unless --output-voxel is set

    pcd = read_point_cloud_any(args.input_path)
    P = np.asarray(pcd.points)
    logging.info('Total points: %d', P.shape[0])

    plane = fit_ground_plane(pcd)
    plane = orient_plane_up(plane)
    plane, bias = adjust_plane_bias(pcd, plane, q=2.0)
    H = plane_signed_distance(P, plane)
    logging.info('Ground plane bias=%.3f | H stats: min=%.3f med=%.3f max=%.3f',
                 bias, H.min(), float(np.median(H)), H.max())

    if args.voxel_canopy > 0:
        canopy_ds = pcd.voxel_down_sample(args.voxel_canopy)
        P_ds = np.asarray(canopy_ds.points)
        H_ds = plane_signed_distance(P_ds, plane)
    else:
        P_ds = P
        H_ds = H

    if args.canopy_quantile and args.canopy_quantile > 0:
        idx_c = select_canopy_points(P_ds, H_ds, min_h=None, q_top=float(args.canopy_quantile))
        logging.info('Canopy selection by quantile=%.2f -> %d pts', args.canopy_quantile, idx_c.size)
    else:
        idx_c = select_canopy_points(P_ds, H_ds, min_h=float(args.canopy_min_h), q_top=None)
        logging.info('Canopy selection by min_h=%.2f -> %d pts', args.canopy_min_h, idx_c.size)
    if idx_c.size == 0:
        raise RuntimeError('No points in canopy selection. Adjust --canopy-min-h or use --canopy-quantile 0.65')
    P_canopy = P_ds[idx_c]

    mu, u, v = estimate_row_axis_from_canopy(P_canopy)

    if args.expected_per_row:
        try:
            expected = [int(x.strip()) for x in args.expected_per_row.split(',') if x.strip()]
        except Exception:
            expected = []
    else:
        expected = []

    per_row, centers_t, t_bounds, s_limits, shadow_rects,     row_centers_s, row_bounds_s, t_centers, s_centers = cluster_canopy_per_row(
        P_canopy, mu, u, v, rows=int(args.rows), expected_per_row=expected,
        q_s=float(args.classify_s_quantile), m_s=float(args.classify_s_margin),
        q_t=float(args.classify_t_quantile), m_t=float(args.classify_t_margin)
    )

    total_trees = sum(per_row)
    logging.info('Rows=%d | trees per row=%s | total=%d', args.rows, per_row, total_trees)

    chunk_pts = max(1_000_000, int(args.classify_chunk_m * 1_000_000))
    logging.info('Classification chunk size: %d pts', chunk_pts)

    labels, cluster_indices = classify_by_prisms_chunked(
        P, mu, u, v,
        int(args.rows), per_row, t_bounds, s_limits,
        row_centers_s, t_centers, s_centers,
        chunk_size_points=chunk_pts
    )

    save_clusters_with_cleanup(
        args.input_path, pcd, plane, labels, K=total_trees, save_ground_dist=args.save_ground_dist,
        mu=mu, u=u, v=v, shadow_rects=shadow_rects,
        cluster_indices=cluster_indices,
        clean_ground=bool(args.clean_ground),
        clean_local_floor=float(args.clean_local_floor),
        clean_hull_margin_t=float(args.clean_hull_margin_t),
        clean_hull_margin_s=float(args.clean_hull_margin_s),
        clean_shadow_floor=float(args.clean_shadow_floor),
        cc_prune=bool(args.cc_prune), cc_radius=float(args.cc_radius), cc_min_size=int(args.cc_min_size),
        cc_trunk_touch=bool(args.cc_trunk_touch), cc_trunk_radius=float(args.cc_trunk_radius),
        cc_ds=float(args.cc_ds), cc_max_points=int(args.cc_max_points) if args.cc_max_points else None,
        cc_map_back=(not args.no_cc_map_back),   # map-back ON by default; --no-cc-map-back disables it
        cc_map_back_radius=float(args.cc_map_back_radius),
        cc_map_back_chunk=int(args.cc_map_back_chunk),
        clean_artefacts=bool(args.clean_artefacts),
        artefact_stat_nb=int(args.artefact_stat_nb),
        artefact_stat_std=float(args.artefact_stat_std),
        artefact_min_component=int(args.artefact_min_component),
        artefact_cc_radius=float(args.artefact_cc_radius),
        output_voxel=float(args.output_voxel),
    )

if __name__ == '__main__':
    main()
