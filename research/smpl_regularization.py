"""OmniRe's SMPLNodes regularizers for R3 (DriveStudio configs/omnire.yaml, MIT), plus a box tether.

* trans_temporal_smooth: second difference of the pelvis over a random interval of 1-5 poses.
* joint_smooth: second difference of posed joint positions over consecutive poses.
* knn_reg: appearance and shape agree with the 3 nearest canonical neighbours.
* max_s_square / sharp_shape: no large or needle-like Gaussians, which bleed into the road under the feet.
* local_offset (R3 only): the root stays close to where it started inside its learned box.

Both smoothing terms act on scene-frame positions. For a track anchored to its box they cover
the box and the local pose together, so jitter cannot come back through the box trajectory.
"""
import random

import torch


class SMPLRegularizer:
    """Holds the k-NN cache of one SMPLTrack. Not a module, so checkpoints keep the R2 layout."""

    def __init__(self, person, neighbors=3, refresh_every=100):
        self.neighbors = neighbors
        self.refresh_every = refresh_every
        self.knn = None
        self.knn_step = None

    def neighbours(self, person, step):
        if self.knn is None or step - self.knn_step >= self.refresh_every:
            with torch.no_grad():
                means = person.means.detach()
                distances = torch.cdist(means, means)
                self.knn = distances.topk(self.neighbors + 1, largest=False).indices[:, 1:]
            self.knn_step = step
        return self.knn

    @staticmethod
    def pose_index(person, time):
        """Nearest valid pose to the training time, or None between separated segments."""
        index = (person.times - time.reshape(())).abs().argmin()
        if (person.times[index] - time.reshape(())).abs() > 0.055:
            return None
        return int(index)

    @staticmethod
    def consecutive(person, lo, hi):
        if lo < 0 or hi >= len(person.times):
            return False
        span = hi - lo
        return (int(person.frame_ids[hi] - person.frame_ids[lo]) == span
                and float(person.times[hi] - person.times[lo]) <= 0.12 * span)

    def losses(self, person, time, step, weights):
        out = {}
        index = self.pose_index(person, time)
        if index is None:
            return out
        device = person.times.device
        if weights["trans_smooth"]:
            interval = random.randint(1, 5)
            lo, hi = index - interval, index + interval
            if self.consecutive(person, lo, hi):
                pelvis = person.joints_at(torch.tensor([lo, index, hi], device=device))[:, 0]
                # OmniRe detaches the neighbours, so only the middle pose is pulled.
                out["trans_smooth"] = (pelvis[2].detach() + pelvis[0].detach() - 2 * pelvis[1]).abs().mean()
        if weights["joint_smooth"] and self.consecutive(person, index - 1, index + 1):
            joints = person.joints_at(torch.arange(index - 1, index + 2, device=device))
            out["joint_smooth"] = ((joints[2] - joints[1]) - (joints[1] - joints[0])).abs().mean()
        if weights.get("local_offset") and getattr(person, "tether_translation", None) is not None:
            # Root position and box position explain the same pixels. Keep the box doing the travel.
            out["local_offset"] = (person.translation - person.tether_translation).norm(dim=-1).mean()
        if weights["knn"]:
            knn = self.neighbours(person, step)
            attributes = (person.quats, person.scales.exp(), person.opacities.sigmoid(),
                          person.features_dc, person.features_rest)
            out["knn"] = sum(value[knn].std(dim=1).mean() for value in attributes)
        scales = person.scales.exp()
        if weights["max_scale"]:
            out["max_scale"] = scales.amax(-1).square().mean()
        if weights["sharp_shape"] and step % 10 == 0:
            ratio = scales.amax(-1) / scales.amin(-1)
            out["sharp_shape"] = (ratio.clamp_min(10.0) - 10.0).mean()
        return out
