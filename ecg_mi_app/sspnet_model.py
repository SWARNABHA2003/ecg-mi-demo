"""
SSPNet model (copied unchanged from the training notebook) plus ensemble loading and prediction.
"""
import torch
import torch.nn as nn


class ACB(nn.Module):
    """Asymmetric convolution block: square kxk conv + (kx1 | 1xk) conv, summed, then BN + ReLU.
    orient='v' -> vertical kernel (K x 1) for the alpha band (upper/lower asymmetry)
    orient='h' -> horizontal kernel (1 x K) for the delta band (left/right asymmetry)"""

    def __init__(self, cin, cout, orient, stride=1, k=3):
        super().__init__()
        self.square = nn.Conv2d(cin, cout, k, stride, padding=k // 2, bias=False)
        if orient == "v":
            self.asym = nn.Conv2d(cin, cout, (k, 1), stride, padding=(k // 2, 0), bias=False)
        elif orient == "h":
            self.asym = nn.Conv2d(cin, cout, (1, k), stride, padding=(0, k // 2), bias=False)
        else:
            raise ValueError(orient)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.square(x) + self.asym(x)))


class ChannelAttention(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(c, c // r), nn.ReLU(inplace=True), nn.Linear(c // r, c))

    def forward(self, x):
        a = self.mlp(x.mean((2, 3))) + self.mlp(x.amax((2, 3)))
        return x * torch.sigmoid(a)[:, :, None, None]


class SpatialAttention(nn.Module):
    def __init__(self, k=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, k, padding=k // 2)

    def forward(self, x):
        s = torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)
        return x * torch.sigmoid(self.conv(s))


class Branch(nn.Module):
    def __init__(self, orient, in_ch=12):
        super().__init__()
        self.b1 = ACB(in_ch, 64, orient, stride=2)    # 64 x 64 x 64
        self.b2 = ACB(64, 128, orient)                # 128 x 64 x 64
        self.ca = ChannelAttention(128)
        self.b3 = ACB(128, 128, orient, stride=2)     # 128 x 32 x 32
        self.b4 = ACB(128, 256, orient)               # 256 x 32 x 32
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(self.b2(self.b1(x)))
        return self.sa(self.b4(self.b3(x)))


class CrossAttention(nn.Module):
    """Tokens = spatial positions of one branch (query) attending to the other branch (key/value)."""

    def __init__(self, dim=256, heads=4, n_tokens=32 * 32):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, n_tokens, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, q, kv):
        q, kv = q + self.pos, kv + self.pos
        out, _ = self.attn(q, kv, kv, need_weights=False)
        return self.norm(q + out)


class Bottleneck(nn.Module):
    def __init__(self, c=256, mid=64):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(c, mid, 1, bias=False), nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, 3, padding=1, bias=False), nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
            nn.Conv2d(mid, c, 1, bias=False), nn.BatchNorm2d(c),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.f(x))


class SSPNet(nn.Module):
    def __init__(self, num_classes=3, dim=256, heads=4, mha_heads=8, dropout=0.3, token_pool=1):
        super().__init__()
        assert 32 % token_pool == 0
        self.pool = nn.AvgPool2d(token_pool) if token_pool > 1 else nn.Identity()
        n_tok = (32 // token_pool) ** 2
        self.alpha = Branch("v", in_ch=12)   # alpha band: vertical asymmetric kernel
        self.delta = Branch("h", in_ch=12)   # delta band: horizontal asymmetric kernel
        self.xa_alpha = CrossAttention(dim, heads, n_tokens=n_tok)   # alpha queries, delta keys/values
        self.xa_delta = CrossAttention(dim, heads, n_tokens=n_tok)   # delta queries, alpha keys/values
        self.fuse = nn.Sequential(nn.Conv2d(2 * dim, dim, 1, bias=False), nn.BatchNorm2d(dim), nn.ReLU(inplace=True))
        self.bottleneck = Bottleneck(dim)
        self.mha_norm = nn.LayerNorm(dim)
        self.mha = nn.MultiheadAttention(dim, mha_heads, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(dim, 512), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(512, num_classes)
        )

    @staticmethod
    def _tok(x):    # (B,C,H,W) -> (B,HW,C)
        return x.flatten(2).transpose(1, 2)

    def forward(self, alpha, delta):
        fa, fd = self.pool(self.alpha(alpha)), self.pool(self.delta(delta))   # (B,256,32/p,32/p) each
        B, C, H, W = fa.shape
        ta, td = self._tok(fa), self._tok(fd)
        ra = self.xa_alpha(ta, td)                               # alpha refined by delta
        rd = self.xa_delta(td, ta)                               # delta refined by alpha
        shared = ra + rd                                         # element-wise summation (shared)
        contrast = rd - ra                                       # element-wise subtraction (delta - alpha)
        x = torch.cat([shared, contrast], dim=-1)                # (B,HW,512)
        x = x.transpose(1, 2).reshape(B, 2 * C, H, W)
        x = self.bottleneck(self.fuse(x))                        # 512ch -> 256ch -> bottleneck
        t = self._tok(x)
        n = self.mha_norm(t)
        t = t + self.mha(n, n, n, need_weights=False)[0]         # multi-head self-attention
        return self.head(t.mean(1))


class BaselineCNN(nn.Module):
    """Plain CNN baseline / ablation on the same images: the two band images are stacked into 24 channels,
    with no asymmetric kernels, no attention and no cross-attention."""

    def __init__(self, num_classes=3, dropout=0.3):
        super().__init__()

        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o),
                                 nn.ReLU(inplace=True), nn.MaxPool2d(2))

        self.features = nn.Sequential(blk(24, 64), blk(64, 128), blk(128, 256), blk(256, 256))
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(256, num_classes))

    def forward(self, alpha, delta):
        x = self.features(torch.cat([alpha, delta], 1))
        return self.head(x.mean((2, 3)))

# =============================================================================================
# App additions: load the trained ensemble and predict
# =============================================================================================
import json
from pathlib import Path
import numpy as np

CLASS_NAMES = ['NORM', 'IMI', 'ASMI']


def _to_tensors(images_u8):
    """uint8 (N,2,12,128,128) -> two float tensors in [-1, 1], exactly as in training."""
    x = torch.from_numpy(np.ascontiguousarray(images_u8)).float().div_(127.5).sub_(1.0)
    return x[:, 0], x[:, 1]


class Ensemble:
    """The frozen final model: mean softmax of the seed networks + validation-tuned log offsets."""

    def __init__(self, ensemble_dir=None, runs_root=None, threads=4, checkpoints=None, bias=None):
        """Either ensemble_dir (reads ensemble_config.json: member runs + class offsets),
        or checkpoints = list of .pt files (any best.pt / last.pt you trained) with optional bias."""
        torch.set_num_threads(threads)
        self.members, self.nets = [], []
        self.classes = CLASS_NAMES
        if checkpoints:
            for ck in checkpoints:
                self._add(Path(ck))
            self.bias = np.zeros(3) if bias is None else np.asarray(bias, dtype=float)
            return
        ensemble_dir = Path(ensemble_dir)
        cfg = json.loads((ensemble_dir / 'ensemble_config.json').read_text())
        self.bias = np.asarray(cfg['bias'], dtype=float)
        self.classes = cfg.get('classes', CLASS_NAMES)
        root = Path(runs_root) if runs_root else ensemble_dir.parent.parent
        for rd in cfg['run_dirs']:
            rd = Path(rd)
            if not rd.is_absolute():
                rd = root / rd
            self._add(rd / 'best.pt')
        if not self.nets:
            raise ValueError('ensemble_config.json lists no member runs')

    def _add(self, ckpt):
        """Load one trained network. Accepts best.pt (weights only) or last.pt (full training state:
        the EMA weights are used if present). Architecture settings come from config.json next to the
        file; the attention-grid size is also read from the weights, so older runs load too."""
        if not ckpt.is_file():
            raise FileNotFoundError(f'Checkpoint not found: {ckpt}')
        obj = torch.load(ckpt, map_location='cpu', weights_only=False)   # only load your own files
        if isinstance(obj, dict) and obj.get('ema') is not None:
            state = obj['ema']
        elif isinstance(obj, dict) and 'model' in obj and isinstance(obj['model'], dict):
            state = obj['model']
        else:
            state = obj
        cfgf = ckpt.parent / 'config.json'
        mcfg = json.loads(cfgf.read_text()) if cfgf.is_file() else {}
        if mcfg.get('model', 'sspnet') != 'sspnet':
            raise ValueError(f'{ckpt} is a {mcfg.get("model")} model; this app runs SSPNet checkpoints.')
        n_tok = state['xa_alpha.pos'].shape[1]
        token_pool = int(round(32 / n_tok ** 0.5))
        net = SSPNet(num_classes=3, dropout=mcfg.get('dropout', 0.4), token_pool=token_pool)
        net.load_state_dict(state)
        net.eval()
        self.nets.append(net)
        self.members.append(str(ckpt))

    @torch.inference_mode()
    def member_probs(self, images_u8):
        """-> (n_members, N, 3) softmax probabilities."""
        a, d = _to_tensors(images_u8)
        return np.stack([torch.softmax(net(a, d), 1).numpy() for net in self.nets])

    def decide(self, p_mean):
        """Final rule: argmax(log p + b), returned with renormalised adjusted probabilities."""
        z = np.log(np.clip(p_mean, 1e-12, 1.0)) + self.bias[None, :]
        z -= z.max(1, keepdims=True)
        e = np.exp(z)
        return e.argmax(1), e / e.sum(1, keepdims=True)

    def predict(self, images_u8):
        pm = self.member_probs(images_u8)
        p_mean = pm.mean(0)
        label, p_adj = self.decide(p_mean)
        return dict(member_probs=pm, prob=p_mean, prob_adjusted=p_adj, label=label)

    @torch.inference_mode()
    def lead_importance(self, image_u8, cls):
        """Occlusion: blank one lead in both images (set to background) and measure the drop in the
        ensemble probability of class `cls`. Larger drop = the decision relied more on that lead."""
        base = self.member_probs(image_u8[None]).mean(0)[0, cls]
        batch = np.repeat(image_u8[None], 12, axis=0).copy()
        for l in range(12):
            batch[l, :, l] = 0                         # 0 -> -1.0 after scaling = empty image
        p = self.member_probs(batch).mean(0)[:, cls]
        return base - p
