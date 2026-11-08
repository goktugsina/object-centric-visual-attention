"""Small, permutation-equivariant region-choice heads with explicit controls."""
import numpy as np
import torch
from torch import nn
from scipy.stats import rankdata


class RegionNetwork(nn.Module):
    def __init__(self, input_dim, hidden=64, relation='self'):
        super().__init__()
        if relation not in ('self', 'uniform', 'learned'):
            raise ValueError('Unknown relation policy')
        self.relation = relation
        self.encoder = nn.Linear(input_dim, hidden)
        self.value = nn.Linear(hidden, hidden, bias=False)
        self.source_score = nn.Parameter(torch.empty(hidden))
        self.target_score = nn.Parameter(torch.empty(hidden))
        nn.init.normal_(self.source_score, std=hidden**-.5)
        nn.init.normal_(self.target_score, std=hidden**-.5)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x, valid):
        h = torch.relu(self.encoder(x))
        z = self.value(h)
        if self.relation == 'self':
            message = z
        elif self.relation == 'uniform':
            message = (z*valid[..., None]).sum(1, keepdim=True)/valid.sum(1)[:, None, None]
        else:
            source = z @ self.source_score
            target = z @ self.target_score
            logits = torch.nn.functional.leaky_relu(source[:, :, None]+target[:, None, :], .2)
            weights = torch.softmax(logits.masked_fill(~valid[:, None, :], -torch.inf), -1)
            message = weights @ z
        logits = self.head(torch.relu(h+message)).squeeze(-1)
        return logits.masked_fill(~valid, -torch.inf)


def image_cross_entropy(logits, targets, valid):
    logp = torch.log_softmax(logits, -1).masked_fill(~valid, 0)
    return -(targets*logp).sum(-1).mean()


def fit_normalizer(features):
    """Each training image contributes equally, regardless of its region count."""
    if not features:
        raise ValueError('Training features required')
    mean = np.mean([x.astype(float).mean(0) for x in features], axis=0)
    second = np.mean([(x.astype(float)**2).mean(0) for x in features], axis=0)
    std = np.sqrt(np.maximum(0, second-mean**2))
    std[std < 1e-5] = 1.
    return mean.astype(np.float32), std.astype(np.float32)


def batch_tensors(features, scenes, indices, mean, std, device='cpu'):
    size = max(len(features[i]) for i in indices)
    x = np.zeros((len(indices), size, len(mean)), np.float32)
    q = np.zeros((len(indices), size), np.float32)
    valid = np.zeros((len(indices), size), bool)
    for row, i in enumerate(indices):
        n = len(features[i])
        x[row, :n] = (features[i]-mean)/std
        counts = scenes[i]['counts']
        q[row, :n] = counts/counts.sum()
        valid[row, :n] = True
    return tuple(torch.as_tensor(a, device=device) for a in (x, q, valid))


@torch.inference_mode()
def predict_network(model, features, scenes, mean, std, device='cpu', batch_size=32):
    model.eval()
    predictions = []
    for start in range(0, len(scenes), batch_size):
        indices = list(range(start, min(start+batch_size, len(scenes))))
        x, _, valid = batch_tensors(features, scenes, indices, mean, std, device)
        p = torch.softmax(model(x, valid), -1).cpu().numpy()
        for row, i in enumerate(indices):
            mass = np.maximum(p[row, :len(features[i])].astype(float), 1e-12)
            predictions.append(mass/mass.sum())
    return predictions


def observer_disagreement(counts):
    counts = np.asarray(counts, float)
    counts = counts[counts.sum(1) > 0]
    if not len(counts):
        raise ValueError('No eligible observer')
    p = counts/counts.sum(1, keepdims=True)
    mean = p.mean(0)
    def entropy(a):
        return -np.sum(a*np.log2(np.maximum(a, 1e-300)), axis=-1)
    return float(entropy(mean)-entropy(p).mean())


def rank_scores(q, p, k=3):
    """Background included. NDCG uses linear relevance and expected tied order."""
    q, p = np.asarray(q, float), np.asarray(p, float)
    a, b = rankdata(q), rankdata(p)
    rho = None if np.ptp(a) == 0 or np.ptp(b) == 0 else float(np.corrcoef(a, b)[0, 1])
    order = np.argsort(-p, kind='stable')
    discounts = np.zeros(len(p))
    discounts[:min(k, len(p))] = 1/np.log2(np.arange(min(k, len(p)))+2)
    dcg = 0.
    start = 0
    while start < len(p):
        end = start+1
        while end < len(p) and p[order[end]] == p[order[start]]:
            end += 1
        dcg += q[order[start:end]].mean()*discounts[start:end].sum()
        start = end
    ideal = np.sort(q)[::-1] @ discounts
    return dict(spearman=rho, ndcg3=float(dcg/ideal), mae=float(np.abs(q-p).mean()))
