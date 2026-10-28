"""Selectable, fixed-length CDR1/CDR2 residue features."""
from collections import OrderedDict

import torch

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
ESM2_MODELS = {
    "esm2_t6_8M_UR50D": 320,
    "esm2_t12_35M_UR50D": 480,
    "esm2_t30_150M_UR50D": 640,
    "esm2_t33_650M_UR50D": 1280,
}
DEFAULT_ESM2_MODEL = "esm2_t33_650M_UR50D"


def cdr12_feature_dim(encoding="atchley", esm2_model=DEFAULT_ESM2_MODEL):
    if encoding == "atchley":
        return 5
    if encoding == "onehot":
        return 20
    if encoding == "esm2" and esm2_model in ESM2_MODELS:
        return ESM2_MODELS[esm2_model]
    raise ValueError(f"Unsupported CDR12 encoding/model: {encoding!r}, {esm2_model!r}")


def cdr12_converter_kwargs(model_config):
    """Use the same settings for data conversion and the TCR model."""
    return {
        "cdr12_feature_encoding": model_config.get("cdr12_feature_encoding", "atchley"),
        "cdr12_esm2_model": model_config.get("cdr12_esm2_model", DEFAULT_ESM2_MODEL),
        "cdr12_esm2_features_path": model_config.get("cdr12_esm2_features_path"),
    }


def load_cdr12_feature_cache(path, model_name):
    """Validate portable, CPU-only pre-extracted final-layer loop features."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    dim = cdr12_feature_dim("esm2", model_name)
    expected = {"format_version": 1, "model_name": model_name,
                "representation_layer": int(model_name.split("_")[1][1:]),
                "feature_dim": dim, "context": "individual_cdr_loop"}
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"CDR12 cache {path}: {key} must be {value!r}, got {payload.get(key)!r}")
    features = payload.get("features")
    if not isinstance(features, dict):
        raise ValueError(f"CDR12 cache {path}: missing features dictionary")
    for seq, tensor in features.items():
        if (not isinstance(seq, str) or not seq or
                not isinstance(tensor, torch.Tensor) or
                tensor.shape != (len(seq), dim) or not tensor.is_floating_point() or
                not torch.isfinite(tensor).all()):
            raise ValueError(f"CDR12 cache {path}: invalid feature tensor for {seq!r}")
    return payload


class CDR12FeatureEncoder:
    """Frozen CPU ESM2 features, cached by loop sequence within each worker.

    Each CDR loop is embedded separately. BOS/EOS representations are excluded;
    missing loops and padding remain zero. No CUDA model is created in workers.
    """

    @classmethod
    def from_config(cls, model_config):
        options = cdr12_converter_kwargs(model_config)
        return cls(options['cdr12_feature_encoding'], options['cdr12_esm2_model'],
                   options['cdr12_esm2_features_path'])

    def __init__(self, encoding="atchley", esm2_model=DEFAULT_ESM2_MODEL, esm2_features_path=None):
        self.encoding = encoding
        self.esm2_model = esm2_model
        self.dim = cdr12_feature_dim(encoding, esm2_model)
        self._model = None
        self._cache = OrderedDict()
        self.esm2_features_path = esm2_features_path
        self._precomputed = None
        if esm2_features_path is not None and encoding != "esm2":
            raise ValueError("cdr12_esm2_features_path requires esm2 encoding")

    def _esm_features(self, sequences):
        if self.esm2_features_path is not None:
            if self._precomputed is None:
                self._precomputed = load_cdr12_feature_cache(
                    self.esm2_features_path, self.esm2_model)["features"]
            missing = sorted({s for s in sequences if s and s not in self._precomputed})
            if missing:
                raise KeyError(
                    f"CDR12 ESM2 cache {self.esm2_features_path} is missing {len(missing)} "
                    f"loop sequence(s), e.g. {missing[:5]}. Regenerate the cache "
                    "with all required loop sequences before training/evaluation."
                )
            return {s: self._precomputed[s] for s in sequences if s}
        missing = list(dict.fromkeys(s for s in sequences if s and s not in self._cache))
        if missing and self._model is None:
            try:
                import esm
            except ImportError as exc:
                raise ImportError("CDR12 ESM2 encoding requires fair-esm (pip install fair-esm).") from exc
            self._model, alphabet = esm.pretrained.load_model_and_alphabet(self.esm2_model)
            self._model = self._model.cpu().eval().requires_grad_(False)
            self._batch_converter = alphabet.get_batch_converter()
        for start in range(0, len(missing), 64):
            chunk = missing[start:start + 64]
            _, _, tokens = self._batch_converter([(str(i), s.replace("*", "X")) for i, s in enumerate(chunk)])
            layer = self._model.num_layers
            with torch.no_grad():
                representations = self._model(tokens, repr_layers=[layer], return_contacts=False)["representations"][layer]
            for i, seq in enumerate(chunk):
                self._cache[seq] = representations[i, 1:len(seq) + 1].float().clone()
                # '*' is the legacy zero-feature placeholder, not an ESM token.
                for position, aa in enumerate(seq):
                    if aa == "*":
                        self._cache[seq][position] = 0
        return {s: self._cache[s] for s in sequences if s}

    def __call__(self, batch):
        if self.encoding == "atchley":
            from .encoding import paired_cdr12_batch_encoding
            return paired_cdr12_batch_encoding(batch)
        loops = [("cdr1a", 8), ("cdr2a", 8), ("cdr1b", 6), ("cdr2b", 7)]
        sequences = []
        for key, max_len in loops:
            for row in batch:
                seq = row[key] if isinstance(row[key], str) else ""
                if len(seq) > max_len:
                    raise ValueError(f"{key} length {len(seq)} exceeds maximum {max_len}")
                if any(aa not in AMINO_ACIDS + "X*" for aa in seq):
                    raise ValueError(f"Unsupported residue in {key}: {seq!r}")
                sequences.append(seq)
        features = self._esm_features(sequences) if self.encoding == "esm2" else {}
        encoded = []
        for loop_idx, (_, max_len) in enumerate(loops):
            out = torch.zeros(len(batch), max_len, self.dim)
            for i, seq in enumerate(sequences[loop_idx * len(batch):(loop_idx + 1) * len(batch)]):
                if self.encoding == "esm2" and seq:
                    out[i, :len(seq)] = features[seq]
                elif self.encoding == "onehot":
                    for j, aa in enumerate(seq):
                        if aa in AMINO_ACIDS:
                            out[i, j, AMINO_ACIDS.index(aa)] = 1
            encoded.append(out)
        # Bound memory without evicting features needed by the current batch.
        while len(self._cache) > 4096:
            self._cache.popitem(last=False)
        return torch.cat(encoded[:2], dim=1), torch.cat(encoded[2:], dim=1)
