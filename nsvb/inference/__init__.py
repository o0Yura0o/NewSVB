"""
nsvb/inference/
=================

NSVB-ZH 推理子套件。

【模組組織】
    feature_pipeline : 單檔特徵抽取（mel/F0/PPG/spk_emb/...）
    model_loader     : Stage 1 + Stage 2 + Vocoder ckpt 載入
    dtw_warp         : Mode B 的 DTW + gather 時間對齊
    pipeline         : run_mode_a / run_mode_b 主入口

【公開 API】
    最常用的兩個 entrypoint：
        from nsvb.inference import run_mode_a, run_mode_b

    需要自行組裝（例如 batch 推理、串 web demo）時：
        from nsvb.inference import (
            InferenceFeatureExtractor, load_inference_models,
            InferenceFeatures, InferenceModels, InferenceResult,
        )

【為什麼把這些 re-export 到頂層】
- 終端使用者寫 `from nsvb.inference import run_mode_a` 比
  `from nsvb.inference.pipeline import run_mode_a` 簡潔
- 頂層 __init__ 是「公開介面」契約；未來重構內部 module 不會破壞外部 import
"""

from nsvb.inference.feature_pipeline import (
    InferenceFeatureExtractor,
    InferenceFeatures,
    features_to_batch,
)
from nsvb.inference.model_loader import (
    InferenceModels,
    load_inference_models,
)
from nsvb.inference.pipeline import (
    InferenceResult,
    run_mode_a,
    run_mode_b,
)

__all__ = [
    # Feature extraction
    "InferenceFeatureExtractor",
    "InferenceFeatures",
    "features_to_batch",
    # Model loading
    "InferenceModels",
    "load_inference_models",
    # Pipeline
    "InferenceResult",
    "run_mode_a",
    "run_mode_b",
]