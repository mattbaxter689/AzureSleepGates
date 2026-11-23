from pydantic import BaseModel


class PSIConfig(BaseModel):
    n_bins: int = 10
    eps: float = 1e-6


class DetectionConfig(BaseModel):
    threshold: float = 0.25
    rng: int = 42
    scale: float = 1e-6


class DriftConfig(BaseModel):
    psi: PSIConfig
    drift: DetectionConfig
