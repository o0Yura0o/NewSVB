# -*- coding: utf-8 -*-
"""NSVB-ZH voice conversion extensions.

These modules are additive to the original NSVB repo's `modules/voice_conversion/`
package. Import paths assume the original repo root is on sys.path, i.e.
drop these files into modules/voice_conversion/ directly.
"""
from modules.voice_conversion.nsvb_zh_model import (
    ResidualM,
    DiscriminatorZ,
    DomainDiscriminator,
    PatchNCELoss,
    grad_reverse,
    hinge_d_loss,
    hinge_g_loss,
    domain_bce_loss,
)
from modules.voice_conversion.soft_bucket import (
    SoftRegisterEncoder,
    f0_to_soft_register,
    build_register_centers,
)

__all__ = [
    'ResidualM',
    'DiscriminatorZ',
    'DomainDiscriminator',
    'PatchNCELoss',
    'SoftRegisterEncoder',
    'f0_to_soft_register',
    'build_register_centers',
    'grad_reverse',
    'hinge_d_loss',
    'hinge_g_loss',
    'domain_bce_loss',
]
