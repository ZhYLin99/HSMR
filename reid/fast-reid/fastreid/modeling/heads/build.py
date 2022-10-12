# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from ...utils.registry import Registry

REID_HEADS_REGISTRY = Registry("HEADS")
REID_HEADS_REGISTRY.__doc__ = """
Registry for reid heads in a baseline model.

ROIHeads take feature maps and region proposals, and
perform per-region computation.
The registered object will be called with `obj(cfg, input_shape)`.
The call is expected to return an :class:`ROIHeads`.
"""
REID_HEADS_REGISTRY = Registry("MID_HEADS")

def build_heads(cfg):
    """
    Build REIDHeads defined by `cfg.MODEL.REID_HEADS.NAME`.
    """
    head = cfg.MODEL.HEADS.NAME
    return REID_HEADS_REGISTRY.get(head)(cfg)


def build_mid_heads(cfg):
    """
    Build REIDHeads defined by `cfg.MODEL.REID_HEADS.NAME`.
    """
    head = cfg.MODEL.MID_HEADS.NAME
    return REID_HEADS_REGISTRY.get(head)(cfg)


def build_mid_heads(cfg, n):
    """
    Build REIDHeads defined by `cfg.MODEL.REID_HEADS.NAME`.
    """
    if n == 1:
        head = cfg.MODEL.L1_HEADS.NAME
    elif n == 2:
        head = cfg.MODEL.L2_HEADS.NAME
    elif n == 3:
        head = cfg.MODEL.L3_HEADS.NAME
    return REID_HEADS_REGISTRY.get(head)(cfg)
