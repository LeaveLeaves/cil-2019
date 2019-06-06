from .ade import ADE
from .voc import VOC
from .cityscapes import Cityscapes
from .camvid import CamVid
# from .stuff10k import Stuff10K
from .pcontext import PascalContext
from .cil import Cil

__all__ = ['ADE', 'VOC', 'Cityscapes', 'Cil', 'CamVid', 'PascalContext']