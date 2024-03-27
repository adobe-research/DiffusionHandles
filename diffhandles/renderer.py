from __future__ import annotations

from typing import List, Dict, Any, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod

import torch

class Camera:
    def __init__(self, intrinsics, extrinsics_R=None, extrinsics_t=None):
        self.intrinsics = intrinsics
        self.extrinsics_R = extrinsics_R if extrinsics_R is not None else torch.eye(3, device=intrinsics.device, dtype=intrinsics.dtype)
        self.extrinsics_t = extrinsics_t if extrinsics_t is not None else torch.zeros(3, device=intrinsics.device, dtype=intrinsics.dtype)

@dataclass
class RendererArgs:
    device: Union[str, torch.device] = "cpu"


class Renderer(torch.nn.Module, ABC):
    def __init__(self, args: RendererArgs=RendererArgs()):
        super().__init__()

        self.device = torch.device(args.device)
        self.args = args

    @abstractmethod
    def update_scene(
        self, scene_elements: Dict[str, Any], ignore_unsupported_elements: bool = False
    ):
        """
        Update the renderer's scene representation with new/updated elements.

        Args:
            scene_elements: dictionary of new/updated scene elements.
                These will be interpreted by the renderer implementations based on their names.
            ignore_unsupported_elements: ignore scene elements that are not supported by the renderer.
        """
        pass

    @abstractmethod
    def set_output_layers(self, output_names: List[str]):
        """
        Define which layers to output.

        Args:
            output_names: names of layers to output as list of strings.
        """
        pass

    @abstractmethod
    def render(self) -> Dict[str, torch.Tensor]:
        """
        Render the scene from all cameras.

        Returns:
            A dictionary of named output layers.
        """
        channels = {}
        return channels
