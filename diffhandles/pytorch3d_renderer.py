from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import itertools

import torch
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    MeshRasterizer,
    RasterizationSettings,
    TexturesUV,
    look_at_view_transform,
)
from pytorch3d.renderer.mesh.shader import ShaderBase

from pytorch3d.renderer.blending import (
    BlendParams,
    sigmoid_alpha_blend,
    _get_background_color,
)
from pytorch3d.structures import Meshes as Pytorch3DMeshes

# from pytorch3d.structures.utils import list_to_packed
from pytorch3d.structures.meshes import join_meshes_as_scene

from diffhandles.mesh import Mesh
from diffhandles.renderer import Renderer, RendererArgs


@dataclass
class PyTorch3DRendererArgs(RendererArgs):
    # rasterizer settings
    output_res: Union[int, Tuple[int, int]] = 512 # (height, width) when providing two values
    blur_radius: float = 0
    faces_per_pixel: int = 1
    perspective_correct: bool = True
    cull_backfaces: bool = False
    clip_barycentric_coords: bool = True
    bin_size: Optional[int] = None

    # shader settings
    # blending type, currently supports: ("hard", "sigmoid")
    blend_type: str = "hard"
    # blending parameters for smoothly combining multiple fragments into a pixel
    blend_sigma: float = 1e-4
    blend_gamma: float = 1e-4
    # backgrouund color for pixels not covered by scene content
    background_color: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # near frustum plane
    z_near = 0.1
    # far


def hard_rgb_blend(
    colors: torch.Tensor,
    fragments,
    blend_params: BlendParams,
    background_color: torch.Tensor = None,
) -> torch.Tensor:
    """
    Naive blending of top K faces to return an RGBA image
      - **RGB** - choose color of the closest point i.e. K=0
      - **A** - 1.0

    Args:
        colors: (N, H, W, K, 3) RGB color for each of the top K faces per pixel.
        fragments: the outputs of rasterization. From this we use
            - pix_to_face: LongTensor of shape (N, H, W, K) specifying the indices
              of the faces (in the packed representation) which
              overlap each pixel in the image. This is used to
              determine the output shape.
        blend_params: BlendParams instance that contains a background_color
        field specifying the color for the background
    Returns:
        RGBA pixel_colors: (N, H, W, 4)
    """
    # background_color = _get_background_color(blend_params, fragments.pix_to_face.device)

    if background_color is None:
        background_color = torch.zeros(
            size=[colors.shape[-1]], device=colors.device, dtype=colors.dtype
        )
    elif isinstance(background_color, float):
        background_color = torch.full(
            fill_value=background_color,
            size=[colors.shape[-1]],
            device=colors.device,
            dtype=colors.dtype,
        )
    if background_color.shape[-1] != colors.shape[-1]:
        raise RuntimeError(
            "The background does not have the same number of channels as the render layer."
        )
    # background_color = background_color[: colors.shape[-1]]
    # if background_color.shape[0] < colors.shape[-1]:
    #     background_color = torch.cat(
    #         [
    #             background_color,
    #             torch.zeros(
    #                 size=[colors.shape[-1] - background_color.shape[0]],
    #                 device=background_color.device,
    #                 dtype=background_color.dtype,
    #             ),
    #         ]
    #     )

    # Mask for the background.
    is_background = fragments.pix_to_face[..., 0] < 0  # (N, H, W)

    # Find out how much background_color needs to be expanded to be used for masked_scatter.
    num_background_pixels = is_background.sum()

    # Set background color.
    pixel_colors = colors[..., 0, :].masked_scatter(
        is_background[..., None],
        background_color[None, :].expand(num_background_pixels, -1),
    )  # (N, H, W, 3)

    # Concat with the alpha channel.
    alpha = (~is_background).type_as(pixel_colors)[..., None]

    return torch.cat([pixel_colors, alpha], dim=-1)  # (N, H, W, 4)


def mesh_barycentric_interpolation(
    verts: torch.Tensor,
    faces: torch.Tensor,
    face_inds: torch.Tensor,
    bary_coords: torch.Tensor,
):
    """
    Barycentric interpolation on the triangles of a mesh.

    Args:
        values: vertex_count-3 tensor
        faces: face_count-3 tensor
        face_inds: selected_face_count-3 tensor
        bary_coords: selected_face_count-3 tensor
    """
    return (verts[faces[face_inds, :], :] * bary_coords[..., None]).sum(dim=1)


class MultioutputMeshRenderer(torch.nn.Module):
    """
    A class for rendering a batch of heterogeneous meshes into multiple output, where
    each output may be produced with a different shader. The class should
    be initialized with a rasterizer (a MeshRasterizer or a MeshRasterizerOpenGL)
    and multiple shaders which each have a forward function.
    """

    @dataclass
    class ShaderChannelSelection:
        channel_list: Optional[List[int]] = None
        shader_idx: int = 0

    def __init__(
        self,
        rasterizer: Any,
        shaders: List[ShaderBase],
        output_mapping: Optional[Dict[str, ShaderChannelSelection]] = None,
    ) -> None:
        super().__init__()
        self.rasterizer = rasterizer
        self.shaders = torch.nn.ModuleList(shaders)
        self.output_mapping = output_mapping

    def to(self, device: torch.device):
        # Rasterizer and shader have submodules which are not of type nn.Module
        self.rasterizer.to(device)
        for shader in self.shaders:
            shader.to(device)
        return self

    def forward(self, meshes_world: Pytorch3DMeshes, **kwargs) -> torch.Tensor:
        """
        Render a batch of images from a batch of meshes by rasterizing and then
        shading with each of the shaders.

        NOTE: If the blur radius for rasterization is > 0.0, some pixels can
        have one or more barycentric coordinates lying outside the range [0, 1].
        For a pixel with out of bounds barycentric coordinates with respect to a
        face f, clipping is required before interpolating the texture uv
        coordinates and z buffer so that the colors and depths are limited to
        the range for the corresponding face.
        For this set rasterizer.raster_settings.clip_barycentric_coords=True
        """
        outputs = {}

        fragments = self.rasterizer(meshes_world, **kwargs)

        for shader_idx, shader in enumerate(self.shaders):
            image = shader(fragments, meshes_world, **kwargs)

            for output_name, output_selection in self.output_mapping.items():
                if output_selection.shader_idx == shader_idx:
                    if output_selection.channel_list is None:
                        outputs[output_name] = image
                    else:
                        if any(
                            channel_idx >= image.shape[-1]
                            for channel_idx in output_selection.channel_list
                        ):
                            raise RuntimeError(
                                f"Channel indices out of bounds for output {output_name}"
                            )

                        outputs[output_name] = image[..., output_selection.channel_list]

        return outputs


class MeshAttributeShader(ShaderBase):
    """
    A shader that returns mesh attributes in a given coordinate system.
    """

    def __init__(
        self,
        mesh_attribute: str = "face",
        mesh_coords: str = "camera",
        cameras=None,
        blend_type="hard",
        blend_params=None,
        blend_renormalize=False,
        eps=1e-8,
        device: Union[str, torch.device] = "cpu",
    ):
        """
        Create a mesh attribute shader.

        Args:
            mesh_attribute: name of the mesh attribute, one of:
                "vertex_normal": interpolated vertex normals
                "face_normal": normal of the face the fragment is part of.
                "vertex_position": position of the fragments (interpolated vertex positions).
            mesh_coords: coordinate system the returned attributes will be in. One of:
                "world":  the world coordinate frame (same coordinate frame as the meshes);
                "camera": the camera coordinate frame
                "ndc": the normalized device coordinate frame.
            cameras: scene cameras, can be left empty if the cameras are passed at rendering time (the default).
            blend_type: fragment blending type. One of:
                "hard": For each pixel, only the fragment closest to the camera is used. This is the default non-differentiable option.
                "sigmoid": For each pixel, fragments are blended with a sigmoid based on their distance from the closest fragment.
                    This option enables differentiability w.r.t fragment depth.
            blend_params: Fragment blending parameters. See pytorch3d.renderer.blending.BlendParams for details.
            blend_renormalize: Renormalize normals after fragment blending.
            eps: Epsilon for mesh coordinate transformations and post-blending renormalization.
        """

        super().__init__(device=device, cameras=cameras, blend_params=blend_params)
        self.mesh_attribute = mesh_attribute
        self.mesh_coords = mesh_coords
        self.blend_type = blend_type
        self.blend_renormalize = blend_renormalize
        self.eps = eps

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        blend_params = kwargs.get("blend_params", self.blend_params)
        cameras = kwargs.get("cameras", self.cameras)

        if len(cameras) != 1 and len(cameras) != len(meshes):
            raise RuntimeError(
                f"Wrong number ({len(cameras)}) of cameras for {len(meshes)} meshes."
            )

        # Get mesh with vertices in the given coordinate system.
        # (Below is copied from the MeshRasterizer, which also performs coordinate system conversions.)
        if self.mesh_coords == "world":
            meshes_transformed = meshes

        elif self.mesh_coords == "camera":
            verts_world = meshes.verts_padded()
            verts_cam = cameras.get_world_to_view_transform(**kwargs).transform_points(
                verts_world, eps=self.eps
            )
            meshes_cam = meshes.update_padded(new_verts_padded=verts_cam)
            meshes_transformed = meshes_cam

        elif self.mesh_coords == "ndc":
            verts_world = meshes.verts_padded()
            to_ndc_transform = cameras.get_ndc_camera_transform(**kwargs)
            verts_proj = cameras.transform_points(verts_world, eps=self.eps)
            verts_ndc = to_ndc_transform.transform_points(verts_proj, eps=self.eps)
            meshes_ndc = meshes.update_padded(new_verts_padded=verts_ndc)

            meshes_transformed = meshes_ndc

        else:
            raise RuntimeError(f"Unknown coordinate type: {self.mesh_coords}.")

        nonempty_fragments_mask = fragments.pix_to_face >= 0

        # Get/interpolate mesh attribute at each fragment.
        if self.mesh_attribute == "vertex_normal":
            nonempty_fragment_attributes = mesh_barycentric_interpolation(
                verts=meshes_transformed.verts_normals_packed(),
                faces=meshes_transformed.faces_packed(),
                face_inds=fragments.pix_to_face[nonempty_fragments_mask],
                bary_coords=fragments.bary_coords[nonempty_fragments_mask, :],
            )
            # re-normalize normals
            nonempty_fragment_attributes_norm = torch.linalg.norm(
                nonempty_fragment_attributes, ord=2, dim=-1, keepdim=True
            )
            nonempty_fragment_attributes = nonempty_fragment_attributes / torch.clamp(
                nonempty_fragment_attributes_norm, min=self.eps
            )

        elif self.mesh_attribute == "face_normal":
            nonempty_fragment_attributes = meshes_transformed.faces_normals_packed()[
                fragments.pix_to_face[nonempty_fragments_mask]
            ]

        elif self.mesh_attribute == "vertex_position":
            nonempty_fragment_attributes = mesh_barycentric_interpolation(
                verts=meshes_transformed.verts_packed(),
                faces=meshes_transformed.faces_packed(),
                face_inds=fragments.pix_to_face[nonempty_fragments_mask],
                bary_coords=fragments.bary_coords[nonempty_fragments_mask, :],
            )

        else:
            raise RuntimeError(f"Unknown mesh attribute: {self.mesh_attribute}.")

        # TODO: check that this gets fragment_attributes gets the right dimension
        fragment_attributes = torch.zeros(
            size=list(fragments.bary_coords.shape[:-1])
            + [nonempty_fragment_attributes.shape[-1]],
            dtype=torch.float32,
            device=nonempty_fragment_attributes.device,
        )
        fragment_attributes[nonempty_fragments_mask, :] = nonempty_fragment_attributes
        if self.blend_type == "hard":
            images = hard_rgb_blend(
                colors=fragment_attributes,
                fragments=fragments,
                blend_params=blend_params,
            )
        elif self.blend_type == "sigmoid":
            images = sigmoid_alpha_blend(
                colors=fragment_attributes,
                fragments=fragments,
                blend_params=blend_params,
            )

            # re-normalize normals
            if self.blend_renormalize and self.mesh_attribute in [
                "vertex_normal",
                "face_normal",
            ]:
                images_norm = torch.linalg.norm(images, ord=2, dim=-1, keepdim=True)
                images[..., :3] = images[..., :3] / torch.clamp(
                    images_norm, min=self.eps
                )
        else:
            raise RuntimeError(f"Unsupported blend type: {self.blend_type}")
        return images  # (N, H, W, 4) RGBA image


class DepthShader(ShaderBase):
    """
    A shader that returns z-buffer depth.
    """

    def __init__(
        self,
        blend_type="hard",
        blend_params=None,
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__(device=device, blend_params=blend_params)
        self.blend_type = blend_type

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        blend_params = kwargs.get("blend_params", self.blend_params)

        # The z-buffer in the MeshRasterizer is actually contains
        # the camera-space z-coordinate
        depth = fragments.zbuf[..., None]
        if self.blend_type == "hard":
            images = hard_rgb_blend(
                colors=depth,
                fragments=fragments,
                blend_params=blend_params,
                background_color=-1.0,
            )
        elif self.blend_type == "sigmoid":
            images = sigmoid_alpha_blend(
                colors=depth,
                fragments=fragments,
                blend_params=blend_params,
            )
        else:
            raise RuntimeError(f"Unsupported blend type: {self.blend_type}")
        return images  # (N, H, W, 4) RGBA image


class FlatGlobalVolumeTextureShader(ShaderBase):
    """
    A shader that uses volumetric textures without lighting.
    """

    def __init__(
        self,
        volume_texture=None,
        blend_type="hard",
        blend_params=None,
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__(device=device, blend_params=blend_params)
        self.volume_texture = volume_texture
        self.blend_type = blend_type

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        blend_params = kwargs.get("blend_params", self.blend_params)
        volume_texture = kwargs.get("global_volume_texture", self.volume_texture)

        nonempty_mask = fragments.pix_to_face >= 0
        nonempty_fragments_world_pos = mesh_barycentric_interpolation(
            verts=meshes.verts_packed(),
            faces=meshes.faces_packed(),
            face_inds=fragments.pix_to_face[nonempty_mask],
            bary_coords=fragments.bary_coords[nonempty_mask, :],
        )

        volume_texture_samples = volume_texture(nonempty_fragments_world_pos)
        fragment_colors = torch.zeros(
            size=list(fragments.bary_coords.shape[:-1])
            + [volume_texture_samples.shape[-1]],
            dtype=torch.float32,
            device=volume_texture_samples.device,
        )
        fragment_colors[nonempty_mask, :] = volume_texture_samples
        if self.blend_type == "hard":
            images = hard_rgb_blend(
                colors=fragment_colors,
                fragments=fragments,
                blend_params=blend_params,
            )
        elif self.blend_type == "sigmoid":
            images = sigmoid_alpha_blend(
                colors=fragment_colors,
                fragments=fragments,
                blend_params=blend_params,
            )
        else:
            raise RuntimeError(f"Unsupported blend type: {self.blend_type}")
        return images  # (N, H, W, 4) RGBA image


class FlatTextureShader(ShaderBase):
    """
    A shader that uses UV textures without lighting.
    """

    def __init__(
        self,
        blend_type="hard",
        blend_params=None,
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__(device=device, blend_params=blend_params)
        self.blend_type = blend_type

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        blend_params = kwargs.get("blend_params", self.blend_params)
        texels = meshes.sample_textures(fragments)
        if self.blend_type == "hard":
            images = hard_rgb_blend(
                colors=texels,
                fragments=fragments,
                blend_params=blend_params,
                background_color=blend_params.background_color,
            )
        elif self.blend_type == "sigmoid":
            images = sigmoid_alpha_blend(
                colors=texels,
                fragments=fragments,
                blend_params=blend_params
            )
        else:
            raise RuntimeError(f"Unsupported blend type: {self.blend_type}")
        return images  # (N, H, W, 3) RGBA image

class FlatVertexAttributeShader(ShaderBase):
    """
    A shader that uses vertex colors without lighting.
    """

    def __init__(
        self,
        attribute_name: str,
        blend_type="hard",
        blend_params=None,
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__(device=device, blend_params=blend_params)
        self.blend_type = blend_type
        self.attribute_name = attribute_name

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        blend_params = kwargs.get("blend_params", self.blend_params)
        vert_attributes = kwargs.get("vert_attributes", None)
        vert_attribute = vert_attributes[self.attribute_name]

        nonempty_fragments_mask = fragments.pix_to_face >= 0

        nonempty_fragment_attributes = mesh_barycentric_interpolation(
            verts=vert_attribute,
            faces=meshes.faces_packed(),
            face_inds=fragments.pix_to_face[nonempty_fragments_mask],
            bary_coords=fragments.bary_coords[nonempty_fragments_mask, :],
        )

        fragment_attributes = torch.zeros(
            size=list(fragments.bary_coords.shape[:-1])
            + [nonempty_fragment_attributes.shape[-1]],
            dtype=torch.float32,
            device=nonempty_fragment_attributes.device,
        )
        fragment_attributes[nonempty_fragments_mask, :] = nonempty_fragment_attributes

        if self.blend_type == "hard":
            images = hard_rgb_blend(
                colors=fragment_attributes,
                fragments=fragments,
                blend_params=blend_params,
            )
        elif self.blend_type == "sigmoid":
            images = sigmoid_alpha_blend(
                colors=fragment_attributes, fragments=fragments, blend_params=blend_params
            )
        else:
            raise RuntimeError(f"Unsupported blend type: {self.blend_type}")
        return images  # (N, H, W, 3) RGBA image


# TODO: make some scene properties like mesh vertices and camera position/orientation parameters that can be optimized.
class PyTorch3DRenderer(Renderer):
    def __init__(
        self,
        output_names: Optional[List[str]] = None,
        args: PyTorch3DRendererArgs = PyTorch3DRendererArgs(),
    ):
        """
        Create a PyTorch3D renderer.

        Args:
            output_names: names of the output layers to render as list of strings. Supported layers are:
                flat_texture: uv textured scene without illumination;
                flat_global_volume_texture: global volume textured scene without illumination;
                flat_vertex_color: vertex colored scene without illumination;
                world_position: position of rendered geometry in world coordinates;
                camera_position: position of rendered geometry in camera coordinates;
                ndc_position: position of rendered geometry in normalized device coordinates;
                world_face_normals: face normals of rendered geometry in world coordinates;
                camera_face_normals: face normals of rendered geometry in camera coordinates;
                ndc_face_normals: face normals of rendered geometry in normalized device coordinates;
                depth: camera-space z-coordinate.
            args: arguments of type PyTorch3DRendererArguments.
        """
        super().__init__(args=args)

        if output_names is None:
            output_names = ["flat_texture"]

        self.set_output_layers(output_names=output_names)

        # The scene is used to re-build PyTorch3D's internal scene representation in each frame.
        self._scene = {}

        # Auxiliary meshes with all-zero vertices that are only used to avoid reconstructing the full meshes in each frame,
        # by only offsetting their vertices to the actual vertex positions in each frame.
        self._zero_meshes = Pytorch3DMeshes(verts=[], faces=[])

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

        # make sure all scene elements are supported
        if not ignore_unsupported_elements:
            supported_scene_elements = [
                "meshes",
                "global_volume_texture",
                "uv_textures",
                "cameras",
            ]
            unsupported_scene_elements = set(scene_elements.keys()) - set(
                supported_scene_elements
            )
            if len(unsupported_scene_elements) > 0:
                raise RuntimeError(
                    f"Unsupported scene elements for the PyTorch3D renderer: {unsupported_scene_elements}"
                )

        self._scene.update(scene_elements)

        uv_textures = None
        if "uv_textures" in scene_elements:
            uv_textures = self._get_uv_textures()

        # update the all-zero meshes to the new vertex and face count
        # (and the new textures if given)
        if "meshes" in scene_elements:
            scene_meshes = scene_elements["meshes"]

            if not isinstance(scene_meshes, list) or not all(
                isinstance(mesh, Mesh) for mesh in scene_meshes
            ):
                raise RuntimeError(
                    "Provided meshes must be given as list of geometry.mesh.Mesh"
                )

            # convert given meshes to PyTorch3D and join them into a single mesh
            self._zero_meshes = join_meshes_as_scene(Pytorch3DMeshes(
                verts=[torch.zeros_like(mesh.verts) for mesh in scene_meshes],
                faces=[mesh.faces for mesh in scene_meshes],
                textures=uv_textures,
            ))


    def set_output_layers(self, output_names: List[str]):
        """
        Define which layers to output.

        Args:
            output_names: names of layers to output as list of strings. Supported layers are:
                flat_texture: uv textured scene without illumination;
                flat_global_volume_texture: global volume textured scene without illumination;
                flat_vertex_color: vertex colored scene without illumination;
                world_position: position of rendered geometry in world coordinates;
                camera_position: position of rendered geometry in camera coordinates;
                ndc_position: position of rendered geometry in normalized device coordinates;
                world_face_normals: face normals of rendered geometry in world coordinates;
                camera_face_normals: face normals of rendered geometry in camera coordinates;
                ndc_face_normals: face normals of rendered geometry in normalized device coordinates;
                depth: camera-space z-coordinate.
        """
        supported_outputs = [
            "flat_texture",
            "flat_global_volume_texture",
            "flat_vertex_color",
            "world_position",
            "camera_position",
            "ndc_position",
            "world_face_normals",
            "camera_face_normals",
            "ndc_face_normals",
            "depth",
        ]
        unsupported_outputs = set(output_names) - set(supported_outputs)
        if len(unsupported_outputs) > 0:
            raise RuntimeError(f"Unsupported output channels: {unsupported_outputs}.")

        # create rasterizer
        raster_settings = RasterizationSettings(
            image_size=self.args.output_res,
            blur_radius=self.args.blur_radius,
            faces_per_pixel=self.args.faces_per_pixel,
            perspective_correct=self.args.perspective_correct,
            cull_backfaces=self.args.cull_backfaces,
            clip_barycentric_coords=self.args.clip_barycentric_coords,
            bin_size=self.args.bin_size,
        )
        rasterizer = MeshRasterizer(raster_settings=raster_settings)

        # create shader
        shaders = []
        output_mapping = {}
        for output_name in output_names:
            if output_name == "flat_texture":
                blend_params = BlendParams(
                    sigma=self.args.blend_sigma,
                    gamma=self.args.blend_gamma,
                    background_color=self.args.background_color,
                )
                shader = FlatTextureShader(
                    blend_type=self.args.blend_type, blend_params=blend_params
                )
                shaders.append(shader)
                output_mapping[
                    output_name
                ] = MultioutputMeshRenderer.ShaderChannelSelection(
                    shader_idx=len(shaders) - 1, channel_list=None
                )

            elif output_name == "flat_vertex_color":
                blend_params = BlendParams(
                    sigma=self.args.blend_sigma,
                    gamma=self.args.blend_gamma,
                    background_color=self.args.background_color,
                )
                shader = FlatVertexAttributeShader(
                    attribute_name="color",
                    blend_type=self.args.blend_type,
                    blend_params=blend_params
                )
                shaders.append(shader)
                output_mapping[
                    output_name
                ] = MultioutputMeshRenderer.ShaderChannelSelection(
                    shader_idx=len(shaders) - 1, channel_list=None
                )

            elif output_name == "flat_global_volume_texture":
                blend_params = BlendParams(
                    sigma=self.args.blend_sigma,
                    gamma=self.args.blend_gamma,
                    background_color=self.args.background_color,
                )
                shader = FlatGlobalVolumeTextureShader(
                    blend_type=self.args.blend_type,
                    blend_params=blend_params,
                )
                shaders.append(shader)
                output_mapping[
                    output_name
                ] = MultioutputMeshRenderer.ShaderChannelSelection(
                    shader_idx=len(shaders) - 1, channel_list=None
                )

            elif output_name in ["world_position", "camera_position", "ndc_position"]:
                blend_params = BlendParams(
                    sigma=self.args.blend_sigma,
                    gamma=self.args.blend_gamma,
                    background_color=self.args.background_color,
                )
                shader = MeshAttributeShader(
                    mesh_attribute="vertex_position",
                    mesh_coords=output_name.split("_")[0],
                    blend_type=self.args.blend_type,
                    blend_params=blend_params,
                )
                shaders.append(shader)
                output_mapping[
                    output_name
                ] = MultioutputMeshRenderer.ShaderChannelSelection(
                    shader_idx=len(shaders) - 1, channel_list=None
                )

            elif output_name in [
                "world_face_normals",
                "camera_face_normals",
                "ndc_face_normals",
            ]:
                blend_params = BlendParams(
                    sigma=self.args.blend_sigma,
                    gamma=self.args.blend_gamma,
                    background_color=self.args.background_color,
                )
                shader = MeshAttributeShader(
                    mesh_attribute="face_normal",
                    mesh_coords=output_name.split("_")[0],
                    blend_type=self.args.blend_type,
                    blend_params=blend_params,
                )
                shaders.append(shader)
                output_mapping[
                    output_name
                ] = MultioutputMeshRenderer.ShaderChannelSelection(
                    shader_idx=len(shaders) - 1, channel_list=None
                )

            elif output_name == "depth":
                blend_params = BlendParams(
                    sigma=self.args.blend_sigma,
                    gamma=self.args.blend_gamma,
                    background_color=self.args.background_color,
                )
                shader = DepthShader(
                    blend_type=self.args.blend_type,
                    blend_params=blend_params,
                )
                shaders.append(shader)
                output_mapping[
                    output_name
                ] = MultioutputMeshRenderer.ShaderChannelSelection(
                    shader_idx=len(shaders) - 1, channel_list=None
                )

        # create renderer
        self.renderer = MultioutputMeshRenderer(
            rasterizer=rasterizer, shaders=shaders, output_mapping=output_mapping
        )

        self.output_names = output_names

    def _get_uv_textures(self):
        textures = None
        if "uv_textures" in self._scene:
            scene_textures = self._scene["uv_textures"]

            if "meshes" not in self._scene:
                raise RuntimeError(
                    "Must provide meshes as well when updating uv textures."
                )

            meshes = self._scene["meshes"]

            if not isinstance(meshes, list) or not all(
                isinstance(mesh, Mesh) for mesh in meshes
            ):
                raise RuntimeError(
                    "Provided meshes must be given as list of geometry.mesh.Mesh"
                )

            if not isinstance(scene_textures, list) or len(scene_textures) != len(
                meshes
            ):
                raise RuntimeError("Need to specify one UV texture per mesh.")

            # get uv coordinates for all meshes
            vert_uvs = []
            face_uvs = []
            for mesh_idx in enumerate(meshes):
                mesh = meshes[mesh_idx]
                if "uv" not in mesh.vert_attributes:
                    raise RuntimeError(
                        "A uv texture was specified for a mesh that does not have a uv parameterization."
                    )
                vert_uvs.append(mesh.vert_attributes["uv"].verts)
                if mesh.vert_attributes["uv"].faces is not None:
                    face_uvs.append(mesh.vert_attributes["uv"].faces)
                else:
                    face_uvs.append(mesh.faces)

            textures = TexturesUV(
                verts_uvs=vert_uvs, faces_uvs=face_uvs, maps=scene_textures
            )

            return textures

    def render(self) -> Dict[str, torch.Tensor]:
        """
        Render the scene from all cameras.

        Returns:
            A dictionary of named output layers. Each layer has a tensor of shape B-H-W-C, where B is the number of cameras.
        """

        # update PyTorch3D meshes from the scene meshes
        vert_attributes = {}
        if "meshes" in self._scene:
            meshes = self._zero_meshes.offset_verts(
                torch.cat([mesh.verts for mesh in self._scene["meshes"]], dim=0)
            )

            num_verts = [mesh.verts.shape[0] for mesh in self._scene["meshes"]]
            
            # get packed vertex attributes (use zeros for meshes where a given attribute is missing)
            vert_attribute_names = sorted(list(set(itertools.chain(*(mesh.vert_attributes.keys() for mesh in self._scene["meshes"])))))
            for name in vert_attribute_names:

                attr_dims = set((mesh.vert_attributes[name].values.shape[-1] if name in mesh.vert_attributes else None) for mesh in self._scene["meshes"])
                attr_dims = attr_dims - {None}
                if len(attr_dims) != 1:
                    raise RuntimeError(f"Vertex attribute {name} has different dimensions across meshes.")
                attr_dim = next(iter(attr_dims))

                start_idx = 0
                vert_attributes[name] = torch.zeros(sum(num_verts), attr_dim, device=self.device, dtype=torch.float32)
                for mesh_idx, mesh in enumerate(self._scene["meshes"]):
                    if name in mesh.vert_attributes:
                        vert_attributes[name][start_idx:start_idx + num_verts[mesh_idx]] = mesh.vert_attributes[name].values
                    start_idx += num_verts[mesh_idx]

        # create PyTorch3D uv textures from scene uv textures
        uv_textures = self._get_uv_textures()
        if uv_textures is not None:
            meshes.texture = uv_textures

        # create PyTorch3D cameras from scene cameras
        if "cameras" in self._scene:
            cameras = self._scene["cameras"]
            # if not isinstance(cameras, ) # TODO: check cameras has the right class
            # (Valentin changed the camer class so leaving this for later when the new camera class is merged)
            # f = 0.5 * w / np.tan(0.5 * 55 * np.pi / 180.0)
            

            if isinstance(self.args.output_res, (list, tuple)):
                # potentially non-square output resolution given by (h, w) output_res
                # PyTorch3D expects the field of view in degrees of the vertical side length of the image,
                # whereas the intrinsics contain the fov of the maximum side length.
                fov = 2 * torch.rad2deg(torch.atan(1 / torch.stack([camera.intrinsics[1, 1] * (self.args.output_res[0] / max(self.args.output_res)) for camera in cameras])))
            else:
                # square output resolution
                fov = 2 * torch.rad2deg(torch.atan(1 / torch.stack([camera.intrinsics[1, 1] for camera in cameras])))

            # fov = 2 * torch.rad2deg(torch.atan(1 / (cameras[0].intrinsics[1, 1] / (0.5*512))))
            # cameras_pos is a tensor of shape [3, nbcamera] where the first dimension describes [distance to center, elevation, azimuth]
            R = torch.stack([camera.extrinsics_R for camera in cameras], dim=0)
            T = torch.stack([camera.extrinsics_t for camera in cameras], dim=0)

            # R, T = look_at_view_transform(
            #     dist=torch.stack([camera.radius for camera in cameras]),
            #     elev=90 - torch.stack([camera.theta for camera in cameras]),
            #     azim=torch.stack([camera.phi for camera in cameras]),
            #     up=((0, 1, 0),),
            # )
            # R = convention_rotation(R)
            cameras = FoVPerspectiveCameras(
                device=self.device, R=R, T=T, fov=fov, znear=self.args.z_near
            )

        # get global volume texture
        global_volume_texture = None
        if "global_volume_texture" in self._scene:
            global_volume_texture = self._scene["global_volume_texture"]

        # Duplicate cameras and meshes to have n_cameras x n_meshes cameras and meshes.
        # (There needs to be a separate mesh for each camera and a separate camera for each mesh.)
        # if len(meshes) != 1:
        #     # Extend cameras as c1,c2,c3, ... c1,c2,c3, ...
        #     cameras_extended = cameras[list(range(len(cameras))) * len(meshes)]
        # else:
        #     cameras_extended = cameras
        if len(cameras) != 1:
            # Extend meshes as m1,m1,m1, ... m2, m2, m2, ...
            meshes_extended = meshes.extend(N=len(cameras))
        else:
            meshes_extended = meshes

        # render scene
        outputs = self.renderer(
            meshes_world=meshes_extended,
            global_volume_texture=global_volume_texture,
            vert_attributes=vert_attributes,
            cameras=cameras, # cameras_extended,
        )

        return outputs
