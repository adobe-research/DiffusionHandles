from __future__ import annotations

import os
from typing import Optional
from pathlib import Path

import numpy as np
import torch
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
import trimesh
import trimesh.visual

from diffhandles import mesh
from diffhandles.mesh_io_obj import load_mesh_from_obj, save_mesh_to_obj


def load_mesh(filename: str):
    ext = os.path.splitext(filename)[1]

    if ext == ".obj":
        # trimesh can't load obj files with custom uv faces
        m, texture = load_mesh_from_obj(filename=filename)
    else:
        # Handle all other file types with trimesh
        m, texture = load_mesh_with_trimesh(filename=filename)

    return m, texture



# def load_mesh_with_igl(filename: str):
#     ext = os.path.splitext(filename)[1]

#     verts = (
#         uvs
#     ) = normals = colors = faces = uv_faces = normal_faces = color_faces = None
#     if ext == ".obj":
#         verts, uvs, normals, faces, uv_faces, normal_faces = igl.read_obj(
#             filename=filename, dtype="float32"
#         )
#     elif ext == ".off":
#         verts, faces, normals, uvs = igl.read_off(filename=filename, dtype="float32")

#     elif ext in [".stl", ".ply"]:
#         verts, faces = igl.read_triangle_mesh(filename=filename, dtypef="float32")
#     else:
#         raise RuntimeError(f"Unsupported mesh format: {ext}")
#     uvs, normals, colors, uv_faces, normal_faces, color_faces = (
#         None if x is None or x.shape[0] == 0 else x
#         for x in (uvs, normals, colors, uv_faces, normal_faces, color_faces)
#     )

#     m = mesh.Mesh(
#         verts=torch.tensor(verts, dtype=torch.float32),
#         faces=torch.tensor(faces, dtype=torch.int64),
#     )

#     if uvs is not None:
#         m.add_vert_attribute(
#             name="uv",
#             values=torch.tensor(uvs, dtype=torch.float32),
#             faces=torch.tensor(uv_faces, dtype=torch.int64)
#             if uv_faces is not None
#             else None,
#         )

#     if normals is not None:
#         m.add_vert_attribute(
#             name="normal",
#             values=torch.tensor(normals, dtype=torch.float32),
#             faces=torch.tensor(normal_faces, dtype=torch.int64)
#             if normal_faces is not None
#             else None,
#         )

#     # IGL does not support loading textures
#     texture = None

#     return m, texture


def load_mesh_with_trimesh(filename: str):
    tmesh = trimesh.load_mesh(filename)

    # convert scene to single mesh if a scene was returned
    if isinstance(tmesh, trimesh.Scene):
        tmesh = trimesh.util.concatenate(
            tuple(
                trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                for g in tmesh.geometry.values()
            )
        )

    m = mesh.Mesh(
        verts=torch.tensor(tmesh.vertices, dtype=torch.float32),
        faces=torch.tensor(tmesh.faces, dtype=torch.int64),
    )
    texture = None

    # load vertex colors
    if tmesh.visual is not None and isinstance(
        tmesh.visual, trimesh.visual.ColorVisuals
    ):
        m.add_vert_attribute(
            name="color",
            values=torch.tensor(tmesh.visual.vertex_colors, dtype=torch.float32),
        )

    # load uv coordinates
    if (
        tmesh.visual is not None
        and isinstance(tmesh.visual, trimesh.visual.TextureVisuals)
        and tmesh.visual.uv is not None
    ):
        m.add_vert_attribute(
            name="uv", values=torch.tensor(tmesh.visual.uv, dtype=torch.float32)
        )
        if (
            tmesh.visual.material is not None
            and tmesh.visual.material.image is not None
        ):
            texture = tmesh.visual.material.image
            texture = (
                pil_to_tensor(texture.convert("RGB"))
                .to(dtype=torch.float32)
                .permute(1, 2, 0)
                / 255.0
            )  # C-H-W -> H-W-C

    # load vertex normals
    if tmesh.vertex_normals is not None:
        m.add_vert_attribute(
            name="normal",
            values=torch.tensor(tmesh.vertex_normals, dtype=torch.float32),
        )

    # load face normals
    if tmesh.face_normals is not None:
        m.add_face_attribute(
            name="normal", values=torch.tensor(tmesh.face_normals, dtype=torch.float32)
        )

    return m, texture


def save_mesh(mesh: mesh.Mesh, filename: str, texture: Optional[torch.Tensor] = None):
    ext = os.path.splitext(filename)[1]

    if ext == ".obj":
        # I could not find a library that support saving OBJ files with custom uv faces,
        # use a method copied from SSMesh
        save_mesh_to_obj(mesh=mesh, filename=filename, texture=texture)
    else:
        save_mesh_with_trimesh(mesh=mesh, filename=filename, texture=texture)


def save_mesh_with_trimesh(
    mesh: mesh.Mesh, filename: str, texture: Optional[torch.Tensor] = None
):
    if "color" in mesh.vert_attributes and "uv" in mesh.vert_attributes:
        print(
            "WARNING: saving both vertex colors and uvs is currently not supported, only UVs will be saved."
        )

    tmesh = trimesh.Trimesh(
        vertices=mesh.verts.detach().cpu().numpy(),
        faces=mesh.faces.detach().cpu().numpy(),
        process=False, validate=False # to ensure trimesh does not alter the mesh in any way
    )

    # save vertex colors
    if "color" in mesh.vert_attributes:
        vcolor_attribute = mesh.vert_attributes["color"]
        if vcolor_attribute.faces is not None:
            vcolor_attribute = mesh.remove_custom_faces(vcolor_attribute)
        tmesh.visual = trimesh.visual.ColorVisuals(
            mesh=tmesh, vertex_colors=vcolor_attribute.values.detach().cpu().numpy()
        )

    # save uv coordinates
    if "uv" in mesh.vert_attributes:
        uv_attribute = mesh.vert_attributes["uv"]
        if uv_attribute.faces is not None:
            uv_attribute = mesh.remove_custom_faces(uv_attribute)
        if texture is not None:
            texture = (
                (torch.clamp(texture, min=0, max=1) * 255.0)
                .permute(2, 0, 1)
                .to(dtype=torch.uint8)
            )  # H-W-C -> C-H-W
            material = trimesh.visual.material.SimpleMaterial(
                image=to_pil_image(texture)
            )
        else:
            material = None
        tmesh.visual = trimesh.visual.TextureVisuals(
            uv=uv_attribute.values.detach().cpu().numpy(), material=material
        )

    # save vertex normals
    if "normal" in mesh.vert_attributes:
        vnormal_attribute = mesh.vert_attributes["normal"]
        if vnormal_attribute.faces is not None:
            vcolor_attribute = mesh.remove_custom_faces(vcolor_attribute)
            # re-normalize normals, since they may have been averaged
            vcolor_attribute.values = torch.nn.functional.normalize(
                vcolor_attribute.values, p=2, dim=-1
            )
        tmesh.vertex_normals = vcolor_attribute.values.detach().cpu().numpy()

    # save face normals
    if "normal" in mesh.face_attributes:
        fnormal_attribute = mesh.face_attributes["normal"]
        tmesh.face_normals = fnormal_attribute.values.detach().cpu().numpy()

    tmesh.export(filename)
