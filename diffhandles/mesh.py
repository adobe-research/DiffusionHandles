from __future__ import annotations

from typing import Optional, Union

import torch

# from . import mesh_io


class VertexAttribute(torch.nn.Module):
    """
    A custom attribute for each vertex of a mesh. May optionally include custom per-face indices into the list of attributes.
    """

    def __init__(
        self,
        values: torch.FloatTensor,
        faces: Optional[torch.LongTensor] = None,
        device: Union[str, torch.device] = "cpu",
    ):
        """
        Args:
            values: NV-D tensor, where NV is the number of attribute values and D is the attribute dimension.
            faces: NF-NP tensor, where NF is the number of faces and NP the number of vertices in a face (e.g. 3 for a triangle).
                This optionally specifies custom per-face indices into the list of attributes.
        """
        super().__init__()

        if faces is not None and faces.max() >= values.shape[0]:
            raise RuntimeError("Face index out of bounds.")

        self.device = torch.device(device)
        self.values = torch.nn.Parameter(values.detach().clone().to(self.device))
        self.register_buffer("faces", faces)


class FaceAttribute(torch.nn.Module):
    """
    A custom attribute for each face of a mesh.
    """

    def __init__(
        self, values: torch.FloatTensor, device: Union[str, torch.device] = "cpu"
    ):
        """
        Args:
            values: NF-D tensor, where NF is the number of faces and D is the attribute dimension.
        """
        super().__init__()

        self.device = torch.device(device)
        self.values = torch.nn.Parameter(values.detach().clone().to(self.device))


class Mesh(torch.nn.Module):
    """
    A general mesh.
    """

    def __init__(
        self,
        verts: torch.FloatTensor,
        faces: torch.LongTensor,
        device: Union[str, torch.device] = None,
    ):
        """
        Args:
            verts: NV-D tensor, where NV is the number of vertices and D is the vertex dimension.
            faces: NF-NP tensor, where NF is the number of faces and NP the number of vertices
                in a face (e.g. 3 for a triangle).
        """
        super().__init__()

        if device is None:
            self.device = verts.device
        else:
            self.device = torch.device(device)
        self.verts = torch.nn.Parameter(verts.to(self.device))
        self.register_buffer("faces", faces)
        self.vert_attributes = torch.nn.ModuleDict()
        self.face_attributes = torch.nn.ModuleDict()

        if faces.max() >= verts.shape[0]:
            raise RuntimeError("Face index out of bounds.")

    def add_vert_attribute(
        self,
        name: str,
        values: torch.FloatTensor,
        faces: Optional[torch.LongTensor] = None,
    ):
        """
        Add a named per-vertex attribute.

        Args:
            name: Name of the vertex attribute. Names are unique, if the name already exists,
                the old vertex attribute will be overwritten.
            values: NV-D tensor, where NV is the number of attribute values and D is the dimension of each value.
            faces: NF-NP tensor, where NF is the number of faces and NP the number of vertices in a face (e.g. 3 for a triangle).
                This optionally specifies custom per-face indices into the list of attribute values.
        """
        vert_attribute = VertexAttribute(values=values, faces=faces, device=self.device)

        if vert_attribute.faces is None:
            if values.shape[0] != self.verts.shape[0]:
                raise RuntimeError(
                    f"Values for vertex attribute {name} don't match existing number of vertices."
                )

        else:
            if vert_attribute.faces.shape != self.faces.shape:
                raise RuntimeError(
                    f"Faces for vertex attribute {name} don't match existing faces."
                )

        self.vert_attributes[name] = vert_attribute

    def remove_vert_attribute(self, name: str):
        """
        Remove a named per-vertex attribute.
        """
        self.vert_attributes.pop(key=name)

    def add_face_attribute(self, name: str, values: torch.FloatTensor):
        """
        Add a named per-face attribute.

        Args:
            name: Name of the face attribute. Names are unique, if the name already exists,
                the old face attribute will be overwritten.
            values: NV-D tensor, where NV is the number of attribute values and D is the attribute dimension.
        """
        face_attribute = FaceAttribute(values=values, device=self.device)
        if face_attribute.values.shape[0] != self.faces.shape[0]:
            raise RuntimeError(
                f"Values for face attribute {name} don't match existing number of faces."
            )
        self.face_attributes[name] = face_attribute

    def remove_face_attribute(self, name: str):
        """
        Remove a named per-face attribute.
        """
        self.face_attributes.pop(key=name)

    def remove_custom_faces(
        self, vert_attribute: VertexAttribute, reduce: str = "mean"
    ):
        """
        Remove custom faces from a vertex attribute.
        The vertex attribute will use the standard mesh faces instead.
        In some cases, this may cause two separate vertex attributes
        to be averaged into a single attribute if the standard mesh
        faces re-use the same vertex for these two attributes.
        (For example, when vertex colors are different for the same vertex in two adjacent faces,
        but vertex positions are re-used in the standard mesh faces, the vertex colors will be averaged.)

        Args:
            vert_attribute: Custom faces will be removed from this vertex attribute.
            reduce: A different operation than the mean can be used to merge vertex attributes
                ("sum", "prod", "mean", "amax", "amin").
        """
        # values = torch.zeros(
        #     size=[self.verts.shape[0], vert_attribute.values.shape[1]],
        #     dtype=vert_attribute.values.dtype,
        #     device=vert_attribute.values.device,
        # )
        values = torch.scatter_reduce(
            dim=0,
            index=self.faces.view(-1),
            src=vert_attribute.values[vert_attribute.faces.view(-1)],
            reduce=reduce,
            include_self=False,
        )
        return VertexAttribute(values=values, device=self.device)

    # @staticmethod
    # def load(filename: str):
    #     """
    #     Load mesh from file.

    #     Args:
    #         filename: File name of the file to load from. Extension is used to determine format.

    #     Returns:
    #         A tuple containing the mesh and the mesh texture, which may be None if the mesh does not have a texture.
    #     """
    #     with torch.no_grad():
    #         return mesh_io.load_mesh(filename=filename)

    # def save(self, filename: str, texture: Optional[torch.Tensor] = None):
    #     """
    #     Save mesh to file.

    #     Args:
    #         filename: File name of the file to save to. Extension is used to determine format.
    #         texture: W-H-C tensor. The mesh texture, is only saved if the mesh has uv coordinates.
    #     """
    #     with torch.no_grad():
    #         mesh_io.save_mesh(mesh=self, filename=filename, texture=texture)

    def normalize(self, size=1.0):
        """
        Normalize the mesh bounding cube to be centered at zero and have the given target size.

        Args:
            size: Target side length of the mesh bounding cube.
        """
        bc_min, bc_max = self.bounding_cube()
        bc_center = 0.5 * (bc_min + bc_max)
        bc_size = (bc_max - bc_min).max()
        self.verts -= bc_center
        self.verts *= (size / bc_size) 

    def bounding_cube(self):
        """
        Get axis-aligned bounding cube of the mesh.

        Returns:
            Tuple of tensors (bc_min, bc_max), containing the coordinates
            of the minimum and maximum bounding cube corners.
        """
        bb_min, bb_max = self.bounding_box()
        bc_size = (bb_max - bb_min).max()
        bc_center = 0.5 * (bb_min + bb_max)
        bc_max = bc_center + 0.5 * bc_size
        bc_min = bc_center - 0.5 * bc_size
        return bc_min, bc_max

    def bounding_box(self):
        """
        Get axis-aligned bounding box of the mesh.

        Returns:
            Tuple of tensors (bb_min, bb_max), containing the coordinates
            of the minimum and maximum bounding box corners.
        """
        bb_min = self.verts.min(dim=0)[0]
        bb_max = self.verts.max(dim=0)[0]
        return bb_min, bb_max
