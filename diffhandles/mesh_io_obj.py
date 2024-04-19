from __future__ import annotations

import os
from typing import Optional
from pathlib import Path

from PIL import Image
import numpy as np
import torch
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

from diffhandles import mesh


def load_mesh_from_obj(
    filename: str,
    skip_normals: bool = False,
    skip_textures: bool = False,
    skip_groups: bool = False,
    skip_mtllib: bool = False,
):
    filename = Path(filename)

    if not os.path.isfile(filename.as_posix()):
        print("ERROR Cannot read " + filename.as_posix())
        assert False

    row_prev = ""
    write_err_once = set()
    vt_count = 0
    vn_count = 0
    v_count = 0
    f_count = 0
    #
    obj_content = []
    obj_type = []
    f_count2 = 0
    #
    group_names = None
    object_names = None
    material_names = None
    face_attributes = None  # group, object, material, smoothness
    #
    mtllib_filename = None
    #
    with open(filename, "r", errors="ignore") as f:
        rows = f.read().splitlines(keepends=True)
        
    for row in rows:
        row = row_prev + row
        if row.endswith("\\\n"):
            row_prev = row.replace("\\\n", " ")
            continue
        row_prev = ""
        add_me = True
        if row[0] == "v":
            if row[1] == "t":
                if skip_textures:
                    add_me = False
                else:
                    vt_count += 1
                    obj_type.append(0)
            elif row[1] == "n":
                if skip_normals:
                    add_me = False
                else:
                    vn_count += 1
                    obj_type.append(1)
            else:
                v_count += 1
                obj_type.append(2)
        elif row[0] == "f" or row[0] == "p":
            f_count2 += 1
            if row[0] == "f":  # f
                corners_str = row[2:].strip().split()
            elif row[0] == "p":  # p
                corners_str = row[3:].strip().split()
            else:
                assert False
            if len(corners_str) < 3:
                if "f4+" not in write_err_once:
                    write_err_once.add("f4+")
                    print(
                        "[WARNING] Encountered a face with fewer than 3 vertices. Skipping..."
                    )
                add_me = False
            else:
                row = corners_str
                f_count += len(corners_str) - 2
                obj_type.append(3)
        elif not skip_groups or not skip_mtllib:
            if not skip_groups:
                if row[0] == "g":
                    obj_type.append(10)
                elif row[0] == "o":
                    obj_type.append(11)
                elif row.startswith("usemtl "):
                    obj_type.append(12)
                elif row[0] == "s":
                    obj_type.append(13)
                else:
                    add_me = False

            if not add_me and not skip_mtllib:
                if row.startswith("mtllib "):
                    mtllib_filename = Path(
                        os.path.dirname(filename), row[len("mtllib ") :].strip()
                    )
        else:
            add_me = False
        if add_me:
            obj_content.append(row)
    # f.close()

    texture_filename = None
    if mtllib_filename is not None:
        if os.path.exists(mtllib_filename):
            with open(mtllib_filename, "r") as f:
                mttlib_content = f.read().splitlines()
            for row in mttlib_content:
                if row.startswith("map_Kd "):
                    texture_filename = Path(
                        os.path.dirname(mtllib_filename), row[len("map_Kd ") :]
                    )
        else:
            print(f"WARNING: could not find material library {mtllib_filename}.")

    valid_fn = vn_count > 0
    valid_ft = vt_count > 0
    #
    vertices = np.zeros((v_count, 3), dtype=np.float32)
    if not skip_textures and valid_ft:
        uv_coords = np.zeros((vt_count, 2), dtype=np.float32)
        uv_faces = np.empty((f_count, 3), dtype=np.int32)
    else:
        uv_coords = None
        uv_faces = None

    if not skip_normals and valid_fn:
        normals = np.zeros((vn_count, 3), dtype=np.float32)
        normal_faces = np.empty((f_count, 3), dtype=np.int32)
    else:
        normals = None
        normal_faces = None
    faces = np.empty((f_count, 3), dtype=np.int32)
    #
    if not skip_groups:
        group_names = []
        object_names = []
        material_names = []
        face_attributes = np.empty(
            (f_count, 4), dtype=np.int32
        )  # group, object, material, smoothness
    #
    vt_count = 0
    vn_count = 0
    v_count = 0
    f_count = 0
    att_tuple = [-1, -1, -1, -1]
    assert len(obj_content) == len(obj_type)
    for r in range(0, len(obj_content)):
        if obj_type[r] == 0:  # vt
            if valid_ft:
                uvs = obj_content[r].strip().split()
                assert len(uvs) >= 3
                try:
                    if uvs[1] == "nan" or uvs[2] == "nan":
                        uvs[1] = "error"
                    uv_coords[vt_count, 0] = float(uvs[1])
                    uv_coords[vt_count, 1] = float(uvs[2])
                except:
                    if "tc" not in write_err_once:
                        write_err_once.add("tc")
                        print(
                            "[WARNING] Invalid Texture Coordinates: "
                            + obj_content[r]
                            + "\n      Replacing with [0,0]"
                        )
                vt_count += 1
        elif obj_type[r] == 1:  # vn
            if valid_fn:
                norm = obj_content[r].strip().split()
                try:
                    normals[vn_count, 0] = float(norm[1])
                    normals[vn_count, 1] = float(norm[2])
                    normals[vn_count, 2] = float(norm[3])
                except:
                    if "nc" not in write_err_once:
                        write_err_once.add("nc")
                        print(
                            "[WARNING] Invalid Normal Coordinates: "
                            + obj_content[r]
                            + "\n      Replacing with [0,0, 0]"
                        )
                vn_count += 1
        elif obj_type[r] == 2:  # v
            coords = obj_content[r].strip().split()
            try:
                vertices[v_count, 0] = float(coords[1])
                vertices[v_count, 1] = float(coords[2])
                vertices[v_count, 2] = float(coords[3])
            except:
                if "vc" not in write_err_once:
                    write_err_once.add("nc")
                    print(
                        "[WARNING] Invalid Vertex Coordinates: "
                        + obj_content[r]
                        + "\n      Replacing with [0, 0, 0]"
                    )
            v_count += 1
        elif obj_type[r] == 3 or obj_type[r] == 4:  # f or p
            corners_str = obj_content[r]
            first_corner = None
            prev_corner = None
            for c in range(0, len(corners_str)):
                corner_list = corners_str[c].split("/")
                coord = None
                texture = None
                normal = None
                if len(corner_list) >= 1:
                    coord = int(corner_list[0])
                    if coord > 0:
                        coord = coord - 1
                    elif coord < 0:
                        coord = v_count + coord
                    else:
                        assert coord != 0
                    assert coord >= 0
                if (
                    len(corner_list) >= 2
                    and corner_list[1] != ""
                    and valid_ft
                    and not skip_textures
                ):
                    texture = int(corner_list[1])
                    if texture > 0:
                        texture = texture - 1
                    elif texture < 0:
                        texture = vt_count + texture
                    else:
                        assert texture != 0
                    assert texture >= 0
                if (
                    len(corner_list) >= 3
                    and corner_list[2] != ""
                    and valid_fn
                    and not skip_normals
                ):
                    normal = int(corner_list[2])
                    if normal > 0:
                        normal = normal - 1
                    elif normal < 0:
                        normal = vn_count + normal
                    else:
                        assert normal != 0
                    assert normal >= 0
                    if normal > len(normals):
                        normal = None
                assert coord != None
                #
                if valid_ft and texture == None:  # erase all uvs
                    valid_ft = False
                    uv_faces = None
                    uv_coords = None
                #
                if valid_fn and normal == None:  # erase all normals
                    valid_fn = False
                    normal_faces = None
                    normals = None
                #
                if first_corner is not None and prev_corner is not None:
                    faces[f_count, 0] = first_corner[0]
                    faces[f_count, 1] = prev_corner[0]
                    faces[f_count, 2] = coord
                    #
                    if not skip_groups:
                        for ai in range(0, 4):
                            face_attributes[f_count, ai] = att_tuple[ai]
                    #
                    if valid_ft:
                        uv_faces[f_count, 0] = first_corner[1]
                        uv_faces[f_count, 1] = prev_corner[1]
                        uv_faces[f_count, 2] = texture
                    #
                    if valid_fn:
                        normal_faces[f_count, 0] = first_corner[2]
                        normal_faces[f_count, 1] = prev_corner[2]
                        normal_faces[f_count, 2] = normal
                    #
                    f_count += 1
                if c == 0:
                    first_corner = (coord, texture, normal)
                else:
                    prev_corner = (coord, texture, normal)
        elif obj_type[r] == 10:  # g
            g_list = obj_content[r].strip().split()
            if len(g_list) > 1:
                gname = g_list[1]
            else:
                gname = "ssmesh_group_" + str(len(group_names))
            if gname in group_names:
                gid = group_names.index(gname)
            else:
                gid = len(group_names)
                group_names.append(gname)
            att_tuple[0] = gid
        elif obj_type[r] == 11:  # o
            o_list = obj_content[r].strip().split()
            if len(o_list) > 1:
                oname = o_list[1]
            else:
                oname = "ssmesh_obj_" + str(len(object_names))
            if oname in object_names:
                oid = object_names.index(oname)
            else:
                oid = len(object_names)
                object_names.append(oname)
            att_tuple[1] = oid
        elif obj_type[r] == 12:  # usemtl
            m_list = obj_content[r].strip().split()
            if len(m_list) > 1:
                mname = m_list[1]
            else:
                mname = "ssmesh_mtl_" + str(len(material_names))
            if mname in material_names:
                mid = material_names.index(mname)
            else:
                mid = len(material_names)
                material_names.append(mname)
            att_tuple[2] = mid
        elif obj_type[r] == 13:  # s
            s_list = obj_content[r].strip().split()
            if len(s_list) > 1:
                smoothness = s_list[1]
            else:
                smoothness = "off"
            if smoothness == "o" or smoothness == "off":
                att_tuple[3] = 0
            else:
                try:
                    att_tuple[3] = int(smoothness)
                except:
                    att_tuple[3] = 0
        else:
            assert False

    m = mesh.Mesh(
        verts=torch.tensor(vertices, dtype=torch.float32),
        faces=torch.tensor(faces, dtype=torch.int64),
    )

    if uv_coords is not None:
        m.add_vert_attribute(
            name="uv",
            values=torch.tensor(uv_coords, dtype=torch.float32),
            faces=torch.tensor(uv_faces, dtype=torch.int64)
            if uv_faces is not None
            else None,
        )

    if normals is not None:
        m.add_vert_attribute(
            name="normal",
            values=torch.tensor(normal, dtype=torch.float32),
            faces=torch.tensor(normal_faces, dtype=torch.int64)
            if normal_faces is not None
            else None,
        )

    texture = None
    if texture_filename is not None:
        if os.path.exists(texture_filename):
            with Image.open(str(texture_filename)) as im:
                texture = (
                    pil_to_tensor(im.convert("RGB"))
                    .to(dtype=torch.float32)
                    .permute(1, 2, 0)
                    / 255.0
                )  # C-H-W -> H-W-C
        else:
            print(f"WARNING: could not find texture {texture_filename}.")

    # if material_names is not None and len(material_names) > 0:
    #     material_names[0]

    return m, texture

    # return ShapeSenseMesh(
    #     vertices=vertices,
    #     faces=faces,
    #     uv_coords=uv_coords,
    #     uv_faces=uv_faces,
    #     normals=normals,
    #     normal_faces=normal_faces,
    #     vertex_colors=None,
    #     group_names=group_names,
    #     object_names=object_names,
    #     material_names=material_names,
    #     face_attributes=face_attributes,
    #     sourcefile=filename,
    # )


def save_mesh_to_obj(
    mesh: mesh.Mesh, filename: str, texture: Optional[torch.Tensor] = None
):
    dirname = os.path.dirname(filename)
    if dirname != "" and not os.path.isdir(dirname):
        os.makedirs(dirname)

    # Write material (.mtl) file
    if texture is not None:
        texture = (
            (torch.clamp(texture, min=0, max=1) * 255.0)
            .permute(2, 0, 1)
            .to(dtype=torch.uint8)
        )  # H-W-C -> C-H-W
        im = to_pil_image(texture)

        material_png_path = filename.replace(".obj", "material.png")
        im.save(material_png_path)
        material_name = Path(material_png_path).parts[-1][:-4]
        with open(filename.replace(".obj", "material.mtl"), "w") as f_out:
            f_out.write(f"newmtl {material_name}" + "\n")
            f_out.write("Ka 0.00000000 0.00000000 0.00000000" + "\n")
            f_out.write("Kd 1.00000000 1.00000000 1.00000000" + "\n")
            f_out.write("Ks 0.00000000 0.00000000 0.00000000" + "\n")
            f_out.write(f"map_Kd {material_name}.png" + "\n")

    # Write mesh (.obj) file
    verts = mesh.verts.detach().cpu().numpy()
    faces = mesh.faces.detach().cpu().numpy()
    with open(filename, "w") as f_out:
        if texture is not None:
            f_out.write(f"mtllib {material_name}.mtl" + "\n")
            f_out.write(f"usemtl {material_name}" + "\n")

        for v in range(verts.shape[0]):
            f_out.write(f"v {verts[v, 0]} {verts[v, 1]} {verts[v, 2]}\n")

        uv_faces = None
        if "uv" in mesh.vert_attributes:
            uv_attribute = mesh.vert_attributes["uv"]
            uvs = uv_attribute.values.detach().cpu().numpy()
            for vt in range(uv_attribute.values.shape[0]):
                f_out.write(
                    f"vt "
                    f"{uvs[vt, 0]} "
                    f"{uvs[vt, 1]} \n"
                )
            uv_faces = uv_attribute.faces.detach().cpu().numpy()
            if uv_faces is None:
                uv_faces = faces

        normal_faces = None
        if "normal" in mesh.vert_attributes:
            normal_attribute = mesh.vert_attributes["normal"]
            normals = normal_attribute.values.detach().cpu().numpy()
            for vn in range(normal_attribute.values.shape[0]):
                f_out.write(
                    f"vn "
                    f"{normals[vn, 0]} "
                    f"{normals[vn, 1]} "
                    f"{normals[vn, 2]}\n"
                )
            normal_faces = normal_attribute.faces.detach().cpu().numpy()
            if normal_faces is None:
                normal_faces = faces

        for f in range(faces.shape[0]):
            f_out.write("f ")
            for d in range(faces.shape[1]):
                f_out.write(f"{faces[f, d] + 1}")

                if uv_faces is not None:
                    f_out.write(f"/{uv_faces[f, d] + 1}")
                elif normal_faces is not None:
                    f_out.write("/")
                if normal_faces is not None:
                    f_out.write(f"/{normal_faces[f, d] + 1}")
                if d < (faces.shape[1] - 1):
                    f_out.write(" ")
                else:
                    f_out.write("\n")
    return
