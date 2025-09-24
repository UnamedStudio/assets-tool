import os
from pathlib import Path
from pxr.Usd import Prim, Attribute
from pxr.UsdGeom import XformCache, XformOp
from numpy import array
from numpy.typing import NDArray
from pxr.Gf import Matrix4d, Quatd, Quatf, Quath, Transform
from pxr.Sdf import Layer

from assets_tool import Relationship, SchemaRegistry, Stage, UsdPath

registry = SchemaRegistry()

def from_usd_quaternion(usd: Quatd) -> NDArray[float]:
    img = usd.GetImaginary()
    real = usd.GetReal()
    return array((img[0], img[1], img[2], real))  # type: ignore


def from_usd_transform(
    transform_mat: Matrix4d,
) -> tuple[NDArray[float], NDArray[float], NDArray[float]]:
    transform = Transform(transform_mat)
    translation = transform.GetTranslation()
    rotation = transform.GetRotation().GetQuat()
    scale = transform.GetScale()
    return (
        array(translation),
        from_usd_quaternion(rotation),
        array(scale),
    )


def from_usd_relative_transform(
    xform_cache: XformCache, prim: Prim, parent: Prim | None = None
) -> tuple[NDArray[float], NDArray[float], NDArray[float]]:
    transform = xform_cache.GetLocalToWorldTransform(prim)
    if parent is not None:
        parent_transform = xform_cache.GetLocalToWorldTransform(parent)
        transform = transform * parent_transform.GetInverse()
    return from_usd_transform(transform)


def quat(precising: XformOp.Precision) -> type:
    match precising:
        case XformOp.PrecisionDouble:
            return Quatd
        case XformOp.PrecisionFloat:
            return Quatf
        case XformOp.PrecisionHalf:
            return Quath
    raise Exception()


def is_attr_authored_in_layer(attr: Attribute, layer: Layer):
    for propSpec in attr.GetPropertyStack():
        if propSpec.layer == layer:
            return True

    return False


def is_prim_authored_in_layer(prim: Prim, layer: Layer) -> bool:
    prim_spec = layer.GetPrimAtPath(prim.GetPath())
    if prim_spec:
        return True
    else:
        return False


def is_attr_blocked(attr: Attribute):
    return attr.GetResolveInfo().ValueIsBlocked()


def unique_path(path: Path):
    stem = path.stem
    count = 0
    while True:
        if count == 0:
            unique_path = path
        else:
            unique_path = path.with_stem(f"{stem}{count}")
        if not unique_path.exists():
            return unique_path
        count += 1

def unique_usd_path(path: UsdPath, stage: Stage) -> UsdPath:
    parent = path.GetParentPath()
    name = path.name
    count = 0
    while True:
        if count == 0:
            unique_path = path
        else:
            unique_path = parent.AppendChild(f"{name}{count}")
        if not stage.GetPrimAtPath(unique_path):
            return unique_path
        count += 1


def copy_prim(
    stage: Stage, source_prim: Prim, dest_path: UsdPath, recursive: bool
) -> Prim:
    dst_prim = stage.DefinePrim(dest_path, source_prim.GetTypeName())

    for prop in source_prim.GetProperties():
        name = prop.GetName()
        if isinstance(prop, Attribute):
            if prop.HasAuthoredValue():
                dst_attr = dst_prim.CreateAttribute(name, typeName=prop.GetTypeName())
                dst_attr.Set(prop.Get())
                for k, v in prop.GetAllAuthoredMetadata().items():
                    dst_attr.SetMetadata(k, v)
        elif isinstance(prop, Relationship):
            if prop.HasAuthoredTargets():
                dst_attr = dst_prim.CreateRelationship(name)
                dst_attr.SetTargets(prop.GetTargets())
                for k, v in prop.GetAllAuthoredMetadata().items():
                    dst_attr.SetMetadata(k, v)

    for k, v in source_prim.GetAllAuthoredMetadata().items():
        dst_prim.SetMetadata(k, v)

    if recursive:
        for child in source_prim.GetChildren():
            child_dst_path = dest_path.AppendChild(child.GetName())
            copy_prim(stage, child, child_dst_path, recursive)
    return dst_prim

def copy_api(source: Prim, dest: Prim, api: str):
    prim_def = registry.FindAppliedAPIPrimDefinition(api)
    for name in prim_def.GetPropertyNames():
        prop = source.GetProperty(name)
        name = prop.GetName()
        if isinstance(prop, Attribute):
            if prop.HasAuthoredValue():
                dst_attr = dest.CreateAttribute(name, typeName=prop.GetTypeName())
                dst_attr.Set(prop.Get())
                for k, v in prop.GetAllAuthoredMetadata().items():
                    dst_attr.SetMetadata(k, v)
        elif isinstance(prop, Relationship):
            if prop.HasAuthoredTargets():
                dst_attr = dest.CreateRelationship(name)
                dst_attr.SetTargets(prop.GetTargets())
                for k, v in prop.GetAllAuthoredMetadata().items():
                    dst_attr.SetMetadata(k, v)


def relativize_sublayers(root: Path, layer: Layer):
    updated_paths = []

    for sublayer in layer.subLayerPaths:
        sublayer_path = Path(sublayer).resolve()
        relative = Path(os.path.relpath(sublayer_path, start=root.parent))
        updated_paths.append(relative.as_posix())

    layer.subLayerPaths.clear()
    for updated_path in updated_paths:
        layer.subLayerPaths.append(updated_path)


def none_str2none(value: str) -> str | None:
    if not value or value == "None":
        return None
    return value
