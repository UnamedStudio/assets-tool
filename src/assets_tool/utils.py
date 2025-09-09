from pathlib import Path
from pxr.Usd import Prim, Attribute
from pxr.UsdGeom import XformCache, XformOp
from numpy import array
from numpy.typing import NDArray
from pxr.Gf import Matrix4d, Quatd, Quatf, Quath, Transform
from pxr.Sdf import Layer


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
