from collections.abc import Callable, Iterable
from copy import deepcopy
from gc import enable
import json
from math import e
from multiprocessing.shared_memory import SharedMemory
from os import rename
import os
from pathlib import Path
from queue import Queue
from typing import Any
import dearpygui.dearpygui as dpg

from numpy import array, float32, int32
import numpy
from numpy.typing import NDArray
from pxr.Usd import (
    Stage,
    Prim,
    SchemaRegistry,
    Attribute,
    Relationship,
    PrimRange,
)
from pxr.Sdf import Layer, Path as UsdPath
from pxr.Gf import Vec3d, Quatd
from pxr.UsdGeom import Mesh, Xformable, PrimvarsAPI, Primvar
from pxr import Vt
from pxr.Tf import Type as UsdType

import software_client
from software_client import Client, RunCommands

from assets_tool.utils import (
    XformCache,
    copy_prim,
    from_usd_relative_transform,
    from_usd_transform,
    is_attr_authored_in_layer,
    is_prim_authored_in_layer,
    quat,
    is_attr_blocked,
    relativize_sublayers,
    unique_path,
)

registry = SchemaRegistry()


class Tree:
    class Node:
        is_open: bool
        open_button: int | str
        children_ui: int | str
        root_ui: int | str

    def __init__(self, open_label: str = "+", close_label="-") -> None:
        self.open_label = open_label
        self.close_label = close_label

    def node(self, label: str, callback: Callable[[], None], parent: int | str) -> Node:
        node = self.Node()
        node.is_open = False
        with dpg.group(parent=parent) as root_ui:
            with dpg.group(horizontal=True):
                node.open_button = dpg.add_button(
                    label=self.open_label, callback=self.toggle_open(node)
                )
                dpg.add_button(label=label, callback=callback)
            node.children_ui = dpg.add_child_window(
                parent=parent, show=False, auto_resize_y=True
            )
        node.root_ui = root_ui
        return node

    def toggle_open(self, node: Node) -> Callable:
        def ret():
            if node.is_open:
                node.is_open = False
                dpg.configure_item(node.children_ui, show=False)
                dpg.configure_item(node.open_button, label=self.open_label)
            else:
                node.is_open = True
                dpg.configure_item(node.children_ui, show=True)
                dpg.configure_item(node.open_button, label=self.close_label)

        return ret


class FileExplorer:
    def __init__(
        self,
        container: int | str,
        load_file_callbacks: list[Callable[[Path], None]],
    ) -> None:
        self.container = container
        self.on_load_file = load_file_callbacks
        self.stage: Stage | None = None
        self.xform_cache: XformCache | None = None
        self.old_metadata: dict | None = None

        self.opened_file_path: Path | None = None
        self.opened_file_ui = dpg.add_text(parent=self.container)

        self.edit_mode_ui = dpg.add_combo(
            ["update", "override", "override replace"],
            label="edit mode",
            default_value="override replace",
            callback=self.update_operation_name_ui,
            parent=self.container,
        )
        self.operation_name_ui = dpg.add_input_text(
            label="operation name",
            default_value="none",
            parent=self.container,
        )
        dpg.add_button(label="save", callback=self.save, parent=self.container)
        self.operation_stack_ui = dpg.add_tree_node(
            label="operation stack", parent=self.container
        )
        self.tree_ui = dpg.add_child_window(parent=self.container)

    def save(self):
        if self.stage:
            assert self.opened_file_path
            stem = self.opened_file_path.stem
            operation_name = dpg.get_value(self.operation_name_ui)
            root_layer = self.stage.GetRootLayer()
            reload = False
            if Layer.IsAnonymousLayerIdentifier(root_layer.identifier):
                edit_mode = dpg.get_value(self.edit_mode_ui)
                match edit_mode:
                    case "override":
                        path = self.opened_file_path.with_name(
                            f"{stem}+{operation_name}.usda"
                        )
                        if path.exists():
                            path.unlink(missing_ok=True)
                        root_layer.identifier = str(path)
                        relativize_sublayers(root_layer)
                        reload = True
                    case "override replace":
                        if self.old_metadata:
                            old_metadata = self.old_metadata
                        else:
                            old_metadata = {
                                "operation": "",
                                "input": "",
                                "output": "",
                                "original_name": stem,
                            }
                        original_name = old_metadata["original_name"]
                        path = self.opened_file_path.with_stem(
                            f"{original_name}-{operation_name}"
                        )
                        old_stage = Stage.Open(str(self.opened_file_path))
                        old_root_layer = old_stage.GetRootLayer()
                        custom_layer_data = old_root_layer.customLayerData
                        metadata = deepcopy(old_metadata)
                        metadata["output"] = Path(
                            os.path.relpath(
                                str(self.opened_file_path.resolve()),
                                str(path.parent.resolve()),
                            )
                        ).as_posix()
                        custom_layer_data["assets_tool:operation"] = metadata
                        old_root_layer.customLayerData = custom_layer_data
                        old_stage.GetRootLayer().Save()
                        del old_stage

                        custom_layer_data = root_layer.customLayerData
                        metadata = deepcopy(old_metadata)
                        metadata["input"] = os.path.relpath(
                            str(path.resolve()),
                            str(self.opened_file_path.parent.resolve()),
                        )
                        metadata["operation"] = operation_name
                        custom_layer_data["assets_tool:operation"] = metadata
                        root_layer.customLayerData = custom_layer_data
                        root_layer.subLayerPaths.clear()

                        if path.exists():
                            path.unlink(missing_ok=True)
                        self.opened_file_path.rename(path)
                        root_layer.subLayerPaths.append(str(path))
                        root_layer.identifier = str(self.opened_file_path)
                        relativize_sublayers(root_layer)
                        reload = True
            self.stage.Save()
            if reload:
                self.load_path(self.opened_file_path.parent)()
                self.update_operation_stack_ui()

    def update_operation_name_ui(self):
        match dpg.get_value(self.edit_mode_ui):
            case "update":
                show = False
            case "override" | "override replace":
                show = True
            case _:
                raise Exception()
        dpg.configure_item(self.operation_name_ui, show=show)

    def update_operation_stack_ui(self):
        dpg.delete_item(self.operation_stack_ui, children_only=True)
        assert self.opened_file_path
        stage = Stage.Open(str(self.opened_file_path))
        metadata = stage.GetRootLayer().customLayerData.get("assets_tool:operation")
        if not metadata:
            return
        inputs = []
        metadata_iter = metadata
        while True:
            if input := metadata_iter["input"]:
                path = self.opened_file_path.parent / Path(input)
                stage = Stage.Open(str(path))
                metadata_iter = stage.GetRootLayer().customLayerData[
                    "assets_tool:operation"
                ]
                inputs.append((path, metadata_iter))
            else:
                break

        outputs = []
        metadata_iter = metadata
        while True:
            if input := metadata_iter["output"]:
                path = self.opened_file_path.parent / Path(input)
                stage = Stage.Open(str(path))
                metadata_iter = stage.GetRootLayer().customLayerData[
                    "assets_tool:operation"
                ]
                outputs.append((path, metadata_iter))
            else:
                break

        for path, metadata_iter in reversed(inputs):
            operation_name = metadata_iter["operation"]
            if not operation_name:
                operation_name = "__origin__"
            with dpg.group(horizontal=True, parent=self.operation_stack_ui):
                dpg.add_text(" ")
                dpg.add_button(
                    label=operation_name,
                    callback=self.load_path(path),
                )

        operation_name = metadata["operation"]
        if not operation_name:
            operation_name = "__origin__"
        with dpg.group(horizontal=True, parent=self.operation_stack_ui):
            dpg.add_text(">")
            dpg.add_button(label=operation_name)

        for path, metadata_iter in outputs:
            operation_name = metadata_iter["operation"]
            with dpg.group(horizontal=True, parent=self.operation_stack_ui):
                dpg.add_text(" ")
                dpg.add_button(label=operation_name, callback=self.load_path(path))

    def reload_file(self):
        if self.opened_file_path:
            self.load_path(self.opened_file_path)

    def load_path(
        self,
        path: Path,
    ):
        def ret():
            if path.is_dir():
                dpg.delete_item(self.tree_ui, children_only=True)
                dpg.add_button(
                    label="..",
                    callback=self.load_path(path.parent),
                    parent=self.tree_ui,
                )
                for child in sorted(path.iterdir()):
                    dpg.add_button(
                        label=child.name,
                        callback=self.load_path(child),
                        parent=self.tree_ui,
                    )
            elif path.is_file():
                self.opened_file_path = path
                dpg.set_value(self.opened_file_ui, str(path))
                self.stage = None
                self.old_metadata = None
                dpg.delete_item(self.operation_stack_ui, children_only=True)
                self.update_operation_stack_ui()
                match path.suffix:
                    case ".usd" | ".usda" | ".usdc":
                        old_stage = Stage.Open(str(self.opened_file_path))
                        self.old_metadata = (
                            old_stage.GetRootLayer().customLayerData.get(
                                "assets_tool:operation"
                            )
                        )
                        del old_stage
                        self.update_operation_name_ui()
                        match dpg.get_value(self.edit_mode_ui):
                            case "update":
                                self.stage = Stage.Open(str(path))
                            case "override" | "override replace":
                                root_layer = Layer.CreateAnonymous()
                                root_layer.subLayerPaths.append(str(path))
                                self.stage = Stage.Open(root_layer)
                            case _:
                                raise Exception()
                        self.xform_cache = XformCache()

                    case _:
                        raise Exception(f"unsupported file {path}")
                for callback in self.on_load_file:
                    callback(path)
            else:
                raise Exception()

        return ret


class Hierarchy:
    def __init__(
        self,
        container: int | str,
        on_select_prim: list[Callable[[Prim], None]],
        get_stage: Callable[[], Stage | None],
    ) -> None:
        self.container = container
        self.on_select_prim = on_select_prim
        self.get_stage = get_stage
        self.tree = Tree()
        self.selected_prim: Prim | None = None
        self.selected_prim_ui = dpg.add_text(parent=self.container)
        self.tree_ui = dpg.add_child_window(parent=self.container)
        self.prim2node = dict[Prim, Tree.Node]()

    def load_stage(self):
        self.prim2node.clear()
        dpg.delete_item(self.tree_ui, children_only=True)
        self.selected_prim = None
        if stage := self.get_stage():
            root_node = Tree.Node()
            root_node.children_ui = self.tree_ui
            self.load_prim(stage.GetPseudoRoot(), root_node)

    def load_prim(self, prim: Prim, node: Tree.Node):
        self.prim2node[prim] = node
        child: Prim
        for child in prim.GetChildren():
            self.add_prim_raw(child, node.children_ui)

    def add_prim_raw(self, prim: Prim, parent_ui: int | str):
        node = self.tree.node(
            prim.GetName(),
            self.select_prim(prim),
            parent_ui,
        )
        self.load_prim(prim, node)

    def add_prim(self, prim: Prim):
        if prim in self.prim2node:
            return
        self.add_prim_raw(prim, self.prim2node[prim.GetParent()].children_ui)

    def remove_prim(self, prim: Prim):
        dpg.delete_item(self.prim2node[prim].root_ui)

    def select_prim(self, prim: Prim):
        def ret():
            stage = self.get_stage()
            assert stage
            self.selected_prim = prim
            dpg.set_value(self.selected_prim_ui, str(prim.GetPath()))
            for callback in self.on_select_prim:
                callback(prim)

        return ret


class Properties:
    def __init__(
        self, container: int | str, get_stage: Callable[[], Stage | None]
    ) -> None:
        self.container = container
        self.get_stage = get_stage
        self.selected_api_ui = dpg.add_text(parent=container)
        self.selected_api_name: str | None = None
        self.tree = Tree()
        self.tree_ui = dpg.add_child_window(parent=container)

    def select_schema(self, schema_name: str) -> Callable[[], None]:
        def ret():
            self.selected_api_name = schema_name
            dpg.set_value(self.selected_api_ui, self.selected_api_name)

        return ret

    def select_prim(self, prim: Prim | None):
        self.selected_api_name = None
        dpg.delete_item(self.tree_ui, children_only=True)
        if not prim:
            return
        type_name, property_names = self.type_properties(prim)
        node = self.tree.node(type_name, lambda: None, self.tree_ui)
        for property_name in property_names:
            self.property_ui(prim, property_name, parent=node.children_ui)
            if property_name == "xformOpOrder":
                xform_ops = prim.GetAttribute(property_name).Get()
                if xform_ops:
                    for xform_op in xform_ops:
                        self.property_ui(prim, str(xform_op), parent=node.children_ui)
        for schema_name, property_names in self.schema_properties(prim):
            node = self.tree.node(
                schema_name, self.select_schema(schema_name), self.tree_ui
            )
            for property_name in property_names:
                self.property_ui(prim, property_name, parent=node.children_ui)

    def property_ui(self, prim: Prim, name: str, parent: int | str):
        prop = prim.GetProperty(name)
        name = name.replace(":", ".")

        if not prop:
            dpg.add_text(f"?{name}", parent=parent)

        elif isinstance(prop, Attribute):
            type_name = str(prop.GetTypeName())
            value = prop.Get()
            stage = self.get_stage()
            assert stage

            is_authored = is_attr_authored_in_layer(prop, stage.GetRootLayer())
            is_blocked = is_attr_blocked(prop)

            dpg.add_text(f"{name}: {type_name}", parent=parent)
            with dpg.group(horizontal=True, parent=parent):
                authored_ui = dpg.add_checkbox(default_value=is_authored, enabled=True)
                blocked_ui = dpg.add_checkbox(default_value=is_blocked, enabled=False)
                edit_ui = None
                read_ui = None
                if type_name == "token[]" or type_name == "token":
                    allowed_tokens = prop.GetMetadata("allowedTokens")
                    if type_name == "token":
                        if allowed_tokens:

                            def on_toggle_authored(sender, app_data, user_data):
                                prop.Set(app_data)

                            edit_ui = dpg.add_combo(
                                tuple(str(x) for x in allowed_tokens),
                                default_value=str(value),
                                enabled=is_authored,
                                callback=on_toggle_authored,
                            )
                        else:

                            def on_toggle_authored(sender, app_data, user_data):
                                prop.Set(app_data)

                            edit_ui = dpg.add_input_text(
                                default_value=value,
                                enabled=is_authored,
                                callback=on_toggle_authored,
                            )

                    elif type_name == "token[]":
                        if value:
                            read_ui = dpg.add_text(f"{value}")
                elif type_name.endswith("[]"):
                    if value:
                        read_ui = dpg.add_text(f"length: {len(value)}")
                else:
                    match type_name:
                        case "bool":

                            def on_toggle_authored(sender, app_data, user_data):
                                prop.Set(app_data)

                            edit_ui = dpg.add_checkbox(
                                default_value=value,
                                enabled=is_authored,
                                callback=on_toggle_authored,
                            )
                        case "int":

                            def on_toggle_authored(sender, app_data, user_data):
                                prop.Set(app_data)

                            edit_ui = dpg.add_input_int(
                                default_value=value,
                                callback=on_toggle_authored,
                                enabled=is_authored,
                            )
                        case "float" | "double":

                            def on_toggle_authored(sender, app_data, user_data):
                                prop.Set(app_data)

                            edit_ui = dpg.add_input_float(
                                default_value=value,
                                callback=on_toggle_authored,
                                enabled=is_authored,
                            )
                        case "vector3f" | "point3f" | "float3" | "double3":

                            def on_toggle_authored(sender, app_data, user_data):
                                prop.Set(Vec3d(app_data[0], app_data[1], app_data[2]))

                            edit_ui = dpg.add_input_floatx(
                                default_value=(value[0], value[1], value[2]),
                                enabled=is_authored,
                                size=3,
                                callback=on_toggle_authored,
                            )
                        case "quatf" | "quatd":
                            img = value.GetImaginary()
                            real = value.GetReal()

                            def on_toggle_authored(sender, app_data, user_data):
                                img = app_data.GetImaginary()
                                real = app_data.GetReal()
                                prop.Set(
                                    Quatd(
                                        real,
                                        img[0],
                                        img[1],
                                        img[2],
                                    )
                                )

                            edit_ui = dpg.add_input_floatx(
                                default_value=(img[0], img[1], img[2], real),
                                enabled=is_authored,
                                size=4,
                                callback=on_toggle_authored,
                            )

                def on_toggle_authored(sender, app_data, user_data):
                    if not app_data:
                        prop.Clear()
                        is_blocked = is_attr_blocked(prop)
                        dpg.set_value(blocked_ui, is_blocked)
                    if edit_ui:
                        dpg.configure_item(edit_ui, enabled=app_data)
                        if not app_data:
                            dpg.set_value(edit_ui, prop.Get())
                    if read_ui:
                        if not app_data:
                            dpg.set_value(read_ui, prop.Get())
                    dpg.configure_item(blocked_ui, enabled=app_data)

                dpg.configure_item(
                    authored_ui, callback=on_toggle_authored, enabled=True
                )

                def on_toggle_blocked(sender, app_data, user_data):
                    if app_data:
                        prop.Block()
                    if not app_data:
                        prop.Clear()
                    if edit_ui:
                        dpg.configure_item(edit_ui, enabled=not app_data)
                        dpg.set_value(edit_ui, prop.Get())
                        dpg.configure_item(edit_ui, enabled=app_data)

                dpg.configure_item(blocked_ui, callback=on_toggle_blocked)

        elif isinstance(prop, Relationship):
            dpg.add_text(f"{name}: relationship", parent=parent)
            paths = prop.GetTargets()
            with dpg.child_window(parent=parent, auto_resize_y=True):
                for path in paths:
                    dpg.add_text(str(path))

        else:
            raise Exception()

    def schema_properties(self, prim: Prim) -> Iterable[tuple[str, Iterable[str]]]:
        schema_attributes = []

        # 1. Get applied schemas
        applied_schemas = prim.GetAppliedSchemas()  # list of schema names
        for schema_name in applied_schemas:
            # 2. Detect if schema is multiple-apply
            if ":" in schema_name:  # instance form like "CollectionAPI:render"
                base_name, instance_name = schema_name.split(":", 1)
                prim_def = registry.FindAppliedAPIPrimDefinition(base_name)  # type: ignore
            else:
                base_name = schema_name
                instance_name = None
                prim_def = registry.FindAppliedAPIPrimDefinition(base_name)  # type: ignore

            if not prim_def:
                continue

            # 3. Get all property names
            prop_names = prim_def.GetPropertyNames()

            # 4. Namespace attributes for multiple-apply schemas
            if instance_name:
                prop_names = [
                    p.replace("__INSTANCE_NAME__", instance_name) for p in prop_names
                ]

            schema_attributes.append((schema_name, prop_names))

        attr_names = []
        primvar_api = PrimvarsAPI(prim)
        primvars = primvar_api.GetPrimvars()
        print(primvars)
        for primvar in primvars:
            attr_name = primvar.GetAttr().GetName()
            attr_names.append(attr_name)
        if len(attr_names) > 0:
            schema_attributes.append(("PrimvarAPI", attr_names))

        return schema_attributes

    def type_properties(self, prim: Prim) -> tuple[str, Iterable[str]]:
        type_name = prim.GetTypeName()
        if not type_name:
            return "Prim", ()
        prim_def = registry.FindConcretePrimDefinition(type_name)
        if prim_def is None:
            raise Exception(f"type {type_name} of prim {prim} unknown")
        return (type_name, prim_def.GetPropertyNames())


class BlenderClient:
    class SyncedMesh:
        mesh: Mesh
        mesh_shared: tuple[SharedMemory, SharedMemory]

    class Synced:
        def __init__(self, root_prim: Prim, stage: Stage) -> None:
            self.stage = stage
            self.root_prim = root_prim
            self.meshes = dict[Path, BlenderClient.SyncedMesh]()

    def __init__(
        self,
        container: int | str,
        get_selected_prim: Callable[[], Prim | None],
        get_stage: Callable[[], Stage | None],
        get_xform_cache: Callable[[], XformCache | None],
        mut_on_tick: list[Callable[[], None]],
        mut_on_end: list[Callable[[], None]],
    ) -> None:
        self.tasks = Queue[Callable[[], None]]()
        self.run_commands = RunCommands(
            (
                software_client.SyncMesh(self.sync_mesh),
                software_client.SyncXform(self.sync_xform),
            )
        )
        self.client = Client(
            lambda data: self.run_commands.run(data),
            (self.on_start,),
            (self.on_end,),
        )
        self.container = container
        self.get_selected_prim = get_selected_prim
        self.get_stage = get_stage
        self.get_xform_cache = get_xform_cache
        with dpg.child_window(auto_resize_y=True, parent=self.container):
            dpg.add_text("Blender Client")
            self.synced_ui = dpg.add_input_text(label="synced")
            self.port_input = dpg.add_input_int(label="port", default_value=8888)
            with dpg.group(horizontal=True):
                self.connect_ui = dpg.add_checkbox(
                    label="connect", callback=self.toggle_connection
                )
                self.connected_ui = dpg.add_checkbox(enabled=False)
            self.sync_ui = dpg.add_button(label="sync", callback=self.sync)
        self.synced: BlenderClient.Synced | None = None
        mut_on_tick.append(self.on_tick)
        mut_on_end.append(lambda: self.client.end())

    def toggle_connection(self, sender, app_data, user_data):
        if app_data:
            self.client.start(dpg.get_value(self.port_input))
        else:
            self.client.end()

    def on_start(self):
        dpg.set_value(self.connected_ui, True)

    def on_end(self):
        self.synced = None
        dpg.set_value(self.connect_ui, False)
        dpg.set_value(self.connected_ui, False)

    def on_tick(self):
        while not self.tasks.empty():
            task = self.tasks.get_nowait()
            task()

    def unsync(self):
        if self.client.on:
            software_client.clear(self.client)
            dpg.set_value(self.sync_ui, "")

    def sync(self):
        software_client.clear(self.client)
        if selected_prim := self.get_selected_prim():
            dpg.set_value(self.sync_ui, selected_prim.GetPath())
            root = selected_prim.GetParent()
            stage = self.get_stage()
            xform_cache = self.get_xform_cache()
            assert stage and xform_cache
            self.synced = self.Synced(root, stage)
            root_path = root.GetPath()
            for prim in PrimRange(selected_prim):
                relative_path = Path(str(prim.GetPath().MakeRelativePath(root_path)))
                if prim.IsA(Mesh):  # type: ignore
                    mesh = Mesh(prim)
                    synced_mesh = BlenderClient.SyncedMesh()
                    self.synced.meshes[relative_path] = synced_mesh
                    face_vertex_counts = array(mesh.GetFaceVertexCountsAttr().Get())
                    assert numpy.all(face_vertex_counts == 3)
                    synced_mesh.mesh_shared = software_client.create_mesh(
                        self.client,
                        array(mesh.GetPointsAttr().Get()),
                        array(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3),
                        relative_path,
                    )
                if prim.IsA(Xformable):  # type: ignore
                    translation, rotation, scale = from_usd_transform(
                        xform_cache.GetLocalTransformation(prim)[0]
                    )
                    software_client.set_xform(
                        self.client, translation, rotation, scale, relative_path
                    )

    def sync_mesh(
        self,
        positions: NDArray[float32],
        indices: NDArray[int32],
        path: Path,
        guard: Any,
    ):
        def run():
            guard  # type: ignore
            if self.synced:
                prim = self.synced.stage.GetPrimAtPath(
                    self.synced.root_prim.GetPath().AppendPath(
                        str(path).replace("\\", "/")
                    )
                )
                mesh = Mesh(prim)
                mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(positions))
                mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(indices))
                mesh.GetFaceVertexCountsAttr().Set(
                    Vt.IntArray.FromNumpy(numpy.full(len(indices) // 3, 3))
                )
                mesh.GetNormalsAttr().Block()
                mesh.GetExtentAttr().Block()
                for primvar in PrimvarsAPI(prim).GetPrimvars():
                    primvar.GetAttr().Block()

        self.tasks.put(run)

    def sync_xform(
        self,
        translation: NDArray[float],
        rotation: NDArray[float],
        scale: NDArray[float],
        path: Path,
    ):
        def run():
            if self.synced:
                prim = self.synced.stage.GetPrimAtPath(
                    self.synced.root_prim.GetPath().AppendPath(
                        str(path).replace("\\", "/")
                    )
                )
                xform = Xformable(prim)

                if op := xform.GetTranslateOp():
                    translate_op = op
                else:
                    translate_op = xform.AddTranslateOp()

                if op := xform.GetOrientOp():
                    orient_op = op
                else:
                    orient_op = xform.AddOrientOp()

                if op := xform.GetScaleOp():
                    scale_op = op
                else:
                    scale_op = xform.AddScaleOp()

                translate_op.Set((translation[0], translation[1], translation[2]))

                orient_op.Set(
                    quat(orient_op.GetPrecision())(
                        rotation[3], rotation[0], rotation[1], rotation[2]
                    )
                )
                scale_op.Set((scale[0], scale[1], scale[2]))
                xform.SetXformOpOrder((translate_op, orient_op, scale_op))

        self.tasks.put(run)


class PrimUtil:
    def __init__(
        self,
        container: int | str,
        get_stage: Callable[[], Stage | None],
        get_selected_prim: Callable[[], Prim | None],
        on_add_prim: list[Callable[[Prim], None]],
        on_delete_prim: list[Callable[[Prim], None]],
    ) -> None:
        self.container = container
        self.get_stage = get_stage
        self.get_selected_prim = get_selected_prim
        self.on_add_prim = on_add_prim
        self.on_delete_prim = on_delete_prim
        self.copied_prim: Prim | None = None
        with dpg.child_window(auto_resize_y=True, parent=container):
            dpg.add_text("Prim Util")
            self.name_ui = dpg.add_input_text(label="name", default_value="new_prim")
            self.mode_ui = dpg.add_combo(
                ("child", "brother"), label="mode", default_value="brother"
            )
            with dpg.group(horizontal=True):
                dpg.add_button(label="create", callback=self.create_prim)
                dpg.add_button(label="delete", callback=self.delete_prim)
                dpg.add_button(label="copy", callback=self.copy_prim)
                dpg.add_button(label="paste", callback=self.paste_prim)

    def get_create_path(self) -> UsdPath:
        selected_prim = self.get_selected_prim()
        if not selected_prim:
            path = UsdPath("/")
        else:
            match dpg.get_value(self.mode_ui):
                case "child":
                    path = selected_prim.GetPath()
                case "brother":
                    path = selected_prim.GetParent().GetPath()
                case _:
                    raise Exception()
        path = path.AppendChild(dpg.get_value(self.name_ui))
        return path

    def create_prim(self):
        stage = self.get_stage()
        assert stage
        path = self.get_create_path()
        prim = stage.DefinePrim(path)
        for callback in self.on_add_prim:
            callback(prim)

    def delete_prim(self):
        stage, selected_prim = self.get_stage(), self.get_selected_prim()
        assert stage and selected_prim
        for callback in self.on_delete_prim:
            callback(selected_prim)
        path = selected_prim.GetPath()
        if is_prim_authored_in_layer(selected_prim, stage.GetRootLayer()):
            stage.RemovePrim(path)
        else:
            selected_prim.SetActive(False)

    def copy_prim(self):
        if prim := self.get_selected_prim():
            self.copied_prim = prim

    def paste_prim(self):
        if self.copied_prim:
            stage, selected_prim = self.get_stage(), self.get_selected_prim()
            assert stage and selected_prim
            new_prim = copy_prim(stage, self.copied_prim, self.get_create_path(), True)
            for callback in self.on_add_prim:
                callback(new_prim)


class SchemaUtil:
    def __init__(
        self,
        container: int | str,
        get_selected_prim: Callable[[], Prim | None],
        get_selected_api_name: Callable[[], str | None],
        on_schema_change: list[Callable[[Prim], None]],
    ) -> None:
        self.container = container
        self.get_selected_api_name = get_selected_api_name
        self.get_selected_prim = get_selected_prim
        self.on_add_schema = on_schema_change

        type_types = UsdType.FindByName("UsdTyped").GetAllDerivedTypes()
        api_types = UsdType.FindByName("UsdAPISchemaBase").GetAllDerivedTypes()
        type_names = list[str]()
        api_names = list[str]()
        for type in type_types:
            if SchemaRegistry.IsConcrete(type):
                type_names.append(SchemaRegistry.GetSchemaTypeName(type))
        for type in api_types:
            if SchemaRegistry.IsAppliedAPISchema(type):
                api_names.append(SchemaRegistry.GetSchemaTypeName(type))

        with dpg.child_window(auto_resize_y=True, parent=container):
            dpg.add_text("Schema Util")
            with dpg.tree_node(label="type", default_open=True):
                self.select_type_ui = dpg.add_combo(type_names, label="type")
                dpg.add_button(label="set type", callback=self.set_type)
            with dpg.tree_node(label="api", default_open=True):
                self.select_api_ui = dpg.add_combo(api_names, label="api")
                self.instance_name_ui = dpg.add_input_text(
                    label="instance name", show=False
                )
                with dpg.group(horizontal=True):
                    dpg.add_button(label="add api", callback=self.add_api)
                    dpg.add_button(label="remove api", callback=self.remove_api)

            def select_api(sender, app_data, user_data):
                is_multi = SchemaRegistry.IsMultipleApplyAPISchema(app_data)
                dpg.configure_item(self.instance_name_ui, show=is_multi)

            dpg.configure_item(self.select_api_ui, callback=select_api)

    def remove_api(self):
        if prim := self.get_selected_prim():
            if api_name := self.get_selected_api_name():
                api = SchemaRegistry.GetTypeFromSchemaTypeName(api_name)
                prim.RemoveAPI(api)
                for callback in self.on_add_schema:
                    callback(prim)
            elif api := dpg.get_value(self.select_api_ui):
                prim.RemoveAPI(api)

    def set_type(self):
        selected_prim = self.get_selected_prim()
        assert selected_prim
        selected_prim.SetTypeName(dpg.get_value(self.select_type_ui))
        for callback in self.on_add_schema:
            callback(selected_prim)

    def add_api(self):
        selected_prim = self.get_selected_prim()
        assert selected_prim
        type = SchemaRegistry.GetTypeFromSchemaTypeName(
            dpg.get_value(self.select_api_ui)
        )
        is_multi = SchemaRegistry.IsMultipleApplyAPISchema(type)
        if not is_multi:
            selected_prim.ApplyAPI(type)
        else:
            instance_name = dpg.get_value(self.instance_name_ui)
            selected_prim.ApplyAPI(type, instance_name)
        for callback in self.on_add_schema:
            callback(selected_prim)

class LayerUtil:
    def __init__(
        self,
        container: int | str,
        get_stage: Callable[[], Stage | None],
        on_clear: list[Callable[[], None]],
    ) -> None:
        self.container = container
        self.get_stage = get_stage
        self.on_clear = on_clear
        with dpg.child_window(auto_resize_y=True, parent=self.container):
            dpg.add_text("Layer Util")
            dpg.add_button(label="clear", callback=self.clear)

    def clear(self):
        if stage := self.get_stage():
            root_layer = stage.GetRootLayer()
            operation_metadata = root_layer.customLayerData.get("assets_tool:operation")
            sublayers = list(root_layer.subLayerPaths)  # type: ignore
            root_layer.Clear()
            if operation_metadata:
                custom_layer_data = root_layer.customLayerData
                custom_layer_data["assets_tool:operation"] = operation_metadata
                root_layer.customLayerData = custom_layer_data
            for sublayer in sublayers:
                root_layer.subLayerPaths.append(sublayer)
            for callback in self.on_clear:
                callback()


class UI:
    def __init__(self):
        dpg.create_context()

        self.root = dpg.add_window()
        with dpg.table(
            header_row=False,
            resizable=True,
            policy=dpg.mvTable_SizingStretchProp,
            parent=self.root,
        ):
            dpg.add_table_column()
            dpg.add_table_column()
            dpg.add_table_column()
            dpg.add_table_column()
            with dpg.table_row(label=""):
                self.file_explorer = self.child("File Explorer")
                self.heirarchy = self.child("Heirarchy")
                self.properties = self.child("Properties")
                self.operators = self.child("Operators")

        self.on_tick = list[Callable[[], None]]()
        self.on_end = list[Callable[[], None]]()

    def child(self, label: str) -> int | str:
        with dpg.child_window():
            dpg.add_text(label)
            return dpg.add_child_window()

    def run(self):
        dpg.set_primary_window(self.root, True)
        dpg.create_viewport(title="Assets Tool", width=800, height=400)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        while dpg.is_dearpygui_running():
            for callback in self.on_tick:
                callback()
            dpg.render_dearpygui_frame()
        for callback in self.on_end:
            callback()
        dpg.destroy_context()


class App:
    def __init__(self):
        self.ui = UI()
        self.properties = Properties(
            self.ui.properties, lambda: self.file_explorer.stage
        )
        self.hierarchy = Hierarchy(
            self.ui.heirarchy,
            [self.properties.select_prim],
            lambda: self.file_explorer.stage,
        )
        self.file_explorer = FileExplorer(
            self.ui.file_explorer,
            [
                lambda _: self.hierarchy.load_stage(),
                lambda _: self.properties.select_prim(None),
                lambda _: self.blender_client.unsync(),
            ],
        )
        self.blender_client = BlenderClient(
            self.ui.operators,
            lambda: self.hierarchy.selected_prim,
            lambda: self.file_explorer.stage,
            lambda: self.file_explorer.xform_cache,
            self.ui.on_tick,
            self.ui.on_end,
        )
        self.prim_util = PrimUtil(
            self.ui.operators,
            lambda: self.file_explorer.stage,
            lambda: self.hierarchy.selected_prim,
            [self.hierarchy.add_prim],
            [self.hierarchy.remove_prim, lambda _: self.properties.select_prim(None)],
        )
        self.schema_util = SchemaUtil(
            self.ui.operators,
            lambda: self.hierarchy.selected_prim,
            lambda: self.properties.selected_api_name,
            [self.properties.select_prim],
        )
        self.layer_util = LayerUtil(
            self.ui.operators,
            lambda: self.file_explorer.stage,
            [self.hierarchy.load_stage],
        )
        self.file_explorer.load_path(Path(".").resolve())()

    def run(self):
        self.ui.run()

if __name__ == "__main__":
    app = App()
    app.run()