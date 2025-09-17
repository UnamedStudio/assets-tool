from __future__ import annotations
from collections.abc import Callable, Iterable
from copy import deepcopy
from dataclasses import dataclass
from importlib.resources import as_file
from math import ceil
import os
from pathlib import Path
from queue import PriorityQueue, Queue
from time import sleep
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

import screeninfo
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
        fold_button: int | str
        select_button: int | str
        children_ui: int | str
        root_ui: int | str

    def __init__(self, unfold_label: str = "+", fold_label="-") -> None:
        self.open_label = unfold_label
        self.close_label = fold_label

    def node(self, label: str, callback: Callable[[], None], parent: int | str) -> Node:
        node = self.Node()
        node.is_open = False
        with dpg.group(parent=parent) as root_ui:
            with dpg.group(horizontal=True):
                node.fold_button = dpg.add_button(
                    label=self.open_label, callback=self.toggle_open(node)
                )
                node.select_button = dpg.add_button(label=label, callback=callback)
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
                dpg.configure_item(node.fold_button, label=self.open_label)
            else:
                node.is_open = True
                dpg.configure_item(node.children_ui, show=True)
                dpg.configure_item(node.fold_button, label=self.close_label)

        return ret


class FileExplorer:
    def __init__(
        self,
        container: int | str,
        load_file_callbacks: list[Callable[[Path | None], None]],
        opened_theme: int | str,
    ) -> None:
        self.container = container
        self.on_load_file = load_file_callbacks
        self.opened_theme = opened_theme
        self.stage: Stage | None = None
        self.old_metadata: dict | None = None

        self.opened_file_path: Path | None = None
        self.opened_directory_path: Path | None = None
        self.opened_file_ui = dpg.add_text(parent=self.container)
        self.path2button = dict[Path, int | str]()

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
                        if input := old_metadata["input"]:
                            input_path = self.opened_file_path.parent / input
                            input_stage = Stage.Open(str(input_path))
                            input_root_layer = input_stage.GetRootLayer()
                            input_custom_layer_data = input_root_layer.customLayerData
                            input_metadata = input_custom_layer_data[
                                "assets_tool:operation"
                            ]
                            input_metadata["output"] = Path(
                                os.path.relpath(
                                    str(path.resolve()),
                                    str(input_path.parent.resolve()),
                                )
                            ).as_posix()
                            input_custom_layer_data["assets_tool:operation"] = (
                                input_metadata
                            )
                            input_root_layer.customLayerData = input_custom_layer_data
                            input_root_layer.Save()
                            del input_stage
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
            del self.stage
            if reload:
                self.load_path(self.opened_file_path.parent)()

            self.load_path(None)()

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
        if self.opened_file_path:
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

    def update_opened_file_ui(self):
        assert self.opened_directory_path
        for path, button in self.path2button.items():
            if self.opened_file_path and path == self.opened_file_path:
                dpg.bind_item_theme(button, self.opened_theme)
            else:
                dpg.bind_item_theme(button, 0)
        dpg.set_value(
            self.opened_file_ui,
            os.path.relpath(
                str(self.opened_file_path.resolve()),
                str(self.opened_directory_path.resolve()),
            )
            if self.opened_file_path
            else "",
        )

    def load_path(
        self,
        path: Path | None,
    ):
        def ret():
            if not path:
                self.opened_file_path = path
                self.stage = None
                self.old_metadata = None
                self.update_operation_stack_ui()
                for callback in self.on_load_file:
                    callback(path)
            elif path.is_dir():
                self.opened_directory_path = path
                self.path2button.clear()
                dpg.delete_item(self.tree_ui, children_only=True)
                dpg.add_button(
                    label="..",
                    callback=self.load_path(path.parent),
                    parent=self.tree_ui,
                )
                for child in sorted(path.iterdir()):
                    button = dpg.add_button(
                        label=child.name,
                        callback=self.load_path(child),
                        parent=self.tree_ui,
                    )
                    self.path2button[child] = button
            elif path.is_file():
                self.opened_file_path = path
                self.stage = None
                self.old_metadata = None
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
                                stage = Stage.Open(str(path))
                                defaultPrim = stage.GetRootLayer().defaultPrim
                                del stage
                                root_layer.defaultPrim = defaultPrim
                                self.stage = Stage.Open(root_layer)
                            case _:
                                raise Exception()

                    case _:
                        raise Exception(f"unsupported file {path}")
                for callback in self.on_load_file:
                    callback(path)
            else:
                raise Exception()
            self.update_opened_file_ui()

        return ret


class Hierarchy:
    def __init__(
        self,
        container: int | str,
        select_prim: Callable[[Prim], None],
        get_stage: Callable[[], Stage | None],
        selected_theme: int | str,
    ) -> None:
        self.container = container
        self.select_prim = select_prim
        self.get_stage = get_stage
        self.selected_theme = selected_theme
        self.tree = Tree()
        self.selected_prim_ui = dpg.add_text(parent=self.container)
        self.tree_ui = dpg.add_child_window(parent=self.container)
        self.prim2node = dict[Prim, Tree.Node]()

    def load_stage(self):
        self.prim2node.clear()
        dpg.delete_item(self.tree_ui, children_only=True)
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
            lambda: self.select_prim(prim),
            parent_ui,
        )
        self.load_prim(prim, node)

    def add_prim(self, prim: Prim):
        if prim in self.prim2node:
            return
        self.add_prim_raw(prim, self.prim2node[prim.GetParent()].children_ui)

    def remove_prim(self, prim: Prim):
        dpg.delete_item(self.prim2node[prim].root_ui)
        self.prim2node.pop(prim)

    def on_select_prim(self, prev: Prim | None, new: Prim | None):
        if prev:
            if node := self.prim2node.get(prev):
                dpg.bind_item_theme(node.select_button, 0)
        if new:
            dpg.bind_item_theme(
                self.prim2node[new].select_button,
                self.selected_theme,
            )
        dpg.set_value(self.selected_prim_ui, str(new.GetPath()) if new else "")

class Properties:
    def __init__(
        self,
        container: int | str,
        get_stage: Callable[[], Stage | None],
        selected_theme: int | str,
    ) -> None:
        self.container = container
        self.get_stage = get_stage
        self.selected_api_ui = dpg.add_text(parent=container)
        self.selected_api_name: str | None = None
        self.tree = Tree()
        self.tree_ui = dpg.add_child_window(parent=container)
        self.selected_theme = selected_theme
        self.api_name2node = dict[str, Tree.Node]()

    def select_schema(self, schema_name: str) -> Callable[[], None]:
        def ret():
            if self.selected_api_name:
                dpg.bind_item_theme(
                    self.api_name2node[self.selected_api_name].select_button, 0
                )
            self.selected_api_name = schema_name
            dpg.bind_item_theme(
                self.api_name2node[self.selected_api_name].select_button,
                self.selected_theme,
            )
            dpg.set_value(self.selected_api_ui, self.selected_api_name)

        return ret

    def select_prim(self, prim: Prim | None):
        self.selected_api_name = None
        self.api_name2node.clear()
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
        for api_name, property_names in self.api_properties(prim):
            node = self.tree.node(api_name, self.select_schema(api_name), self.tree_ui)
            self.api_name2node[api_name] = node
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

    def api_properties(self, prim: Prim) -> Iterable[tuple[str, Iterable[str]]]:
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

class SelectionUtil:
    @dataclass
    class SelectInfo:
        recursive: bool
        exclusive: bool

    @dataclass
    class Selection:
        selected: dict[Prim, SelectionUtil.SelectInfo]
        current_info: SelectionUtil.SelectInfo
        api_filter: list[str]
        current: Prim | None = None
        name_filter: str | None = None
        type_filter: str | None = None

        def iter(self) -> Iterable[Prim]:
            def traverse_prim(prim: Prim) -> Iterable[Prim]:
                if self.if_filter(prim):
                    yield prim
                if self.current_info.recursive:
                    child: Prim
                    for child in prim.GetChildren():
                        yield from traverse_prim(child)

            if self.current:
                yield from traverse_prim(self.current)

        def foreach(self, func: Callable[[Prim], bool]):
            def traverse_prim(prim: Prim):
                recursive = True
                if self.if_filter(prim):
                    recursive = func(prim)
                if recursive and self.current_info.recursive:
                    child: Prim
                    for child in prim.GetChildren():
                        yield from traverse_prim(child)

            if self.current:
                traverse_prim(self.current)

        def if_filter(self, prim: Prim) -> bool:
            if self.name_filter:
                if prim.GetName() != self.name_filter:
                    return False
            if self.type_filter:
                if not prim.IsA(self.type_filter):
                    return False
            for api in self.api_filter:
                if not prim.HasAPI(api):
                    return False
            return True

    def __init__(
        self,
        container: int | str,
        selected_theme: int | str,
        on_select: list[Callable[[Prim | None, Prim | None], None]],
    ) -> None:
        self.container = container
        self.selection = self.Selection({}, self.SelectInfo(False, False), [])
        self.selected_theme = selected_theme
        self.on_select = on_select

        type_types = UsdType.FindByName("UsdTyped").GetAllDerivedTypes()
        api_types = UsdType.FindByName("UsdAPISchemaBase").GetAllDerivedTypes()
        type_names: list[str] = []
        api_names: list[str] = []
        for type in type_types:
            type_names.append(SchemaRegistry.GetSchemaTypeName(type))
        for type in api_types:
            api_names.append(SchemaRegistry.GetSchemaTypeName(type))
        type_names = sorted(type_names)
        api_names = sorted(api_names)
        type_names.insert(0, "None")
        api_names.insert(0, "None")

        with dpg.child_window(auto_resize_y=True, parent=self.container):
            dpg.add_text("Selection Util")
            with dpg.tree_node(label="filters"):
                self.name_filter_ui = dpg.add_input_text(label="name")

                def on_type_filter_ui(sender, app_data, user_data):
                    if app_data == "None":
                        self.selection.type_filter = None
                    else:
                        self.selection.type_filter = app_data

                self.type_filter_ui = dpg.add_combo(
                    type_names, label="type", callback=on_type_filter_ui
                )

                def on_api_filter_ui(sender, app_data, user_data):
                    if app_data == "None":
                        self.selection.api_filter = []
                    else:
                        self.selection.api_filter = [app_data]

                self.api_filter_ui = dpg.add_combo(
                    api_names, label="api", callback=on_api_filter_ui
                )
            self.selection_ui = dpg.add_child_window(auto_resize_y=True)
            with dpg.group(horizontal=True):

                def on_recursive_ui(sender, app_data, user_data):
                    self.selection.current_info.recursive = app_data

                def on_exclusive_ui(sender, app_data, user_data):
                    self.selection.current_info.exclusive = app_data

                dpg.add_checkbox(
                    label="recursive", default_value=False, callback=on_recursive_ui
                )
                dpg.add_checkbox(
                    label="exclusive",
                    default_value=False,
                    callback=on_exclusive_ui,
                    show=False,
                )
            with dpg.group(horizontal=True):
                dpg.add_button(label="add", callback=self.add)
                dpg.add_button(label="remove", callback=self.remove)
                dpg.add_button(label="clear", callback=self.clear)

    def add(self):
        if self.selection.current:
            self.selection.selected[self.selection.current] = deepcopy(
                self.selection.current_info
            )
            self.update_selected_ui()

    def remove(self):
        if self.selection.current:
            self.selection.selected.pop(self.selection.current, None)
            self.update_selected_ui()

    def clear(self):
        self.selection.selected.clear()
        self.select(None)
        self.update_selected_ui()

    def update_selected_ui(self):
        dpg.delete_item(self.selection_ui, children_only=True)
        for prim, info in self.selection.selected.items():
            self.add_selected_ui(prim, info)

    def add_selected_ui(self, prim: Prim, info: SelectInfo):
        with dpg.group(horizontal=True, parent=self.selection_ui):
            recursive_ui = dpg.add_checkbox(default_value=info.recursive)
            selected_ui = dpg.add_button(
                label=("^" if info.exclusive else "") + str(prim.GetPath()),
                callback=lambda: self.select(prim),
            )
            dpg.bind_item_theme(
                selected_ui,
                self.selected_theme if prim == self.selection.current else 0,
            )

            def on_recursive_ui(sender, app_data, user_data):
                info.recursive = app_data

            dpg.configure_item(recursive_ui, callback=on_recursive_ui)

    def select(self, prim: Prim | None):
        prev = self.selection.current
        if prev == prim:
            prim = None
        self.selection.current = prim
        self.update_selected_ui()
        for callback in self.on_select:
            callback(prev, prim)


class BlenderClient:
    class SyncedMesh:
        mesh: Mesh

    class Synced:
        def __init__(self, root_prim: Prim, stage: Stage) -> None:
            self.stage = stage
            self.root_prim = root_prim
            self.meshes = dict[Path, BlenderClient.SyncedMesh]()

    def __init__(
        self,
        container: int | str,
        get_selection: Callable[[], SelectionUtil.Selection],
        get_stage: Callable[[], Stage | None],
        mut_on_tick: list[Callable[[], None]],
        mut_on_end: list[Callable[[], None]],
    ) -> None:
        self.tasks = Queue[Callable[[], None]]()
        self.client = Client(
            lambda data: self.run_commands.run(data),
            (self.on_start,),
            (self.on_end,),
        )
        self.run_commands = RunCommands(
            (
                software_client.SyncMesh(self.sync_mesh),
                software_client.SyncXform(self.sync_xform),
            ),
            self.client,
        )
        self.container = container
        self.get_selection = get_selection
        self.get_stage = get_stage
        with dpg.child_window(auto_resize_y=True, parent=self.container):
            dpg.add_text("Blender Client")
            self.port_input = dpg.add_input_int(label="port", default_value=8888)
            with dpg.group(horizontal=True):
                self.connect_ui = dpg.add_checkbox(
                    label="connect", callback=self.toggle_connection
                )
                self.connected_ui = dpg.add_checkbox(enabled=False)
            self.synced_ui = dpg.add_input_text(label="synced")
            with dpg.group(horizontal=True):
                self.if_sync_mesh_ui = dpg.add_checkbox(
                    label="mesh", default_value=True
                )
                self.if_sync_xform_ui = dpg.add_checkbox(
                    label="xform", default_value=True
                )
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
            del self.synced
            self.synced = None
            software_client.clear(self.client)
            dpg.set_value(self.sync_ui, "")

    def sync(self):
        software_client.clear(self.client)
        selection = self.get_selection()
        if selection.current:
            dpg.set_value(self.sync_ui, selection.current.GetPath())
            root = selection.current.GetParent()
            stage = self.get_stage()
            xform_cache = XformCache()
            assert stage and xform_cache
            self.synced = self.Synced(root, stage)
            root_path = root.GetPath()
            command_count = 0
            for prim in PrimRange(selection.current):
                in_selection = selection.if_filter(prim)
                relative_path = Path(str(prim.GetPath().MakeRelativePath(root_path)))
                if in_selection and prim.IsA(Mesh):  # type: ignore
                    mesh = Mesh(prim)
                    synced_mesh = BlenderClient.SyncedMesh()
                    self.synced.meshes[relative_path] = synced_mesh
                    face_vertex_counts = array(mesh.GetFaceVertexCountsAttr().Get())
                    assert numpy.all(face_vertex_counts == 3)
                    software_client.create_mesh(
                        self.client,
                        array(mesh.GetPointsAttr().Get()),
                        array(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3),
                        relative_path,
                        dpg.get_value(self.if_sync_mesh_ui),
                    )
                    command_count += 4
                if prim.IsA(Xformable):  # type: ignore
                    translation, rotation, scale = from_usd_transform(
                        xform_cache.GetLocalTransformation(prim)[0]
                    )
                    software_client.set_xform(
                        self.client,
                        translation,
                        rotation,
                        scale,
                        relative_path,
                        in_selection and dpg.get_value(self.if_sync_xform_ui),
                    )
                    command_count += 1
                if command_count >= 20:
                    sleep(0.1)
                    command_count -= 20

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
        get_selection: Callable[[], SelectionUtil.Selection],
        on_add_prim: list[Callable[[Prim], None]],
        on_delete_prim: list[Callable[[Prim], None]],
    ) -> None:
        self.container = container
        self.get_stage = get_stage
        self.get_selection = get_selection
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

    def get_create_path(self, prim: Prim | None) -> UsdPath:
        if not prim:
            path = UsdPath("/")
        else:
            match dpg.get_value(self.mode_ui):
                case "child":
                    path = prim.GetPath()
                case "brother":
                    path = prim.GetParent().GetPath()
                case _:
                    raise Exception()
        path = path.AppendChild(dpg.get_value(self.name_ui))
        return path

    def create_prim(self):
        stage = self.get_stage()
        assert stage
        for prim in self.get_selection().iter():
            path = self.get_create_path(prim)
            prim = stage.DefinePrim(path)
            for callback in self.on_add_prim:
                callback(prim)

    def delete_prim(self):
        stage = self.get_stage()
        assert stage
        for prim in self.get_selection().iter():
            for callback in self.on_delete_prim:
                callback(prim)
            path = prim.GetPath()
            if is_prim_authored_in_layer(prim, stage.GetRootLayer()):
                stage.RemovePrim(path)
            else:
                prim.SetActive(False)

    def copy_prim(self):
        if prim := self.get_selection().current:
            self.copied_prim = prim

    def paste_prim(self):
        current_prim = self.get_selection().current
        if self.copied_prim and current_prim:
            stage = self.get_stage()
            assert stage
            new_prim = copy_prim(
                stage, self.copied_prim, self.get_create_path(current_prim), True
            )
            for callback in self.on_add_prim:
                callback(new_prim)


class SchemaUtil:
    def __init__(
        self,
        container: int | str,
        get_selection: Callable[[], SelectionUtil.Selection],
        get_selected_api_name: Callable[[], str | None],
        on_schema_change: list[Callable[[Prim], None]],
    ) -> None:
        self.container = container
        self.get_selected_api_name = get_selected_api_name
        self.get_selection = get_selection
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
        type_names = sorted(type_names)
        api_names = sorted(api_names)

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
        for prim in self.get_selection().iter():
            if api_name := self.get_selected_api_name():
                api = SchemaRegistry.GetTypeFromSchemaTypeName(api_name)
                prim.RemoveAPI(api)
                for callback in self.on_add_schema:
                    callback(prim)
            elif api := dpg.get_value(self.select_api_ui):
                prim.RemoveAPI(api)

    def set_type(self):
        for prim in self.get_selection().iter():
            prim.SetTypeName(dpg.get_value(self.select_type_ui))
            for callback in self.on_add_schema:
                callback(prim)

    def add_api(self):
        for prim in self.get_selection().iter():
            type = SchemaRegistry.GetTypeFromSchemaTypeName(
                dpg.get_value(self.select_api_ui)
            )
            is_multi = SchemaRegistry.IsMultipleApplyAPISchema(type)
            if not is_multi:
                prim.ApplyAPI(type)
            else:
                instance_name = dpg.get_value(self.instance_name_ui)
                prim.ApplyAPI(type, instance_name)
            for callback in self.on_add_schema:
                callback(prim)

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
            default_prim = root_layer.defaultPrim
            sublayers = list(root_layer.subLayerPaths)  # type: ignore
            root_layer.Clear()
            if operation_metadata:
                custom_layer_data = root_layer.customLayerData
                custom_layer_data["assets_tool:operation"] = operation_metadata
                root_layer.customLayerData = custom_layer_data
            root_layer.defaultPrim = default_prim
            for sublayer in sublayers:
                root_layer.subLayerPaths.append(sublayer)
            for callback in self.on_clear:
                callback()


class UI:
    def __init__(self):
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
        dpg.create_context()
        from importlib.resources import files

        font_path = files("assets_tool.assets").joinpath("fonts/ARIAL.TTF")
        max_height = 600
        for i in screeninfo.get_monitors():
            if i.height > max_height:
                max_height = i.height
        with dpg.font_registry():
            with as_file(font_path) as path:
                default_font = dpg.add_font(str(path), ceil(max_height * 0.012))
                dpg.bind_font(default_font)
        with dpg.theme() as selected_theme:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(
                    dpg.mvThemeCol_Button, (120, 120, 0), category=dpg.mvThemeCat_Core
                )
        self.selected_theme = selected_theme
        self.ui = UI()
        self.properties = Properties(
            self.ui.properties, lambda: self.file_explorer.stage, self.selected_theme
        )
        self.hierarchy = Hierarchy(
            self.ui.heirarchy,
            lambda prim: self.selection_util.select(prim),
            lambda: self.file_explorer.stage,
            self.selected_theme,
        )
        self.file_explorer = FileExplorer(
            self.ui.file_explorer,
            [
                lambda _: self.hierarchy.load_stage(),
                lambda _: self.selection_util.clear(),
                lambda _: self.blender_client.unsync(),
            ],
            self.selected_theme,
        )
        self.selection_util = SelectionUtil(
            self.ui.operators,
            self.selected_theme,
            [
                self.hierarchy.on_select_prim,
                lambda _, new: self.properties.select_prim(new),
            ],
        )
        self.blender_client = BlenderClient(
            self.ui.operators,
            lambda: self.selection_util.selection,
            lambda: self.file_explorer.stage,
            self.ui.on_tick,
            self.ui.on_end,
        )
        self.prim_util = PrimUtil(
            self.ui.operators,
            lambda: self.file_explorer.stage,
            lambda: self.selection_util.selection,
            [self.hierarchy.add_prim],
            [self.hierarchy.remove_prim, lambda _: self.properties.select_prim(None)],
        )
        self.schema_util = SchemaUtil(
            self.ui.operators,
            lambda: self.selection_util.selection,
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