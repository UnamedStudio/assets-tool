from __future__ import annotations
from collections.abc import Callable, Iterable
from copy import deepcopy
from dataclasses import dataclass
from importlib.resources import as_file
from math import ceil
import os
from pathlib import Path
from queue import Queue
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
from pxr.Gf import Vec3d, Quatd, Transform
from pxr.UsdGeom import (
    Mesh,
    Xformable,
    PrimvarsAPI,
    Cube,
    GetStageUpAxis,
    SetStageUpAxis,
)
from pxr.Tf import Type as UsdType
from pxr.Vt import IntArray, Vec3fArray

import screeninfo
import software_client
from software_client import Client, RunCommands

from assets_tool.utils import (
    Matrix4d,
    XformCache,
    copy_prim,
    from_usd_transform,
    is_attr_authored_in_layer,
    is_prim_authored_in_layer,
    none_str2none,
    quat,
    is_attr_blocked,
    relativize_sublayers,
    unique_path,
)

from assets_tool.utils import registry


class SelectorUI[T]:
    def __init__(
        self,
        label: str,
        get_selected_value: Callable[[], T | None],
        get_value: Callable[[], T | None],
        set_value: Callable[[T | None], None],
        value2text: Callable[[T | None], str],
        parent: int | str = 0,
    ):
        self.get_selected_value = get_selected_value
        self.get_value = get_value
        self.set_value = set_value
        self.value2text = value2text
        self.label_ui = dpg.add_text(label, parent=parent)
        with dpg.group(horizontal=True, parent=parent):
            dpg.add_button(label="clear", callback=self.clear)
            self.value_ui = dpg.add_button(
                label=self.value2text(self.get_value()), callback=self.select
            )

    def select(self):
        self.set_value(self.get_selected_value())
        self.update_ui()

    def clear(self):
        self.set_value(None)
        self.update_ui()

    def update_ui(self):
        value = self.get_value()
        dpg.configure_item(self.value_ui, label=self.value2text(value))


class Tree:
    class Node:
        is_open: bool
        fold_button: int | str
        button_ui: int | str
        children_ui: int | str
        root_ui: int | str

    def __init__(self, unfold_label: str = "+", fold_label="-") -> None:
        self.open_label = unfold_label
        self.close_label = fold_label

    def node(
        self,
        label: str,
        callback: Callable[[], None],
        parent: int | str,
        default_open: bool = False,
    ) -> Node:
        node = self.Node()
        node.is_open = default_open
        with dpg.group(parent=parent) as root_ui:
            with dpg.group(horizontal=True):
                node.fold_button = dpg.add_button(
                    label=self.close_label if default_open else self.open_label,
                    callback=self.toggle_open(node),
                )
                node.button_ui = dpg.add_button(label=label, callback=callback)
            node.children_ui = dpg.add_child_window(
                parent=parent, show=default_open, auto_resize_y=True
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
        self.old_operation_custom_data: dict[str, Any] | None = None

        self.selected_file_path: Path | None = None
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
            assert self.selected_file_path
            stem = self.selected_file_path.stem
            operation_name = dpg.get_value(self.operation_name_ui)
            root_layer = self.stage.GetRootLayer()
            reload = False
            if Layer.IsAnonymousLayerIdentifier(root_layer.identifier):
                edit_mode = dpg.get_value(self.edit_mode_ui)
                match edit_mode:
                    case "override replace" | "override":
                        if_replace = edit_mode == "override replace"
                        if self.old_operation_custom_data:
                            selected_operation_custom_data = (
                                self.old_operation_custom_data
                            )
                        else:
                            selected_operation_custom_data: dict[str, Any] = {
                                "operation": "",
                                "input": "",
                                "output": "",
                                "original_name": stem,
                            }
                        original_name = selected_operation_custom_data["original_name"]
                        path = unique_path(
                            self.selected_file_path.with_stem(
                                f"{original_name}-{operation_name}"
                                if if_replace
                                else f"{original_name}+{operation_name}"
                            )
                        )

                        # input layer
                        if if_replace:
                            input: str
                            if input := selected_operation_custom_data["input"]:
                                input_path = self.selected_file_path.parent / input
                                input_stage = Stage.Open(str(input_path))
                                input_root_layer = input_stage.GetRootLayer()
                                input_custom_layer_data = (
                                    input_root_layer.customLayerData
                                )
                                input_operation_custom_data = input_custom_layer_data[
                                    "assets_tool:operation"
                                ]
                                input_operation_custom_data["output"] = Path(
                                    os.path.relpath(
                                        str(path.resolve()),
                                        str(input_path.parent.resolve()),
                                    )
                                ).as_posix()
                                input_custom_layer_data["assets_tool:operation"] = (
                                    input_operation_custom_data
                                )
                                input_root_layer.customLayerData = (
                                    input_custom_layer_data
                                )
                                input_root_layer.Save()
                                del input_stage

                        # selected layer
                        selected_stage = Stage.Open(str(self.selected_file_path))
                        selected_root_layer = selected_stage.GetRootLayer()
                        custom_layer_data = selected_root_layer.customLayerData
                        operation_custom_data = deepcopy(selected_operation_custom_data)
                        operation_custom_data["output"] = Path(
                            os.path.relpath(
                                str(
                                    (
                                        self.selected_file_path if if_replace else path
                                    ).resolve()
                                ),
                                str(path.parent.resolve()),
                            )
                        ).as_posix()
                        custom_layer_data["assets_tool:operation"] = (
                            operation_custom_data
                        )
                        selected_root_layer.customLayerData = custom_layer_data
                        selected_stage.GetRootLayer().Save()
                        del selected_stage

                        # output layer
                        custom_layer_data = root_layer.customLayerData
                        operation_custom_data = deepcopy(selected_operation_custom_data)
                        operation_custom_data["input"] = os.path.relpath(
                            str(
                                (
                                    path if if_replace else self.selected_file_path
                                ).resolve()
                            ),
                            str(self.selected_file_path.parent.resolve()),
                        )
                        operation_custom_data["operation"] = operation_name
                        custom_layer_data["assets_tool:operation"] = (
                            operation_custom_data
                        )
                        root_layer.customLayerData = custom_layer_data
                        root_layer.subLayerPaths.clear()

                        # file
                        if if_replace:
                            self.selected_file_path.rename(path)
                        root_layer.subLayerPaths.append(
                            str(path if if_replace else self.selected_file_path)
                        )
                        root_layer.identifier = str(
                            self.selected_file_path if if_replace else path
                        )
                        relativize_sublayers(root_layer)
                        reload = True
            self.stage.Save()
            del self.stage
            if reload:
                self.load_path(self.selected_file_path.parent)()

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
        if self.selected_file_path:
            stage = Stage.Open(str(self.selected_file_path))
            metadata = stage.GetRootLayer().customLayerData.get("assets_tool:operation")
            if not metadata:
                return
            inputs = []
            metadata_iter = metadata
            while True:
                if input := metadata_iter["input"]:
                    path = self.selected_file_path.parent / Path(input)
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
                    path = self.selected_file_path.parent / Path(input)
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
        if self.selected_file_path:
            self.load_path(self.selected_file_path)

    def update_opened_file_ui(self):
        assert self.opened_directory_path
        for path, button in self.path2button.items():
            if self.selected_file_path and path == self.selected_file_path:
                dpg.bind_item_theme(button, self.opened_theme)
            else:
                dpg.bind_item_theme(button, 0)
        dpg.set_value(
            self.opened_file_ui,
            os.path.relpath(
                str(self.selected_file_path.resolve()),
                str(self.opened_directory_path.resolve()),
            )
            if self.selected_file_path
            else "",
        )

    def load_path(
        self,
        path: Path | None,
    ):
        def ret():
            if not path:
                self.selected_file_path = path
                self.stage = None
                self.old_operation_custom_data = None
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
                self.selected_file_path = path
                self.stage = None
                self.old_operation_custom_data = None
                self.update_operation_stack_ui()
                match path.suffix:
                    case ".usd" | ".usda" | ".usdc":
                        old_stage = Stage.Open(str(self.selected_file_path))
                        self.old_operation_custom_data = (
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
                                up_axis = GetStageUpAxis(stage)
                                del stage
                                root_layer.defaultPrim = defaultPrim
                                self.stage = Stage.Open(root_layer)
                                SetStageUpAxis(self.stage, up_axis)
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
        get_selection: Callable[[], SelectionUI.Selection],
        selected_theme: int | str,
        guide_theme: int | str,
        selected_and_guide_theme: int | str,
    ) -> None:
        self.container = container
        self.select_prim = select_prim
        self.get_stage = get_stage
        self.get_selection = get_selection
        self.selected_theme = selected_theme
        self.guide_theme = guide_theme
        self.selected_and_guide_theme = selected_and_guide_theme
        self.tree = Tree()
        self.selected_prim_ui = dpg.add_text(parent=self.container)
        self.tree_ui = dpg.add_child_window(parent=self.container)
        self.prim2node = dict[Prim, Tree.Node]()
        self.themed_uis = set[int | str]()

    def load_stage(self):
        self.prim2node.clear()
        self.themed_uis.clear()
        dpg.delete_item(self.tree_ui, children_only=True)
        if stage := self.get_stage():
            self.add_prim_raw(stage.GetPseudoRoot(), self.tree_ui, True)

    def load_prim(self, prim: Prim, node: Tree.Node):
        self.prim2node[prim] = node
        child: Prim
        for child in prim.GetChildren():
            self.add_prim_raw(child, node.children_ui)

    def add_prim_raw(
        self, prim: Prim, parent_ui: int | str, default_open: bool = False
    ):
        node = self.tree.node(
            prim.GetName(), lambda: self.select_prim(prim), parent_ui, default_open
        )
        self.load_prim(prim, node)

    def add_prim(self, prim: Prim):
        if prim in self.prim2node:
            return
        self.add_prim_raw(prim, self.prim2node[prim.GetParent()].children_ui)

    def remove_prim(self, prim: Prim):
        node = self.prim2node.pop(prim)
        self.themed_uis.discard(node.button_ui)
        dpg.delete_item(node.root_ui)

    def update_ui(self):
        for ui in self.themed_uis:
            dpg.bind_item_theme(ui, 0)
        self.themed_uis.clear()
        selection = self.get_selection()
        for selected in selection.selected:
            if node := self.prim2node.get(selected):
                dpg.bind_item_theme(node.button_ui, self.selected_theme)
                self.themed_uis.add(node.button_ui)
        if selection.guide:
            if node := self.prim2node.get(selection.guide):
                dpg.bind_item_theme(
                    node.button_ui,
                    self.selected_and_guide_theme
                    if selection.guide in selection.selected
                    else self.guide_theme,
                )
                self.themed_uis.add(node.button_ui)
        dpg.set_value(
            self.selected_prim_ui,
            str(selection.guide.GetPath()) if selection.guide else "",
        )


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
                    self.api_name2node[self.selected_api_name].button_ui, 0
                )
            self.selected_api_name = schema_name
            dpg.bind_item_theme(
                self.api_name2node[self.selected_api_name].button_ui,
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


class SelectionUI:
    @dataclass
    class Select:
        prim: Prim
        recursive: bool
        exclusive: bool
        name_filter: str | None
        type_filter: str | None
        api_filter: list[str]
        geomtry_filter: Prim | None

        @dataclass
        class PrepareFilter:
            xform_cache: XformCache | None = None
            geometries: list[tuple[NDArray[float], Matrix4d]] | None = None

        def prepare_filter(self) -> PrepareFilter:
            xform_cache = None
            geometries = None
            if self.geomtry_filter:
                xform_cache = XformCache()
                geometries = []

                for geometry in PrimRange(self.geomtry_filter):
                    if geometry.IsA(Cube):  # type: ignore
                        cube = Cube(geometry)
                        extent = (
                            array(
                                Transform(cube.GetLocalTransformation()).GetScale()
                                * cube.GetSizeAttr().Get()
                            )
                            / 2
                        )
                        world2local = xform_cache.GetLocalToWorldTransform(
                            geometry
                        ).GetInverse()
                        print("geometry", geometry)
                        print(
                            "local2world",
                            xform_cache.GetLocalToWorldTransform(geometry),
                        )
                        print("world2local", world2local)
                        geometries.append(
                            (
                                extent,
                                world2local,
                            )
                        )
            return self.PrepareFilter(xform_cache, geometries)

        def iter(self) -> Iterable[Prim]:
            prepare_filter = self.prepare_filter()

            def traverse_prim(prim: Prim) -> Iterable[Prim]:
                if self.if_filter(prim, prepare_filter):
                    yield prim
                child: Prim
                for child in prim.GetChildren():
                    yield from traverse_prim(child)

            if self.recursive:
                yield from traverse_prim(self.prim)
            else:
                yield self.prim

        def if_filter(self, prim: Prim, prepare_filter: PrepareFilter) -> bool:
            if self.name_filter:
                if prim.GetName() != self.name_filter:
                    return False
            if self.type_filter:
                if not prim.IsA(self.type_filter):
                    return False
            for api in self.api_filter:
                if not prim.HasAPI(api):
                    return False
            if self.geomtry_filter:
                assert (
                    prepare_filter.xform_cache and prepare_filter.geometries is not None
                )
                position = prepare_filter.xform_cache.GetLocalToWorldTransform(
                    prim
                ).ExtractTranslation()
                print("prim", prim)
                print("position", position)
                inside = False
                for extend, xform in prepare_filter.geometries:
                    local_position = array(xform.Transform(position))
                    print("local_position", local_position)
                    print("extend", extend)
                    if numpy.all(abs(local_position) <= abs(extend)):
                        inside = True
                        break
                print("inside", inside)
                if not inside:
                    return False

            return True

    @dataclass
    class Selection:
        selected: dict[Prim, SelectionUI.Select]
        guide: Prim | None = None

        def iter(self) -> Iterable[Prim]:
            excludes = set[Prim]()
            for select in self.selected.values():
                if select.exclusive:
                    for prim in select.iter():
                        excludes.add(prim)
            for select in self.selected.values():
                if not select.exclusive:
                    for prim in select.iter():
                        if prim not in excludes:
                            yield prim

    def __init__(
        self,
        container: int | str,
        selected_theme: int | str,
        on_select: list[Callable[[], None]],
        get_stage: Callable[[], Stage | None],
    ) -> None:
        self.container = container
        self.selection = self.Selection({}, None)
        self.selected_theme = selected_theme
        self.on_select = on_select
        self.get_stage = get_stage
        self.editing_select: Prim | None = None

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
            dpg.add_text("Selection")
            self.mode_ui = dpg.add_combo(
                ("single", "multi", "guide"),
                label="mode",
                default_value="single",
            )
            dpg.add_button(label="clear", callback=self.clear)
            with dpg.group(horizontal=True):

                def on_recursive_ui(sender, app_data, user_data):
                    self.on_recursive(app_data)
                    if select := self.get_editing_select():
                        select.recursive = app_data

                def on_exclusive_ui(sender, app_data, user_data):
                    if select := self.get_editing_select():
                        select.exclusive = app_data

                self.recursive_ui = dpg.add_checkbox(
                    label="recursive", default_value=True, callback=on_recursive_ui
                )
                self.exclusive_ui = dpg.add_checkbox(
                    label="exclusive", default_value=False, callback=on_exclusive_ui
                )
            with dpg.tree_node(label="filters") as self.filters_ui:
                self.name_filter_ui = dpg.add_input_text(label="name")

                def on_type_filter_ui(sender, app_data, user_data):
                    if select := self.get_editing_select():
                        if app_data == "None":
                            select.type_filter = None
                        else:
                            select.type_filter = app_data

                self.type_filter_ui = dpg.add_combo(
                    type_names, label="type", callback=on_type_filter_ui
                )

                def on_api_filter_ui(sender, app_data, user_data):
                    if select := self.get_editing_select():
                        if app_data == "None":
                            select.api_filter = []
                        else:
                            select.api_filter = [app_data]

                self.api_filter_ui = dpg.add_combo(
                    api_names, label="api", callback=on_api_filter_ui
                )

                self.geometry_filter: Prim | None = None

                def get_geometry_filter():
                    if select := self.get_editing_select():
                        self.geometry_filter = select.geomtry_filter
                        return select.geomtry_filter
                    else:
                        self.geometry_filter

                def set_geometry_filter(value: Prim | None):
                    if select := self.get_editing_select():
                        select.geomtry_filter = value
                    self.geometry_filter = value

                def get_selected_geometry_filter():
                    return self.selection.guide

                def geometry_filter2text(value: Prim | None):
                    if value:
                        return str(value.GetPath())
                    else:
                        return "None"

                self.geometry_filter_ui = SelectorUI(
                    "geometry",
                    get_selected_geometry_filter,
                    get_geometry_filter,
                    set_geometry_filter,
                    geometry_filter2text,
                )
            self.selected_ui = dpg.add_tree_node(label="selected")

    def get_editing_select(self) -> SelectionUI.Select | None:
        if self.editing_select:
            return self.selection.selected.get(self.editing_select)
        return None

    def on_recursive(self, value: bool):
        dpg.configure_item(self.filters_ui, show=value)

    def clear(self):
        self.selection.selected.clear()
        self.select(None)
        self.update_selected_ui()

    def update_selected_ui(self):
        dpg.delete_item(self.selected_ui, children_only=True)

        def add_selected_ui(prim: Prim, select: SelectionUI.Select):
            with dpg.group(horizontal=True, parent=self.selected_ui):

                def on_button():
                    self.editing_select = prim
                    if select := self.selection.selected.get(self.editing_select):
                        dpg.set_value(self.recursive_ui, select.recursive)
                        self.on_recursive(select.recursive)
                        dpg.set_value(self.exclusive_ui, select.exclusive)

                        dpg.set_value(self.name_filter_ui, select.name_filter or "")
                        dpg.set_value(self.type_filter_ui, select.type_filter or "None")
                        dpg.set_value(
                            self.api_filter_ui,
                            select.api_filter[0]
                            if len(select.api_filter) > 0
                            else "None",
                        )
                        self.geometry_filter_ui.update_ui()

                    self.update_selected_ui()

                selected_ui = dpg.add_button(
                    label=("^" if select.exclusive else "") + str(prim.GetPath()),
                    callback=on_button,
                )
                dpg.bind_item_theme(
                    selected_ui,
                    self.selected_theme if prim == self.editing_select else 0,
                )

        for prim, info in self.selection.selected.items():
            add_selected_ui(prim, info)

    def select(self, prim: Prim | None):
        self.selection.guide = prim
        if prim:
            mode = dpg.get_value(self.mode_ui)
            match mode:
                case "single" | "multi":
                    if mode == "single":
                        self.selection.selected.clear()
                    if prim:
                        if prim in self.selection.selected:
                            self.selection.selected.pop(prim, None)
                        else:
                            api_filter = none_str2none(
                                dpg.get_value(self.api_filter_ui)
                            )
                            self.selection.selected[prim] = self.Select(
                                prim,
                                dpg.get_value(self.recursive_ui),
                                dpg.get_value(self.exclusive_ui),
                                none_str2none(dpg.get_value(self.name_filter_ui)),
                                none_str2none(dpg.get_value(self.type_filter_ui)),
                                [api_filter] if api_filter else [],
                                self.geometry_filter,
                            )
                            self.editing_select = prim
                case "guide":
                    self.selection.guide = prim
        self.update_selected_ui()
        for callback in self.on_select:
            callback()


class BlenderClient:
    class SyncedMesh:
        mesh: Mesh

    class Synced:
        def __init__(self) -> None:
            self.meshes = dict[Path, BlenderClient.SyncedMesh]()

    def __init__(
        self,
        container: int | str,
        get_selection: Callable[[], SelectionUI.Selection],
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
        prims = set(selection.iter())
        referenced_prims = set[Prim]()
        for prim in prims:
            ancestor = prim
            while True:
                if ancestor.IsPseudoRoot() or ancestor in referenced_prims:
                    break
                referenced_prims.add(ancestor)
                ancestor = ancestor.GetParent()
        stage = self.get_stage()
        xform_cache = XformCache()
        assert stage and xform_cache
        self.synced = self.Synced()
        command_count = 0
        for prim in referenced_prims:
            in_selection = prim in prims
            path = Path(str(prim.GetPath()))
            if in_selection:
                if prim.IsA(Mesh):  # type: ignore
                    mesh = Mesh(prim)
                    synced_mesh = BlenderClient.SyncedMesh()
                    self.synced.meshes[path] = synced_mesh
                    face_vertex_counts = array(mesh.GetFaceVertexCountsAttr().Get())
                    assert numpy.all(face_vertex_counts == 3)
                    software_client.create_mesh(
                        self.client,
                        array(mesh.GetPointsAttr().Get()),
                        array(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3),
                        path,
                        dpg.get_value(self.if_sync_mesh_ui),
                    )
                    command_count += 4
                elif prim.IsA(Cube):  # type: ignore
                    cube = Cube(prim)
                    software_client.create_cube(
                        self.client, cube.GetSizeAttr().Get(), path
                    )

            if prim.IsA(Xformable):  # type: ignore
                translation, rotation, scale = from_usd_transform(
                    xform_cache.GetLocalTransformation(prim)[0]
                )
                software_client.set_xform(
                    self.client,
                    translation,
                    rotation,
                    scale,
                    path,
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
                stage = self.get_stage()
                assert stage
                prim = stage.GetPrimAtPath(path.as_posix())
                mesh = Mesh(prim)
                mesh.GetPointsAttr().Set(Vec3fArray.FromNumpy(positions))
                mesh.GetFaceVertexIndicesAttr().Set(IntArray.FromNumpy(indices))
                mesh.GetFaceVertexCountsAttr().Set(
                    IntArray.FromNumpy(numpy.full(len(indices) // 3, 3))
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
                stage = self.get_stage()
                assert stage
                prim = stage.GetPrimAtPath(path.as_posix())
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
        get_selection: Callable[[], SelectionUI.Selection],
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
        stage = self.get_stage()
        assert stage
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
        path_iter = path.AppendChild(dpg.get_value(self.name_ui))
        count = 0
        while stage.GetPrimAtPath(path_iter):
            count += 1
            path_iter = path.AppendChild(dpg.get_value(self.name_ui) + str(count))
        return path_iter

    def create_prim(self):
        stage = self.get_stage()
        assert stage
        for prim in list(self.get_selection().iter()):
            path = self.get_create_path(prim)
            prim = stage.DefinePrim(path)
            for callback in self.on_add_prim:
                callback(prim)

    def delete_prim(self):
        stage = self.get_stage()
        assert stage
        for prim in list(self.get_selection().iter()):
            if prim:
                for callback in self.on_delete_prim:
                    callback(prim)
                path = prim.GetPath()
                if is_prim_authored_in_layer(prim, stage.GetRootLayer()):
                    stage.RemovePrim(path)
                else:
                    prim.SetActive(False)

    def copy_prim(self):
        if prim := self.get_selection().guide:
            self.copied_prim = prim

    def paste_prim(self):
        current_prim = self.get_selection().guide
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
        get_selection: Callable[[], SelectionUI.Selection],
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
            up_axis = GetStageUpAxis(stage)
            root_layer.Clear()
            if operation_metadata:
                custom_layer_data = root_layer.customLayerData
                custom_layer_data["assets_tool:operation"] = operation_metadata
                root_layer.customLayerData = custom_layer_data
            root_layer.defaultPrim = default_prim
            for sublayer in sublayers:
                root_layer.subLayerPaths.append(sublayer)
            SetStageUpAxis(stage, up_axis)
            for callback in self.on_clear:
                callback()


class FileUtil:
    def __init__(
        self, container: int | str, get_file_path: Callable[[], Path | None]
    ) -> None:
        self.container = container
        self.get_file_path = get_file_path
        with dpg.child_window(auto_resize_y=True, parent=container):
            dpg.add_button(label="to readable", callback=self.to_readable)

    def to_readable(self):
        if file_path := self.get_file_path():
            Stage.Open(str(file_path)).Export(str(file_path.with_suffix(".usda")))


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
    def __init__(self, font_scale: float = 1.0):
        dpg.create_context()
        from importlib.resources import files

        font_path = files("assets_tool.assets").joinpath("fonts/ARIAL.TTF")
        max_height = 600
        for i in screeninfo.get_monitors():
            if i.height > max_height:
                max_height = i.height
        with dpg.font_registry():
            with as_file(font_path) as path:
                default_font = dpg.add_font(
                    str(path), ceil(max_height * 0.012 * font_scale)
                )
                dpg.bind_font(default_font)
        with dpg.theme() as self.selected_theme:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(
                    dpg.mvThemeCol_Button, (120, 120, 0), category=dpg.mvThemeCat_Core
                )
        with dpg.theme() as self.guide_theme:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(
                    dpg.mvThemeCol_Button, (30, 90, 120), category=dpg.mvThemeCat_Core
                )
        with dpg.theme() as self.selected_and_guide_theme:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(
                    dpg.mvThemeCol_Button, (120, 90, 30), category=dpg.mvThemeCat_Core
                )
        self.ui = UI()
        self.properties = Properties(
            self.ui.properties, lambda: self.file_explorer.stage, self.selected_theme
        )
        self.hierarchy = Hierarchy(
            self.ui.heirarchy,
            lambda prim: self.selection_ui.select(prim),
            lambda: self.file_explorer.stage,
            lambda: self.selection_ui.selection,
            self.selected_theme,
            self.guide_theme,
            self.selected_and_guide_theme,
        )
        self.file_explorer = FileExplorer(
            self.ui.file_explorer,
            [
                lambda _: self.hierarchy.load_stage(),
                lambda _: self.selection_ui.clear(),
                lambda _: self.blender_client.unsync(),
            ],
            self.selected_theme,
        )
        self.selection_ui = SelectionUI(
            self.ui.operators,
            self.selected_theme,
            [
                self.hierarchy.update_ui,
                lambda: self.properties.select_prim(self.selection_ui.selection.guide),
            ],
            lambda: self.file_explorer.stage,
        )
        self.blender_client = BlenderClient(
            self.ui.operators,
            lambda: self.selection_ui.selection,
            lambda: self.file_explorer.stage,
            self.ui.on_tick,
            self.ui.on_end,
        )
        self.prim_util = PrimUtil(
            self.ui.operators,
            lambda: self.file_explorer.stage,
            lambda: self.selection_ui.selection,
            [self.hierarchy.add_prim],
            [self.hierarchy.remove_prim, lambda _: self.properties.select_prim(None)],
        )
        self.schema_util = SchemaUtil(
            self.ui.operators,
            lambda: self.selection_ui.selection,
            lambda: self.properties.selected_api_name,
            [self.properties.select_prim],
        )
        self.layer_util = LayerUtil(
            self.ui.operators,
            lambda: self.file_explorer.stage,
            [self.hierarchy.load_stage],
        )
        self.file_util = FileUtil(
            self.ui.operators,
            lambda: self.file_explorer.selected_file_path,
        )
        self.file_explorer.load_path(Path(".").resolve())()

    def run(self):
        self.ui.run()


if __name__ == "__main__":
    app = App()
    app.run()
