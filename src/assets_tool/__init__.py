from __future__ import annotations
from collections.abc import Callable, Iterable
from copy import deepcopy
from dataclasses import dataclass
from importlib.resources import as_file
from math import ceil, exp, exp2, log2
import os
from pathlib import Path
from pprint import pprint
from queue import Queue
from time import sleep
from typing import Any
import dearpygui.dearpygui as dpg

from numpy import array, float32, float64, int32
import numpy
from numpy.typing import NDArray
from pxr.Usd import (
    Stage,
    Prim,
    SchemaRegistry,
    Attribute,
    Property,
    Relationship,
    PrimRange,
)
from pxr.Sdf import Layer, Path as UsdPath, ComputeAssetPathRelativeToLayer
from pxr.Gf import Vec3d, Quatd, Transform
from pxr.UsdGeom import (
    Mesh,
    Xformable,
    PrimvarsAPI,
    Cube,
    Cylinder,
    GetStageUpAxis,
    SetStageUpAxis,
)
from pxr.Tf import Type as UsdType
from pxr.Vt import IntArray, Vec3fArray
from pxr.UsdPhysics import CollisionAPI

import screeninfo
import software_client
from software_client import Client, RunCommands
import software_client.command

from assets_tool.utils import (
    Lazy,
    Matrix4d,
    Ref,
    Scope,
    XformCache,
    copy_api,
    copy_prim,
    find_usd_dependencies,
    from_usd_transform,
    is_attr_authored_in_layer,
    is_prim_authored_in_layer,
    none_str2none,
    overwrite,
    quat,
    is_attr_blocked,
    relativize_sublayers,
    unique_path,
    unique_usd_path,
)

from assets_tool.utils import registry


class SelectorUI[T]:
    def __init__(
        self,
        get_selected_value: Callable[[], T | None],
        get_value: Callable[[], T | None],
        set_value: Callable[[T | None], None],
        value2text: Callable[[T | None], str],
        label: str | None = None,
        parent: int | str = 0,
    ):
        self.get_selected_value = get_selected_value
        self.get_value = get_value
        self.set_value = set_value
        self.value2text = value2text
        self.label_ui = dpg.add_text(label, parent=parent) if label else None
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


class TreeUI:
    class Node:
        def __init__(
            self,
            is_open: bool,
            fold_button: int | str,
            button_ui: int | str,
            children_ui: int | str,
            root_ui: int | str,
            group_ui: int | str,
            on_first_open: Callable[[], None] | None,
        ):
            self.is_open = is_open
            self.fold_button = fold_button
            self.button_ui = button_ui
            self.children_ui = children_ui
            self.root_ui = root_ui
            self.group_ui = group_ui
            self.on_first_open = on_first_open

    def __init__(
        self,
        parent: int | str = 0,
        unfold_label: str = "+",
        fold_label="-",
        label: str | None = None,
    ) -> None:
        self.open_label = unfold_label
        self.close_label = fold_label
        with dpg.group(parent=parent) as self.root_ui:
            if label:
                self.label_ui = dpg.add_text(label)
            else:
                self.label_ui = None
            self.children_ui = dpg.add_child_window(auto_resize_y=True)

    def node(
        self,
        label: str,
        callback: Callable[[], None] | None = None,
        parent: Node | None = None,
        on_first_open: Callable[[], None] | None = None,
    ) -> Node:
        parent_ui = parent.children_ui if parent else self.children_ui
        with dpg.group(parent=parent_ui) as root_ui:
            with dpg.group(horizontal=True) as group_ui:
                fold_button = dpg.add_button(label=self.open_label)
                button_ui = dpg.add_button(label=label, callback=callback)  # type: ignore
            children_ui = dpg.add_child_window(show=False, auto_resize_y=True)
        ret = self.Node(
            False, fold_button, button_ui, children_ui, root_ui, group_ui, on_first_open
        )
        dpg.configure_item(fold_button, callback=lambda: self.toggle_open(ret))
        return ret

    def toggle_open(self, node: Node):
        if node.is_open:
            node.is_open = False
            dpg.configure_item(node.children_ui, show=False)
            dpg.configure_item(node.fold_button, label=self.open_label)
        else:
            node.is_open = True
            on_first_open = node.on_first_open
            if on_first_open:
                node.on_first_open = None
                on_first_open()
            dpg.configure_item(node.children_ui, show=True)
            dpg.configure_item(node.fold_button, label=self.close_label)

    def clear(self):
        dpg.delete_item(self.children_ui, children_only=True)


class FileExplorer:
    @dataclass
    class LoadedStage:
        stage: Ref[Stage]
        dirty: bool

    def __init__(
        self,
        parent: int | str,
        get_selection: Callable[[], SelectionUI.Selection],
        select_stage: list[Callable[[Path | None], None]],
        selected_theme: int | str,
        guide_theme: int | str,
        selected_and_guide_theme: int | str,
    ) -> None:
        self.get_selection = get_selection
        self.select_stage = select_stage
        self.selected_theme = selected_theme
        self.guide_theme = guide_theme
        self.selected_and_guide_theme = selected_and_guide_theme
        self.stage: Stage | None = None

        self.selected_file_path: Path | None = None
        self.opened_directory_path: Path | None = None
        self.editing_stages = dict[Path, FileExplorer.LoadedStage]()

        self.opened_file_ui = dpg.add_text(parent=parent)
        self.path2button = dict[Path, int | str]()
        self.path2node = dict[Path, TreeUI.Node]()
        self.themed_uis = set[int | str]()

        self.edit_mode_ui = dpg.add_combo(
            ["update", "override", "override replace"],
            label="edit mode",
            default_value="override replace",
            callback=self.update_operation_name_ui,
            parent=parent,
        )
        self.operation_name_ui = dpg.add_input_text(
            label="operation name",
            default_value="none",
            parent=parent,
        )
        dpg.add_button(label="save", callback=self.save, parent=parent)
        self.operation_stack_ui = dpg.add_tree_node(
            label="operation stack", parent=parent
        )
        self.editing_stages_ui = dpg.add_tree_node(
            label="editing stages", parent=parent
        )
        self.tree_ui = TreeUI(parent=parent)

    def update_editing_stages_ui(self):
        if not self.opened_directory_path:
            return
        dpg.delete_item(self.editing_stages_ui, children_only=True)
        droppeds = set[Path]()
        for path, editing_stage in self.editing_stages.items():
            if stage := editing_stage.stage():

                def callback(path: Path):
                    self.load_path(path)

                dpg.add_button(
                    label=("*" if editing_stage.dirty else "")
                    + os.path.relpath(path.resolve(), self.opened_directory_path),
                    callback=Scope(callback)(path),
                    parent=self.editing_stages_ui,
                )
            else:
                droppeds.add(path)
        for dropped in droppeds:
            self.editing_stages.pop(dropped)

    def make_dirty(self, path: Path, dirty=True):
        editing_stage = self.editing_stages[path]
        editing_stage.dirty = dirty
        editing_stage.stage.make_strong(dirty)

        self.update_editing_stages_ui()

    def save(self):
        def save_stage(stage: Stage, current_path: Path) -> None:
            if not self.editing_stages[current_path].dirty:
                return
            root_layer = stage.GetRootLayer()
            stem = current_path.stem
            operation_name = dpg.get_value(self.operation_name_ui)
            if Layer.IsAnonymousLayerIdentifier(root_layer.identifier):
                edit_mode = dpg.get_value(self.edit_mode_ui)
                match edit_mode:
                    case "override replace" | "override":
                        if_replace = edit_mode == "override replace"
                        old_stage = Stage.Open(str(current_path))
                        old_operation_custom_data: dict[str, Any] | None = (
                            old_stage.GetRootLayer().customLayerData.get(
                                "assets_tool:operation"
                            )
                        )
                        del old_stage
                        if not old_operation_custom_data:
                            old_operation_custom_data = {
                                "operation": "",
                                "input": "",
                                "output": "",
                                "original_name": stem,
                            }
                        original_name = old_operation_custom_data["original_name"]
                        path = unique_path(
                            current_path.with_stem(
                                f"{original_name}-{operation_name}"
                                if if_replace
                                else f"{original_name}+{operation_name}"
                            )
                        )

                        # input layer
                        if if_replace:
                            input: str
                            if input := old_operation_custom_data["input"]:
                                input_path = current_path.parent / input
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
                        selected_stage = Stage.Open(str(current_path))
                        selected_root_layer = selected_stage.GetRootLayer()
                        custom_layer_data = selected_root_layer.customLayerData
                        operation_custom_data = deepcopy(old_operation_custom_data)
                        operation_custom_data["output"] = Path(
                            os.path.relpath(
                                str((current_path if if_replace else path).resolve()),
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
                        operation_custom_data = deepcopy(old_operation_custom_data)
                        operation_custom_data["input"] = os.path.relpath(
                            str((path if if_replace else current_path).resolve()),
                            str(current_path.parent.resolve()),
                        )
                        operation_custom_data["operation"] = operation_name
                        custom_layer_data["assets_tool:operation"] = (
                            operation_custom_data
                        )
                        root_layer.customLayerData = custom_layer_data

                        # file
                        if if_replace:
                            root_layer.subLayerPaths.clear()
                            current_path.rename(path)
                            root_layer.subLayerPaths.append(str(path))
                            relativize_sublayers(current_path, root_layer)
                        root_layer_path = current_path if if_replace else path
                        overwrite(root_layer, root_layer_path)
                        self.on_add_path(path)
                    case _:
                        raise Exception()
            else:
                assert root_layer.Save()
            self.editing_stages.pop(current_path, None)
            self.update_editing_stages_ui()
            del stage

        selection = self.get_selection()
        if len(selection.selects) > 0:
            if self.stage:
                assert self.selected_file_path
                save_stage(self.stage, self.selected_file_path)
                self.load_path(None)
        else:
            for path, stage, _ in selection.stage_iter():
                save_stage(stage, path)
            self.load_path(None)

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
                        callback=Scope(self.load_path)(path),
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
                    dpg.add_button(
                        label=operation_name,
                        callback=Scope(self.load_path)(path),
                    )

    def reload_file(self):
        if self.selected_file_path:
            self.load_path(self.selected_file_path)

    def update_opened_file_ui(self):
        assert self.opened_directory_path
        selection = self.get_selection()
        for ui in self.themed_uis:
            dpg.bind_item_theme(ui, 0)
        self.themed_uis.clear()
        if selection.stage_guide:
            if button := self.path2button.get(selection.stage_guide):
                dpg.bind_item_theme(button, self.guide_theme)
                self.themed_uis.add(button)
        for path in selection.stage_selects:
            is_guide = (
                selection.stage_guide is not None
                and path.resolve() == selection.stage_guide.resolve()
            )
            if button := self.path2button.get(path):
                if is_guide:
                    dpg.bind_item_theme(button, self.selected_and_guide_theme)
                else:
                    dpg.bind_item_theme(button, self.selected_theme)
                self.themed_uis.add(button)
        dpg.set_value(
            self.opened_file_ui,
            os.path.relpath(
                str(self.selected_file_path.resolve()),
                str(self.opened_directory_path.resolve()),
            )
            if self.selected_file_path
            else "",
        )

    def on_add_path(self, path: Path):
        if node := self.path2node.get(path.parent):
            self.on_add_path_raw(path, node)
        elif (
            self.opened_directory_path is not None
            and path.parent.resolve() == self.opened_directory_path.resolve()
        ):
            self.on_add_path_raw(path, None)

    def on_add_path_raw(self, path: Path, parent: TreeUI.Node | None = None):
        if parent and parent.on_first_open:
            return
        raw_parent = parent.children_ui if parent else self.tree_ui.children_ui

        def on_select():
            for callback in self.select_stage:
                callback(path)
            self.update_opened_file_ui()

        if path.is_dir():
            node = self.tree_ui.node(
                path.name,
                Scope(self.load_path)(path),
                parent,
            )

            def on_first_open():
                self.on_add_paths(path, node)

            node.on_first_open = on_first_open

            dpg.add_button(
                label=" ",
                parent=node.group_ui,
                callback=on_select,
                before=node.fold_button,
            )
            button = node.button_ui
            self.path2node[path] = node
        elif path.is_file():
            with dpg.group(horizontal=True, parent=raw_parent):
                dpg.add_button(label=" ", callback=on_select)
                button = dpg.add_button(
                    label=path.name, callback=Scope(self.load_path)(path)
                )
        else:
            button = None
        if button:
            self.path2button[path] = button

    def on_add_paths(self, path: Path, parent: TreeUI.Node | None = None):
        for child in sorted(path.iterdir()):
            self.on_add_path_raw(child, parent)

    def load_path(
        self,
        path: Path | None,
    ):
        if not path:
            self.selected_file_path = path
            self.stage = None
            self.update_operation_stack_ui()
            for callback in self.select_stage:
                callback(path)
        elif path.is_dir():
            self.opened_directory_path = path
            self.path2button.clear()
            self.themed_uis.clear()
            self.path2node.clear()
            self.tree_ui.clear()
            dpg.add_button(
                label="..",
                callback=Scope(self.load_path)(path.parent),
                parent=self.tree_ui.children_ui,
            )
            self.on_add_paths(path)
        elif path.is_file():
            self.selected_file_path = path
            self.stage = None
            self.update_operation_stack_ui()
            match path.suffix:
                case ".usd" | ".usda" | ".usdc":
                    self.update_operation_name_ui()
                    self.stage = self.load_stage(self.selected_file_path)

                case _:
                    raise Exception(f"unsupported file {path}")
            for callback in self.select_stage:
                callback(path)
        else:
            raise Exception()
        self.update_opened_file_ui()

    def load_stage(self, path: Path) -> Stage:
        stage = None
        if prev_stage := self.editing_stages.get(path):
            if stage := prev_stage.stage():
                anonymous = Layer.IsAnonymousLayerIdentifier(
                    stage.GetRootLayer().identifier
                )
        match dpg.get_value(self.edit_mode_ui):
            case "update":
                if stage and not anonymous:  # type: ignore
                    return stage
                stage = Stage.Open(filePath=str(path))
            case "override" | "override replace":
                if stage and anonymous:  # type: ignore
                    return stage
                root_layer = Layer.CreateAnonymous()
                root_layer.subLayerPaths.append(str(path))
                stage = Stage.Open(str(path))
                defaultPrim = stage.GetRootLayer().defaultPrim
                up_axis = GetStageUpAxis(stage)
                del stage
                root_layer.defaultPrim = defaultPrim
                stage = Stage.Open(root_layer)
                SetStageUpAxis(stage, up_axis)
            case _:
                raise Exception()
        self.editing_stages[path] = self.LoadedStage(Ref(stage), False)
        self.update_editing_stages_ui()
        return stage


class Hierarchy:
    def __init__(
        self,
        select_prim: Callable[[Prim], None],
        get_stage: Callable[[], Stage | None],
        get_selection: Callable[[], SelectionUI.Selection],
        on_remove_prim: list[Callable[[Prim], None]],
        selected_theme: int | str,
        guide_theme: int | str,
        selected_and_guide_theme: int | str,
        parent: int | str = 0,
    ) -> None:
        self.container = parent
        self.select_prim = select_prim
        self.get_stage = get_stage
        self.get_selection = get_selection
        self._on_remove_prim = on_remove_prim
        self.selected_theme = selected_theme
        self.guide_theme = guide_theme
        self.selected_and_guide_theme = selected_and_guide_theme
        self.selected_prim_ui = dpg.add_text(parent=self.container)
        self.tree_ui = TreeUI(parent=parent)
        self.prim2node = dict[Prim, TreeUI.Node]()
        self.themed_uis = set[int | str]()

    def load_stage(self):
        self.prim2node.clear()
        self.themed_uis.clear()
        self.tree_ui.clear()
        if stage := self.get_stage():
            self.add_prim_raw(stage.GetPseudoRoot(), None, defalt_open=True)

    def load_prim(self, prim: Prim, node: TreeUI.Node):
        child: Prim
        for child in prim.GetChildren():
            self.add_prim_raw(child, node)

    def add_prim_raw(
        self, prim: Prim, parent: TreeUI.Node | None, defalt_open: bool = False
    ):
        node = self.tree_ui.node(
            prim.GetName(),
            lambda: self.select_prim(prim),
            parent,
            on_first_open=lambda: self.load_prim(prim, node),
        )
        self.prim2node[prim] = node
        if defalt_open:
            self.tree_ui.toggle_open(node)

    def on_add_prim(self, prim: Prim):
        if not prim:
            return
        if prim in self.prim2node:
            return
        if node := self.prim2node.get(prim.GetParent()):
            if not node.on_first_open:
                self.add_prim_raw(prim, node)
        else:
            self.on_add_prim(prim.GetParent())

    def on_remove_prim(self, prim: Prim):
        for callback in self._on_remove_prim:
            callback(prim)
        if node := self.prim2node.pop(prim, None):
            dpg.delete_item(node.root_ui)
            self.themed_uis.discard(node.button_ui)

    def update_ui(self):
        for ui in self.themed_uis:
            dpg.bind_item_theme(ui, 0)
        self.themed_uis.clear()
        selection = self.get_selection()
        for selected in selection.selects:
            if node := self.prim2node.get(selected):
                dpg.bind_item_theme(node.button_ui, self.selected_theme)
                self.themed_uis.add(node.button_ui)
        if selection.guide:
            if node := self.prim2node.get(selection.guide):
                dpg.bind_item_theme(
                    node.button_ui,
                    self.selected_and_guide_theme
                    if selection.guide in selection.selects
                    else self.guide_theme,
                )
                self.themed_uis.add(node.button_ui)
        dpg.set_value(
            self.selected_prim_ui,
            str(selection.guide.GetPath()) if selection.guide else "",
        )


class PropertiesUI:
    def __init__(
        self,
        get_stage: Callable[[], Stage | None],
        get_selected_file_path: Callable[[], Path | None],
        get_selection: Callable[[], SelectionUI.Selection],
        make_dirty: Callable[[Path, bool]],
        select_type: Callable[[str], None],
        select_api: Callable[[str], None],
        parent: int | str = 0,
    ) -> None:
        self.get_stage = get_stage
        self.get_selected_file_path = get_selected_file_path
        self.get_selection = get_selection
        self.make_dirty = make_dirty
        self.select_type = select_type
        self.select_api = select_api
        self.editing_prim: Prim | None = None
        self.meta_data_ui = TreeUI(parent=parent)
        self.type_ui = TreeUI(parent=parent, label="type")
        self.api_ui = TreeUI(parent=parent, label="api")
        self.none_apply_api_ui = TreeUI(parent=parent, label="none apply api")
        self.other_ui = TreeUI(parent=parent, label="other")

    def on_remove_prim(self, prim: Prim):
        if prim == self.editing_prim:
            self.select_prim(None)

    def select_prim(self, prim: Prim | None):
        self.editing_prim = prim
        self.meta_data_ui.clear()
        self.type_ui.clear()
        self.api_ui.clear()
        self.none_apply_api_ui.clear()
        self.other_ui.clear()
        if not prim:
            return

        specs = prim.GetPrimStack()

        meta_data_ui = self.meta_data_ui.node("meta data")
        with dpg.group(parent=meta_data_ui.children_ui):
            dpg.add_text("reference")
            with dpg.child_window(auto_resize_y=True):
                for spec in specs:
                    path = spec.layer.identifier
                    if selected_file_path := self.get_selected_file_path():
                        path = os.path.relpath(
                            spec.layer.identifier, selected_file_path.resolve()
                        )
                    dpg.add_text(path)

        raw_property_names = set(prim.GetPropertyNames())

        def add_schema_ui(
            schema_name: str,
            property_names: Iterable[str],
            on_select: Callable[[str], None],
            ui: TreeUI,
        ):
            node = ui.node(schema_name, lambda: on_select(schema_name), None)
            for property_name in property_names:
                if raw_property_names is not property_names:
                    raw_property_names.discard(property_name)
                self.property_ui(prim, property_name, parent=node.children_ui)
                if property_name == "xformOpOrder":
                    xform_ops = prim.GetAttribute(property_name).Get()
                    if xform_ops:
                        for xform_op in xform_ops:
                            name = str(xform_op)
                            if raw_property_names is not property_names:
                                raw_property_names.discard(name)
                            self.property_ui(prim, name, parent=node.children_ui)

        if type_properties := self.get_type_properties(prim):
            type_name, property_names = type_properties
            add_schema_ui(type_name, property_names, self.select_type, self.type_ui)

        for api_name, property_names in self.get_api_properties(prim):
            add_schema_ui(api_name, property_names, self.select_api, self.api_ui)

        for api_name, property_names in self.get_none_apply_api_properties(prim):
            add_schema_ui(
                api_name, property_names, self.select_api, self.none_apply_api_ui
            )
        if len(raw_property_names) > 0:
            add_schema_ui("__raw__", raw_property_names, lambda _: None, self.other_ui)

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

                def get_selected_props() -> Iterable[Property]:
                    selection = self.get_selection()
                    for path, stage, prims in selection.stage_iter():
                        has = False
                        for prim in prims:
                            if selected_prop := prim.GetProperty(prop.GetName()):
                                has = True
                                yield selected_prop
                        if has:
                            self.make_dirty(path, True)

                def set_value(value):
                    for selected_prop in get_selected_props():
                        assert isinstance(selected_prop, Attribute)
                        selected_prop.Set(value)

                def on_edit(sender, app_data, user_data):
                    set_value(app_data)

                if type_name == "token[]" or type_name == "token":
                    allowed_tokens = prop.GetMetadata("allowedTokens")
                    if type_name == "token":
                        if allowed_tokens:
                            edit_ui = dpg.add_combo(
                                tuple(str(x) for x in allowed_tokens),
                                default_value=str(value),
                                enabled=is_authored,
                                callback=on_edit,
                            )
                        else:
                            edit_ui = dpg.add_input_text(
                                default_value=value,
                                enabled=is_authored,
                                callback=on_edit,
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
                            edit_ui = dpg.add_checkbox(
                                default_value=value,
                                enabled=is_authored,
                                callback=on_edit,
                            )
                        case "int":
                            edit_ui = dpg.add_input_int(
                                default_value=value,
                                callback=on_edit,
                                enabled=is_authored,
                            )
                        case "float" | "double":
                            edit_ui = dpg.add_input_float(
                                default_value=value,
                                callback=on_edit,
                                enabled=is_authored,
                            )
                        case "vector3f" | "point3f" | "float3" | "double3":

                            def on_edit3(sender, app_data, user_data):
                                set_value(Vec3d(app_data[0], app_data[1], app_data[2]))

                            edit_ui = dpg.add_input_floatx(
                                default_value=(value[0], value[1], value[2]),
                                enabled=is_authored,
                                size=3,
                                callback=on_edit3,
                            )
                        case "quatf" | "quatd":
                            img = value.GetImaginary()
                            real = value.GetReal()

                            def on_edit_quat(sender, app_data, user_data):
                                img = app_data.GetImaginary()
                                real = app_data.GetReal()
                                set_value(
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
                                callback=on_edit_quat,
                            )

                def on_toggle_authored(sender, app_data, user_data):
                    for selected_prop in get_selected_props():
                        assert isinstance(selected_prop, Attribute)
                        selected_prop.Clear()
                        is_blocked = is_attr_blocked(prop)
                        if not app_data:
                            dpg.set_value(blocked_ui, is_blocked)
                        else:
                            if is_blocked:
                                selected_prop.Block()
                            else:
                                selected_prop.Set(prop.Get())
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
                    for seleced_prop in get_selected_props():
                        assert isinstance(seleced_prop, Attribute)
                        if app_data:
                            seleced_prop.Block()
                        else:
                            seleced_prop.Clear()

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

    def get_api_properties(self, prim: Prim) -> Iterable[tuple[str, Iterable[str]]]:
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

        return schema_attributes

    def get_none_apply_api_properties(
        self, prim: Prim
    ) -> Iterable[tuple[str, Iterable[str]]]:
        schema_attributes = []
        attr_names = []
        primvar_api = PrimvarsAPI(prim)
        primvars = primvar_api.GetPrimvars()
        for primvar in primvars:
            attr_name = primvar.GetAttr().GetName()
            attr_names.append(attr_name)
        if len(attr_names) > 0:
            schema_attributes.append(("PrimvarAPI", attr_names))

        return schema_attributes

    def get_type_properties(self, prim: Prim) -> tuple[str, Iterable[str]] | None:
        type_name = prim.GetTypeName()
        if not type_name:
            return None
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
            geometries: list[tuple[float, Matrix4d]] | None = None

        def prepare_filter(self) -> PrepareFilter:
            xform_cache = None
            geometries = None
            if self.geomtry_filter:
                xform_cache = XformCache()
                geometries = []

                for geometry in PrimRange(self.geomtry_filter):
                    if geometry.IsA(Cube):  # type: ignore
                        cube = Cube(geometry)
                        extent = abs(cube.GetSizeAttr().Get() / 2)
                        world2local = xform_cache.GetLocalToWorldTransform(
                            geometry
                        ).GetInverse()
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
                ).Transform(Vec3d(0.0, 0.0, 0.0))
                inside = False
                for extend, xform in prepare_filter.geometries:
                    local_position = array(xform.Transform(position))
                    if numpy.all(abs(local_position) <= extend):
                        inside = True
                        break
                if not inside:
                    return False

            return True

    @dataclass
    class StageSelect:
        path: Path
        exclusive: bool

    class Selection:
        def __init__(self, context: SelectionUI) -> None:
            self.selects = dict[Prim, SelectionUI.Select]()
            self.guide: Prim | None = None
            self.stage_selects = dict[Path, SelectionUI.StageSelect]()
            self.stage_guide: Path | None = None
            self.context = context

        def iter(self) -> Iterable[Prim]:
            excludes = set[Prim]()
            for select in self.selects.values():
                if select.exclusive:
                    for prim in select.iter():
                        excludes.add(prim)
            for select in self.selects.values():
                if not select.exclusive:
                    for prim in select.iter():
                        if prim not in excludes:
                            yield prim

        def stage_iter(
            self, auto_dirty_prim: bool = False, auto_dirty_stage: bool = False
        ) -> Iterable[tuple[Path, Stage, Iterable[Prim]]]:
            paths = list[Path]()

            def collect_path(path: Path, select: SelectionUI.StageSelect):
                if path.is_dir():
                    for path in path.iterdir():
                        if path in self.stage_selects:
                            continue

                        collect_path(path, select)
                elif path.is_file():
                    match path.suffix:
                        case ".usd" | ".usda" | ".usdc":
                            paths.append(path.resolve())

            for select in self.stage_selects.values():
                if not select.exclusive:
                    collect_path(select.path, select)

            operation_name_filter = dpg.get_value(self.context.operation_name_filter_ui)
            dependent_paths = set[Path]()
            if not operation_name_filter:
                for path in paths:
                    for dependent_path in find_usd_dependencies(path, False, False):
                        dependent_paths.add(dependent_path)
            for path in paths:
                if not operation_name_filter and path in dependent_paths:
                    continue
                stage = Stage.Open(str(path))
                if operation_name_filter:
                    if (
                        operation_custom_layer_data
                        := stage.GetRootLayer().customLayerData.get(
                            "assets_tool:operation"
                        )
                    ):
                        operation_name = operation_custom_layer_data["operation"]
                    else:
                        operation_name = None

                    if not operation_name:
                        continue
                    elif operation_name_filter != operation_name:
                        continue
                stage = self.context.load_stage(path)
                if auto_dirty_stage:
                    self.context.make_dirty(path, True)

                def prims():
                    dirty = False
                    for template_prim in self.iter():
                        if prim := stage.GetPrimAtPath(template_prim.GetPath()):
                            yield prim
                            if not auto_dirty_stage and auto_dirty_prim and not dirty:
                                dirty = True
                                self.context.make_dirty(path, True)

                yield (
                    path,
                    stage,
                    prims() if len(self.selects) > 0 else stage.Traverse(),
                )

    def __init__(
        self,
        selected_theme: int | str,
        on_select: list[Callable[[], None]],
        on_stage_select: list[Callable[[], None]],
        get_stage: Callable[[], Stage | None],
        make_dirty: Callable[[Path, bool], None],
        on_add_prim: Callable[[Prim], None],
        load_stage: Callable[[Path], Stage],
        parent: int | str = 0,
    ) -> None:
        self.selection = self.Selection(self)
        self.selected_theme = selected_theme
        self.on_select = on_select
        self.on_stage_select = on_stage_select
        self.get_stage = get_stage
        self.make_dirty = make_dirty
        self.add_prim = on_add_prim
        self.load_stage = load_stage
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

        with dpg.child_window(auto_resize_y=True, parent=parent):
            dpg.add_text("Selection")

            # def test():
            #     for path, stage, prims in self.selection.stage_iter():
            #         print(path)

            # dpg.add_button(label="test", callback=test)

            self.mode_ui = dpg.add_combo(
                ("single", "multi", "multi continuous", "guide"),
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
                    label="recursive", default_value=False, callback=on_recursive_ui
                )
                self.exclusive_ui = dpg.add_checkbox(
                    label="exclusive", default_value=False, callback=on_exclusive_ui
                )
            self.operation_name_filter_ui = dpg.add_input_text(label="operation name")
            with dpg.tree_node(label="filters", show=False) as self.filters_ui:
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

                with dpg.tree_node(label="geometry"):
                    self.geometry_filter_ui = SelectorUI(
                        get_selected_geometry_filter,
                        get_geometry_filter,
                        set_geometry_filter,
                        geometry_filter2text,
                    )

                    def add_cube():
                        if stage := self.get_stage():
                            path = unique_usd_path(
                                UsdPath("/assets_tool/selection_geometry_filter/cube"),
                                stage,
                            )
                            cube = Cube.Define(stage, path)
                            cube.GetPurposeAttr().Set("guide")
                            self.add_prim(cube.GetPrim())

                    dpg.add_button(label="add cube", callback=add_cube)

            self.selected_ui = dpg.add_tree_node(label="selected")

    def get_editing_select(self) -> SelectionUI.Select | None:
        if self.editing_select:
            return self.selection.selects.get(self.editing_select)
        return None

    def on_recursive(self, value: bool):
        dpg.configure_item(self.filters_ui, show=value)

    def clear(self):
        self.selection.selects.clear()
        self.select(None)
        self.update_selected_ui()

    def update_selected_ui(self):
        dpg.delete_item(self.selected_ui, children_only=True)

        def add_selected_ui(prim: Prim, select: SelectionUI.Select):
            with dpg.group(horizontal=True, parent=self.selected_ui):

                def on_button():
                    self.editing_select = prim
                    if select := self.selection.selects.get(self.editing_select):
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

        for prim, info in self.selection.selects.items():
            add_selected_ui(prim, info)

    def select(self, prim: Prim | None):
        prev_guide = self.selection.guide
        self.selection.guide = prim
        if prim:
            mode = dpg.get_value(self.mode_ui)
            if dpg.is_key_down(dpg.mvKey_LControl):
                mode = "multi"
            elif dpg.is_key_down(dpg.mvKey_LShift):
                mode = "multi continuous"
            match mode:
                case "single" | "multi" | "multi continuous":
                    if mode == "single":
                        self.selection.selects.clear()
                    if prev_guide is not None and mode == "multi continuous":
                        brothers = prim.GetParent().GetChildren()
                        prims = []
                        counter = 0
                        for brother in brothers:
                            if brother == prim:
                                counter += 1
                            if brother == prev_guide:
                                counter += 1
                            if counter > 0 and brother != prev_guide:
                                prims.append(brother)
                            if counter >= 2:
                                break
                    else:
                        prims = (prim,)

                    for prim in prims:
                        if prim in self.selection.selects:
                            self.selection.selects.pop(prim, None)
                        else:
                            api_filter = none_str2none(
                                dpg.get_value(self.api_filter_ui)
                            )
                            self.selection.selects[prim] = self.Select(
                                prim,
                                dpg.get_value(self.recursive_ui),
                                dpg.get_value(self.exclusive_ui),
                                none_str2none(dpg.get_value(self.name_filter_ui)),
                                none_str2none(dpg.get_value(self.type_filter_ui)),
                                [api_filter] if api_filter else [],
                                self.geometry_filter,
                            )
                            self.editing_select = prim
        self.update_selected_ui()
        for callback in self.on_select:
            callback()

    def select_stage(self, path: Path | None):
        if path is not None:
            path = path.resolve()
        self.selection.stage_guide = path
        if path:
            mode = dpg.get_value(self.mode_ui)
            if dpg.is_key_down(dpg.mvKey_LShift) or dpg.is_key_down(dpg.mvKey_LControl):
                mode = "multi"
            match mode:
                case "single" | "multi":
                    if mode == "single":
                        self.selection.stage_selects.clear()
                    if path in self.selection.stage_selects:
                        self.selection.stage_selects.pop(path, None)
                    else:
                        self.selection.stage_selects[path] = self.StageSelect(
                            path,
                            dpg.get_value(self.exclusive_ui),
                        )
        for callback in self.on_stage_select:
            callback()


class BlenderClient:
    class SyncedMesh:
        mesh: Mesh

    class Synced:
        def __init__(self) -> None:
            self.meshes = dict[Path, BlenderClient.SyncedMesh]()

    def __init__(
        self,
        parent: int | str,
        get_selection: Callable[[], SelectionUI.Selection],
        make_dirty: Callable[[Path, bool], None],
        on_add_prim: Callable[[Prim], None],
        mut_on_tick: list[Callable[[], None]],
        mut_on_end: list[Callable[[], None]],
    ) -> None:
        self.get_selection = get_selection
        self.make_dirty = make_dirty
        self.on_add_prim = on_add_prim

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
        with dpg.child_window(auto_resize_y=True, parent=parent):
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
                self.if_protect_visual = dpg.add_checkbox(
                    label="protect visual", default_value=True
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
        self.synced = self.Synced()
        xform_cache = XformCache()
        command_count = 0
        for file_path, stage, prims in selection.stage_iter():
            prims = set(prims)
            referenced_prims = set[Prim]()
            for prim in prims:
                ancestor = prim
                while True:
                    if ancestor.IsPseudoRoot() or ancestor in referenced_prims:
                        break
                    referenced_prims.add(ancestor)
                    ancestor = ancestor.GetParent()

            for prim in referenced_prims:
                in_selection = prim in prims
                path = Path(str(prim.GetPath()))
                if in_selection:
                    if prim.IsA(Mesh):  # type: ignore
                        mesh = Mesh(prim)
                        synced_mesh = BlenderClient.SyncedMesh()
                        self.synced.meshes[path] = synced_mesh
                        if (
                            face_vertex_counts_raw
                            := mesh.GetFaceVertexCountsAttr().Get()
                        ):
                            face_vertex_counts = array(face_vertex_counts_raw)
                            assert numpy.all(face_vertex_counts == 3)

                        if positions_raw := mesh.GetPointsAttr().Get():
                            positions = array(positions_raw)
                        else:
                            positions = array(((), ()))

                        if indices_raw := mesh.GetFaceVertexIndicesAttr().Get():
                            triangles = array(indices_raw).reshape(-1, 3)
                        else:
                            triangles = NDArray((0, 3), int32)

                        software_client.create_mesh(
                            self.client,
                            array(positions),
                            array(triangles),
                            path,
                            file_path,
                            dpg.get_value(self.if_sync_mesh_ui),
                        )
                        command_count += 4
                    elif prim.IsA(Cube):  # type: ignore
                        cube = Cube(prim)
                        software_client.create_cube(
                            self.client,
                            cube.GetSizeAttr().Get(),
                            path,
                            file_path,
                        )
                    elif prim.IsA(Cylinder):  # type: ignore
                        cylinder = Cylinder(prim)
                        software_client.command.create_cylinder(
                            self.client,
                            cylinder.GetRadiusAttr().Get(),
                            cylinder.GetHeightAttr().Get(),
                            str(cylinder.GetAxisAttr().Get()),
                            path,
                            file_path,
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
                        file_path,
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
        file_path: Path,
        guard: Any,
    ):
        def run():
            guard  # type: ignore
            selection = self.get_selection()
            if self.synced:
                stage = selection.context.load_stage(file_path)
                self.make_dirty(file_path, True)
                prim = stage.GetPrimAtPath(path.as_posix())
                mesh = Mesh(prim)
                if mesh.GetPurposeAttr().Get() != "guide" and dpg.get_value(
                    self.if_protect_visual
                ):  # type: ignore
                    if not prim.HasAPI(CollisionAPI):  # type: ignore
                        return
                    new_mesh = Mesh.Define(
                        stage,
                        unique_usd_path(
                            UsdPath((path / "protect_visual").as_posix()), stage
                        ),
                    )
                    new_prim = new_mesh.GetPrim()
                    CollisionAPI.Apply(new_prim)
                    copy_api(prim, new_prim, "PhysicsCollisionAPI")
                    prim.RemoveAPI(CollisionAPI)  # type: ignore
                    prim = new_prim
                    mesh = Mesh(prim)
                    mesh.GetPurposeAttr().Set("guide")
                mesh.GetPointsAttr().Set(Vec3fArray.FromNumpy(positions))
                mesh.GetFaceVertexIndicesAttr().Set(IntArray.FromNumpy(indices))
                mesh.GetFaceVertexCountsAttr().Set(
                    IntArray.FromNumpy(numpy.full(len(indices) // 3, 3))
                )
                mesh.GetNormalsAttr().Block()
                mesh.GetExtentAttr().Block()
                for primvar in PrimvarsAPI(prim).GetPrimvars():
                    primvar.GetAttr().Block()
                self.on_add_prim(prim)

        self.tasks.put(run)

    def sync_xform(
        self,
        translation: NDArray[float],
        rotation: NDArray[float],
        scale: NDArray[float],
        path: Path,
        file_path: Path,
    ):
        def run():
            selection = self.get_selection()
            if self.synced:
                stage = selection.context.load_stage(file_path)
                self.make_dirty(file_path, True)
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
        parent: int | str,
        get_selection: Callable[[], SelectionUI.Selection],
        add_prim: Callable[[Prim], None],
        remove_prim: Callable[[Prim], None],
    ) -> None:
        self.get_selection = get_selection
        self.add_prim = add_prim
        self.remove_prim = remove_prim
        self.copied_prim: Prim | None = None
        with dpg.child_window(auto_resize_y=True, parent=parent):
            dpg.add_text("Prim Util")
            self.name_ui = dpg.add_input_text(label="name", default_value="new_prim")
            self.mode_ui = dpg.add_combo(
                ("child", "brother"), label="mode", default_value="child"
            )
            with dpg.group(horizontal=True):
                dpg.add_button(label="create", callback=self.create_prim)
                dpg.add_button(label="delete", callback=self.delete_prim)
                dpg.add_button(label="copy", callback=self.copy_prim)
                dpg.add_button(label="paste", callback=self.paste_prim)

    def get_create_path(self, stage: Stage, prim: Prim | None) -> UsdPath:
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
        return unique_usd_path(path, stage)

    def create_prim(self):
        selection = self.get_selection()
        for path, stage, prims in selection.stage_iter(auto_dirty_prim=True):
            for prim in list(prims):
                usd_path = self.get_create_path(stage, prim)
                prim = stage.DefinePrim(usd_path)
                self.add_prim(prim)

    def delete_prim(self):
        selection = self.get_selection()
        for path, stage, prims in selection.stage_iter(auto_dirty_prim=True):
            for prim in list(prims):
                if prim:
                    self.remove_prim(prim)
                    usd_path = prim.GetPath()
                    if is_prim_authored_in_layer(prim, stage.GetRootLayer()):
                        stage.RemovePrim(usd_path)
                    else:
                        prim.SetActive(False)

    def copy_prim(self):
        if prim := self.get_selection().guide:
            self.copied_prim = prim

    def paste_prim(self):
        if self.copied_prim:
            selection = self.get_selection()
            for path, stage, prims in selection.stage_iter(auto_dirty_prim=True):
                for prim in list(prims):
                    usd_path = self.get_create_path(stage, prim)
                    prim = copy_prim(stage, self.copied_prim, usd_path, True)
                    self.add_prim(prim)


class SchemaUtil:
    def __init__(
        self,
        parent: int | str,
        get_selection: Callable[[], SelectionUI.Selection],
        on_schema_change: list[Callable[[Prim], None]],
    ) -> None:
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
        type_names.insert(0, "None")
        api_names = sorted(api_names)
        self.type_names = set(type_names)
        self.api_names = set(api_names)

        with dpg.child_window(auto_resize_y=True, parent=parent):
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
        if api_name := dpg.get_value(self.select_api_ui):
            selection = self.get_selection()
            api = SchemaRegistry.GetTypeFromSchemaTypeName(api_name)
            for path, stage, prims in selection.stage_iter(auto_dirty_prim=True):
                for prim in prims:
                    prim.RemoveAPI(api)
                    for callback in self.on_add_schema:
                        callback(prim)

    def set_type(self):
        selection = self.get_selection()
        type_name = dpg.get_value(self.select_type_ui)
        for path, stage, prims in selection.stage_iter(auto_dirty_prim=True):
            for prim in prims:
                if type_name == "None":
                    prim.ClearTypeName()
                else:
                    prim.SetTypeName(type_name)
                for callback in self.on_add_schema:
                    callback(prim)

    def add_api(self):
        selection = self.get_selection()
        for path, stage, prims in selection.stage_iter(auto_dirty_prim=True):
            for prim in prims:
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

    def select_type(self, name: str):
        if name in self.type_names:
            dpg.set_value(self.select_type_ui, name)

    def select_api(self, name: str):
        if name in self.api_names:
            dpg.set_value(self.select_api_ui, name)


class LayerUtil:
    def __init__(
        self,
        parent: int | str,
        get_selection: Callable[[], SelectionUI.Selection],
        on_clear: list[Callable[[], None]],
    ) -> None:
        self.container = parent
        self.get_selection = get_selection
        self.on_clear = on_clear
        with dpg.child_window(auto_resize_y=True, parent=self.container):
            dpg.add_text("Layer Util")
            dpg.add_button(label="clear", callback=self.clear)

    def clear(self):
        selection = self.get_selection()
        for path, stage, prims in selection.stage_iter(auto_dirty_stage=True):
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
            dpg.add_text("File Util")
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


class FontUtil:
    from importlib.resources import files

    font_path = files("assets_tool.assets").joinpath("fonts/FiraCode-Regular.ttf")
    size_exp_range = (4, 8)

    def __init__(self, default_scale: float) -> None:
        self.scale = default_scale
        self.max_width = 600
        self.fonts: list[int | str] = [0] * (
            self.size_exp_range[1] - self.size_exp_range[0]
        )
        with dpg.font_registry():
            with as_file(self.font_path) as path:
                for i in range(*self.size_exp_range):
                    self.fonts[i - self.size_exp_range[0]] = dpg.add_font(
                        str(path), round(exp2(i))
                    )
        for i in screeninfo.get_monitors():
            if i.width > self.max_width:
                self.max_width = i.width
        self.update()
        with dpg.handler_registry():
            dpg.add_mouse_wheel_handler(callback=self.on_scroll)

    def on_scroll(self, sender, app_data):
        if dpg.is_key_down(dpg.mvKey_LControl):
            self.scale *= exp(app_data / 8)
            self.update()

    def update(self):
        scale = self.scale * self.max_width / 64
        size_exp = log2(scale)
        if size_exp > self.size_exp_range[1] - 1:
            size_exp = self.size_exp_range[1] - 1
        elif size_exp < self.size_exp_range[0]:
            size_exp = self.size_exp_range[0]
        size_exp = round(size_exp)
        size_scale = exp2(size_exp)
        scale = scale / size_scale
        font = self.fonts[size_exp - self.size_exp_range[0]]
        dpg.bind_font(font)
        dpg.set_global_font_scale(scale)
        for item in dpg.get_all_items():
            dpg.bind_item_font(item, font)


class App:
    def __init__(self, font_scale: float = 1.0):
        dpg.create_context()

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
        self.font_util = FontUtil(font_scale)
        self.ui = UI()
        self.properties = PropertiesUI(
            lambda: self.file_explorer.stage,
            lambda: self.file_explorer.selected_file_path,
            lambda: self.selection_ui.selection,
            lambda path, dirty: self.file_explorer.make_dirty(path, dirty),
            lambda name: self.schema_util.select_type(name),
            lambda name: self.schema_util.select_api(name),
            parent=self.ui.properties,
        )
        self.hierarchy = Hierarchy(
            lambda prim: self.selection_ui.select(prim),
            lambda: self.file_explorer.stage,
            lambda: self.selection_ui.selection,
            [self.properties.on_remove_prim],
            self.selected_theme,
            self.guide_theme,
            self.selected_and_guide_theme,
            parent=self.ui.heirarchy,
        )
        self.file_explorer = FileExplorer(
            self.ui.file_explorer,
            lambda: self.selection_ui.selection,
            [
                lambda _: self.hierarchy.load_stage(),
                lambda _: self.selection_ui.clear(),
                lambda _: self.blender_client.unsync(),
                lambda path: self.selection_ui.select_stage(path),
            ],
            self.selected_theme,
            self.guide_theme,
            self.selected_and_guide_theme,
        )
        self.selection_ui = SelectionUI(
            self.selected_theme,
            [
                self.hierarchy.update_ui,
                lambda: self.properties.select_prim(self.selection_ui.selection.guide),
            ],
            [],
            lambda: self.file_explorer.stage,
            Lazy(lambda: self.file_explorer.make_dirty),
            self.hierarchy.on_add_prim,
            self.file_explorer.load_stage,
            parent=self.ui.operators,
        )
        self.blender_client = BlenderClient(
            self.ui.operators,
            lambda: self.selection_ui.selection,
            self.file_explorer.make_dirty,
            self.hierarchy.on_add_prim,
            self.ui.on_tick,
            self.ui.on_end,
        )
        self.prim_util = PrimUtil(
            self.ui.operators,
            lambda: self.selection_ui.selection,
            self.hierarchy.on_add_prim,
            self.hierarchy.on_remove_prim,
        )
        self.schema_util = SchemaUtil(
            self.ui.operators,
            lambda: self.selection_ui.selection,
            [self.properties.select_prim],
        )
        self.layer_util = LayerUtil(
            self.ui.operators,
            lambda: self.selection_ui.selection,
            [self.hierarchy.load_stage],
        )
        self.file_util = FileUtil(
            self.ui.operators,
            lambda: self.file_explorer.selected_file_path,
        )
        self.file_explorer.load_path(Path(".").resolve())

    def run(self):
        self.ui.run()


if __name__ == "__main__":
    app = App()
    app.run()
