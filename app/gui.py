from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from PySide6 import QtCore, QtGui, QtWidgets

from app.worker import StitchWorker
from core.io import load_project, load_tiles, save_project, save_transforms
from core.types import (
    ColorMatchMode,
    InpaintMethod,
    ProjectState,
    StitchMode,
    StitchSettings,
    TileTransform,
)


class DropListWidget(QtWidgets.QListWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event: QtGui.QDragMoveEvent) -> None:
        event.acceptProposedAction()

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        for url in event.mimeData().urls():
            self.addItem(url.toLocalFile())


class MapRestorerWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Map Restorer Stitcher")
        self.worker_thread: QtCore.QThread | None = None
        self.worker: StitchWorker | None = None
        self.transforms: Dict[str, TileTransform] = {}
        self._build_ui()

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        file_bar = QtWidgets.QHBoxLayout()
        self.add_files_btn = QtWidgets.QPushButton("Add Files")
        self.add_folder_btn = QtWidgets.QPushButton("Add Folder")
        self.remove_btn = QtWidgets.QPushButton("Remove Selected")
        file_bar.addWidget(self.add_files_btn)
        file_bar.addWidget(self.add_folder_btn)
        file_bar.addWidget(self.remove_btn)
        layout.addLayout(file_bar)

        self.file_list = DropListWidget()
        layout.addWidget(self.file_list)

        output_bar = QtWidgets.QHBoxLayout()
        self.output_path = QtWidgets.QLineEdit("out.tif")
        self.output_browse = QtWidgets.QPushButton("Browse")
        output_bar.addWidget(QtWidgets.QLabel("Output"))
        output_bar.addWidget(self.output_path)
        output_bar.addWidget(self.output_browse)
        layout.addLayout(output_bar)

        options_layout = QtWidgets.QGridLayout()
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["fast", "restoration"])
        self.debug_check = QtWidgets.QCheckBox("Debug Stitch")
        self.protect_check = QtWidgets.QCheckBox("Protect lines")
        self.protect_check.setChecked(True)
        self.optimize_check = QtWidgets.QCheckBox("Optimize TIFF (tiled+overviews)")
        self.optimize_check.setChecked(True)
        options_layout.addWidget(QtWidgets.QLabel("Mode"), 0, 0)
        options_layout.addWidget(self.mode_combo, 0, 1)
        options_layout.addWidget(self.debug_check, 0, 2)
        options_layout.addWidget(self.protect_check, 0, 3)
        options_layout.addWidget(self.optimize_check, 0, 4)
        layout.addLayout(options_layout)

        settings_group = QtWidgets.QGroupBox("Parameters")
        form = QtWidgets.QFormLayout(settings_group)
        self.bg_threshold = QtWidgets.QSpinBox()
        self.bg_threshold.setRange(0, 255)
        self.bg_threshold.setValue(245)
        self.bg_mode = QtWidgets.QComboBox()
        self.bg_mode.addItems(["lab_paper", "rgb_threshold"])
        self.crop_padding = QtWidgets.QSpinBox()
        self.crop_padding.setRange(0, 50)
        self.crop_padding.setValue(4)
        self.edge_trim = QtWidgets.QSpinBox()
        self.edge_trim.setRange(0, 200)
        self.edge_trim.setValue(40)
        self.max_angle = QtWidgets.QDoubleSpinBox()
        self.max_angle.setRange(0, 15)
        self.max_angle.setValue(7.0)
        self.overlap_max = QtWidgets.QSpinBox()
        self.overlap_max.setRange(0, 15)
        self.overlap_max.setValue(15)
        self.seam_band = QtWidgets.QSpinBox()
        self.seam_band.setRange(10, 80)
        self.seam_band.setValue(30)
        self.refine_iterations = QtWidgets.QSpinBox()
        self.refine_iterations.setRange(1, 5)
        self.refine_iterations.setValue(3)
        self.max_scale = QtWidgets.QDoubleSpinBox()
        self.max_scale.setRange(0.0, 1.0)
        self.max_scale.setSingleStep(0.1)
        self.max_scale.setValue(0.5)
        self.max_edge_warp = QtWidgets.QSpinBox()
        self.max_edge_warp.setRange(0, 5)
        self.max_edge_warp.setValue(2)
        self.feather = QtWidgets.QSpinBox()
        self.feather.setRange(0, 20)
        self.feather.setValue(10)
        self.color_match = QtWidgets.QComboBox()
        self.color_match.addItems([mode.value for mode in ColorMatchMode])
        self.seam_fill_check = QtWidgets.QCheckBox("Seam Fill")
        self.seam_fill_max = QtWidgets.QSpinBox()
        self.seam_fill_max.setRange(0, 30)
        self.seam_fill_max.setValue(15)
        self.inpaint_radius = QtWidgets.QSpinBox()
        self.inpaint_radius.setRange(1, 10)
        self.inpaint_radius.setValue(3)
        self.inpaint_method = QtWidgets.QComboBox()
        self.inpaint_method.addItems([method.value for method in InpaintMethod])
        self.compression = QtWidgets.QComboBox()
        self.compression.addItems(["deflate", "lzw"])
        form.addRow("bg_threshold", self.bg_threshold)
        form.addRow("bg_mode", self.bg_mode)
        form.addRow("crop_padding_px", self.crop_padding)
        form.addRow("edge_trim_px", self.edge_trim)
        form.addRow("max_angle", self.max_angle)
        form.addRow("overlap_max", self.overlap_max)
        form.addRow("seam_band_px", self.seam_band)
        form.addRow("refine_iterations", self.refine_iterations)
        form.addRow("max_scale_percent", self.max_scale)
        form.addRow("max_edge_warp_px", self.max_edge_warp)
        form.addRow("feather_px", self.feather)
        form.addRow("compression", self.compression)
        form.addRow("Color match", self.color_match)
        form.addRow(self.seam_fill_check)
        form.addRow("seam_fill_max_px", self.seam_fill_max)
        form.addRow("inpaint_radius", self.inpaint_radius)
        form.addRow("inpaint_method", self.inpaint_method)
        layout.addWidget(settings_group)

        manual_group = QtWidgets.QGroupBox("Manual Adjust")
        manual_layout = QtWidgets.QGridLayout(manual_group)
        self.manual_dx = QtWidgets.QSpinBox()
        self.manual_dx.setRange(-20, 20)
        self.manual_dy = QtWidgets.QSpinBox()
        self.manual_dy.setRange(-20, 20)
        self.manual_scale = QtWidgets.QDoubleSpinBox()
        self.manual_scale.setRange(0.995, 1.005)
        self.manual_scale.setDecimals(4)
        self.manual_scale.setSingleStep(0.0005)
        self.apply_manual = QtWidgets.QPushButton("Apply to Selected Tile")
        manual_layout.addWidget(QtWidgets.QLabel("dx"), 0, 0)
        manual_layout.addWidget(self.manual_dx, 0, 1)
        manual_layout.addWidget(QtWidgets.QLabel("dy"), 0, 2)
        manual_layout.addWidget(self.manual_dy, 0, 3)
        manual_layout.addWidget(QtWidgets.QLabel("scale"), 0, 4)
        manual_layout.addWidget(self.manual_scale, 0, 5)
        manual_layout.addWidget(self.apply_manual, 1, 0, 1, 6)
        layout.addWidget(manual_group)

        action_bar = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton("Start")
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.progress = QtWidgets.QProgressBar()
        action_bar.addWidget(self.start_btn)
        action_bar.addWidget(self.stop_btn)
        action_bar.addWidget(self.progress)
        layout.addLayout(action_bar)

        preview_bar = QtWidgets.QHBoxLayout()
        self.preview_label = QtWidgets.QLabel("Preview")
        self.preview_label.setFixedHeight(300)
        self.preview_label.setAlignment(QtCore.Qt.AlignCenter)
        self.heatmap_label = QtWidgets.QLabel("Seam Heatmap")
        self.heatmap_label.setFixedHeight(300)
        self.heatmap_label.setAlignment(QtCore.Qt.AlignCenter)
        preview_bar.addWidget(self.preview_label)
        preview_bar.addWidget(self.heatmap_label)
        layout.addLayout(preview_bar)

        self.log_box = QtWidgets.QTextEdit()
        self.log_box.setReadOnly(True)
        layout.addWidget(self.log_box)

        project_bar = QtWidgets.QHBoxLayout()
        self.save_project_btn = QtWidgets.QPushButton("Save Project")
        self.load_project_btn = QtWidgets.QPushButton("Load Project")
        project_bar.addWidget(self.save_project_btn)
        project_bar.addWidget(self.load_project_btn)
        layout.addLayout(project_bar)

        self.add_files_btn.clicked.connect(self.add_files)
        self.add_folder_btn.clicked.connect(self.add_folder)
        self.remove_btn.clicked.connect(self.remove_selected)
        self.output_browse.clicked.connect(self.choose_output)
        self.start_btn.clicked.connect(self.start)
        self.stop_btn.clicked.connect(self.stop)
        self.apply_manual.clicked.connect(self.apply_manual_transform)
        self.save_project_btn.clicked.connect(self.save_project_dialog)
        self.load_project_btn.clicked.connect(self.load_project_dialog)

    def add_files(self) -> None:
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Select tiles")
        for file in files:
            self.file_list.addItem(file)

    def add_folder(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select folder")
        if folder:
            self.file_list.addItem(folder)

    def remove_selected(self) -> None:
        for item in self.file_list.selectedItems():
            row = self.file_list.row(item)
            self.file_list.takeItem(row)

    def choose_output(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Output TIFF", "out.tif", "TIFF (*.tif *.tiff)")
        if path:
            self.output_path.setText(path)

    def _settings_from_ui(self) -> StitchSettings:
        return StitchSettings(
            bg_threshold=self.bg_threshold.value(),
            bg_mode=self.bg_mode.currentText(),
            crop_padding_px=self.crop_padding.value(),
            edge_trim_px=self.edge_trim.value(),
            max_angle=self.max_angle.value(),
            overlap_max=self.overlap_max.value(),
            seam_band_px=self.seam_band.value(),
            refine_iterations=self.refine_iterations.value(),
            max_scale_percent=self.max_scale.value(),
            max_edge_warp_px=self.max_edge_warp.value(),
            feather_px=self.feather.value(),
            compression=self.compression.currentText(),
            color_match=ColorMatchMode(self.color_match.currentText()),
            protect_lines=self.protect_check.isChecked(),
            seam_fill_enabled=self.seam_fill_check.isChecked(),
            seam_fill_max_px=self.seam_fill_max.value(),
            inpaint_radius=self.inpaint_radius.value(),
            inpaint_method=InpaintMethod(self.inpaint_method.currentText()),
            optimize_tiff=self.optimize_check.isChecked(),
            debug_stitch=self.debug_check.isChecked(),
        )

    def _collect_paths(self) -> List[Path]:
        return [Path(self.file_list.item(idx).text()) for idx in range(self.file_list.count())]

    def start(self) -> None:
        if self.worker_thread and self.worker_thread.isRunning():
            return
        paths = self._collect_paths()
        if not paths:
            self.log_box.append("No tiles selected")
            return
        settings = self._settings_from_ui()
        output_path = Path(self.output_path.text())
        mode = StitchMode(self.mode_combo.currentText())
        self.worker_thread = QtCore.QThread()
        self.worker = StitchWorker(paths, settings, output_path, mode, self.transforms)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.log.connect(self.log_box.append)
        self.worker.preview_ready.connect(self.update_preview)
        self.worker.failed.connect(self.log_box.append)
        self.worker.finished.connect(self.on_finished)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker_thread.start()

    def stop(self) -> None:
        if self.worker:
            self.worker.cancel()
            self.log_box.append("Cancel requested")

    def on_finished(self, output: Path) -> None:
        self.log_box.append(f"Saved: {output}")
        self.progress.setValue(100)

    def update_preview(self, image: object, heatmap: object) -> None:
        self.preview_label.setPixmap(self._to_pixmap(image))
        self.heatmap_label.setPixmap(self._to_pixmap(heatmap))

    def _to_pixmap(self, image: object) -> QtGui.QPixmap:
        if isinstance(image, QtGui.QImage):
            return QtGui.QPixmap.fromImage(image)
        array = image
        h, w, c = array.shape
        qimage = QtGui.QImage(array.data, w, h, c * w, QtGui.QImage.Format_RGB888)
        return QtGui.QPixmap.fromImage(qimage)

    def apply_manual_transform(self) -> None:
        selected = self.file_list.selectedItems()
        if not selected:
            self.log_box.append("Select a tile in the list to adjust")
            return
        path = Path(selected[0].text())
        tiles = load_tiles([path])
        if not tiles:
            self.log_box.append("Tile name does not match x,y pattern")
            return
        tile = tiles[0]
        key = f"{tile.x},{tile.y}"
        self.transforms[key] = TileTransform(
            dx=self.manual_dx.value(),
            dy=self.manual_dy.value(),
            scale_x=self.manual_scale.value(),
            scale_y=self.manual_scale.value(),
        )
        self.log_box.append(f"Applied manual transform to {key}")

    def save_project_dialog(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Project", "project.json", "JSON (*.json)")
        if not path:
            return
        tiles = load_tiles(self._collect_paths())
        settings = self._settings_from_ui()
        state = ProjectState(tiles=tiles, settings=settings, transforms=self.transforms, output_path=Path(self.output_path.text()))
        save_project(state, Path(path))
        save_transforms(self.transforms, Path(path).with_name("transforms.json"))
        self.log_box.append("Project saved")

    def load_project_dialog(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load Project", "project.json", "JSON (*.json)")
        if not path:
            return
        state = load_project(Path(path), Path("."))
        self.file_list.clear()
        for tile in state.tiles:
            self.file_list.addItem(str(tile.path))
        self.output_path.setText(str(state.output_path) if state.output_path else "out.tif")
        self.transforms = state.transforms
        self.log_box.append("Project loaded")
