#!/usr/bin/env python3
import sys, argparse
import numpy as np
import nibabel as nib
from PySide6 import QtWidgets, QtCore
import pyqtgraph as pg

def load_nifti_ras(path):
    img = nib.load(path)
    ras = nib.as_closest_canonical(img)
    data = ras.get_fdata(dtype=np.float32)
    if np.isfinite(data).any():
        p995 = np.percentile(data[np.isfinite(data)], 99.5)
        if p995 > 0:
            data = np.clip(data / p995, 0, 1)
    return data

class SliceView(pg.GraphicsLayoutWidget):
    sliceChanged = QtCore.Signal(int)
    def __init__(self, vol, axis, title):
        super().__init__()
        self.vol = vol
        self.axis = axis
        self.idx  = vol.shape[self.axis] // 2
        self.setWindowTitle(title)
        self.view = self.addViewBox(lockAspect=True, enableMouse=False)
        self.img_item = pg.ImageItem(axisOrder='row-major')
        self.view.addItem(self.img_item)
        bar = QtWidgets.QWidget(); lay = QtWidgets.QHBoxLayout(bar); lay.setContentsMargins(0,0,0,0)
        self.label = QtWidgets.QLabel(f"{title}  idx={self.idx}")
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, self.vol.shape[self.axis]-1)
        self.slider.setValue(self.idx)
        self.slider.valueChanged.connect(self._on_slider)
        lay.addWidget(self.label); lay.addWidget(self.slider)
        proxy = QtWidgets.QGraphicsProxyWidget(); proxy.setWidget(bar); self.addItem(proxy)
        self.update_slice()

    def _extract_slice(self):
        if self.axis == 0:  # axial Z fixed -> (Y,X)
            sl = self.vol[self.idx, :, :]; sl = np.flipud(sl)
        elif self.axis == 1:  # coronal Y fixed -> (Z,X)
            sl = self.vol[:, self.idx, :]; sl = np.flipud(sl)
        else:  # sagittal X fixed -> (Z,Y)
            sl = self.vol[:, :, self.idx]; sl = np.flipud(sl)
        return sl

    def update_slice(self):
        self.img_item.setImage(self._extract_slice(), autoLevels=True)
        self.label.setText(f"{self.windowTitle()}  idx={self.idx}")

    def wheelEvent(self, ev):
        self.set_index(int(np.clip(self.idx + (1 if ev.angleDelta().y()>0 else -1), 0, self.vol.shape[self.axis]-1)))
        self.sliceChanged.emit(self.idx); ev.accept()

    def _on_slider(self, v):
        self.set_index(v); self.sliceChanged.emit(self.idx)

    def set_index(self, v):
        v = int(v)
        if v != self.idx:
            self.idx = v
            self.slider.blockSignals(True); self.slider.setValue(v); self.slider.blockSignals(False)
            self.update_slice()

class Main(QtWidgets.QMainWindow):
    def __init__(self, vol):
        super().__init__()
        self.setWindowTitle("Simple NIfTI Viewer — 3 Views")
        self.vol = vol
        self.axial, self.coronal, self.sagittal = (
            SliceView(vol, 0, "Axial (Z)"),
            SliceView(vol, 1, "Coronal (Y)"),
            SliceView(vol, 2, "Sagittal (X)"),
        )
        central = QtWidgets.QWidget(); lay = QtWidgets.QHBoxLayout(central)
        lay.setContentsMargins(6,6,6,6); lay.setSpacing(6)
        lay.addWidget(self.axial); lay.addWidget(self.coronal); lay.addWidget(self.sagittal)
        self.setCentralWidget(central)
        tb = self.addToolBar("View")
        act_center = QtWidgets.QAction("Center slices", self); act_center.triggered.connect(self.center)
        tb.addAction(act_center)
        self.resize(1650, 600)

    def center(self):
        self.axial.set_index(self.vol.shape[0]//2)
        self.coronal.set_index(self.vol.shape[1]//2)
        self.sagittal.set_index(self.vol.shape[2]//2)

def parse_args():
    ap = argparse.ArgumentParser(description="Simple 3-view NIfTI viewer (Qt + PyQtGraph)")
    ap.add_argument("--vol", required=True, help="Path to .nii or .nii.gz (3D)")
    return ap.parse_args()

def main():
    args = parse_args()
    vol = load_nifti_ras(args.vol)
    if vol.ndim != 3:
        print(f"ERROR: expected 3D volume, got shape {vol.shape}"); sys.exit(1)
    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(imageAxisOrder='row-major')
    win = Main(vol); win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
