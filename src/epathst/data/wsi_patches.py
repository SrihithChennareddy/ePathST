"""WSI readers (openslide; wsidicom for CPTAC's IDC-hosted DICOM) and patch extraction.

Patches are fixed-physical-size regions read at native resolution and resized only to the patch
size; whole slides are never resized. Both readers expose the same duck-typed interface.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from epathst.data.cptac_select import tissue_mask_from_thumbnail


def _clamp_region(
    x0: int,
    y0: int,
    size: int,
    level_width: int,
    level_height: int,
) -> tuple[int, int]:
    """Shift (x0, y0) so the box stays inside [0, level_width) x [0, level_height).

    Trades a small shift in patch center for real tissue instead of a crash or black padding at
    the slide edge.
    """
    x0 = max(0, min(x0, max(0, level_width - size)))
    y0 = max(0, min(y0, max(0, level_height - size)))
    return x0, y0


def open_slide(path: str):
    """Open a slide with the matching reader.

    A directory (or a path with sibling .dcm files) is a DICOM series: CPTAC/IDC ships one series as
    multiple SOP instances addressed by directory, never by filename. Anything else uses openslide.
    """
    p = Path(path)
    if p.is_dir() or (p.exists() and any(p.parent.glob("*.dcm"))):
        return WsiDicomReader(str(p if p.is_dir() else p.parent))
    return OpenSlideReader(str(p))


class WsiDicomReader:
    """wsidicom.WsiDicom wrapper with OpenSlideReader's interface.

    `path` is a directory of one series' .dcm instances. read_region coordinates are level-0 pixels,
    as in openslide, so the arithmetic in this module is unchanged.
    """

    def __init__(self, path: str):
        from wsidicom import WsiDicom

        self.path = path
        self.slide = WsiDicom.open(path)
        self.dimensions = (self.slide.size.width, self.slide.size.height)
        self.mpp = float(self.slide.mpp.width) if self.slide.mpp else None
        self._levels = sorted(self.slide.levels, key=lambda l: l.size.width, reverse=True)

    def _infer_mpp(self) -> float | None:
        return self.mpp

    def get_thumbnail(self, max_dim: int) -> np.ndarray:
        w, h = self.dimensions
        scale = max_dim / max(w, h)
        thumb = self.slide.read_thumbnail((max(1, int(w * scale)), max(1, int(h * scale))))
        return np.asarray(thumb.convert("RGB"))

    def level_for_mpp(self, target_mpp: float) -> int:
        """Index into self._levels (0 = highest resolution) nearest target_mpp.

        This is the `level` argument read_region expects, not the DICOM-native level number.
        """
        if self.mpp is None:
            return 0
        best_idx, best_diff = 0, float("inf")
        for idx, lvl in enumerate(self._levels):
            diff = abs(float(lvl.mpp.width) - target_mpp)
            if diff < best_diff:
                best_diff, best_idx = diff, idx
        return best_idx

    def read_region_20x(self, x_thumb: int, y_thumb: int, size_px: int) -> np.ndarray:
        lvl = self.level_for_mpp(0.5)
        region = self.slide.read_region((int(x_thumb), int(y_thumb)), lvl, (size_px, size_px))
        return np.asarray(region.convert("RGB"))

    def sample_tissue_tile_coords(
        self,
        tissue_mask: np.ndarray,
        n_tiles: int,
        tile_px: int,
    ) -> list[tuple[int, int]]:
        ys, xs = np.where(tissue_mask)
        if len(xs) == 0:
            return []
        rng = np.random.RandomState(0)
        idx = rng.choice(len(xs), size=min(n_tiles, len(xs)), replace=False)
        w0, h0 = self.dimensions
        th, tw = tissue_mask.shape
        scale_x, scale_y = w0 / tw, h0 / th
        coords = []
        for i in idx:
            x0 = int(xs[i] * scale_x - tile_px / 2)
            y0 = int(ys[i] * scale_y - tile_px / 2)
            coords.append((max(0, x0), max(0, y0)))
        return coords

    def read_region_at_mpp(
        self,
        center_x_px: float,
        center_y_px: float,
        physical_size_um: float,
        out_px: int,
        source_mpp: float | None = None,
    ) -> np.ndarray:
        from PIL import Image

        mpp = source_mpp or self.mpp or 0.5
        native_span = physical_size_um / mpp
        half = native_span / 2.0
        x0, y0 = int(round(center_x_px - half)), int(round(center_y_px - half))
        size = max(1, int(round(native_span)))
        lvl = self.level_for_mpp(mpp)
        lvl_w, lvl_h = self._levels[lvl].size.width, self._levels[lvl].size.height
        x0, y0 = _clamp_region(x0, y0, size, lvl_w, lvl_h)
        region = self.slide.read_region((x0, y0), lvl, (size, size)).convert("RGB")
        if size != out_px:
            region = region.resize((out_px, out_px), Image.BILINEAR)
        return np.asarray(region, dtype=np.uint8)

    def close(self):
        self.slide.close()


class OpenSlideReader:
    """openslide.OpenSlide wrapper with the same interface as WsiDicomReader."""

    def __init__(self, path: str):
        import openslide

        self.path = path
        self.slide = openslide.OpenSlide(path)
        self.dimensions = self.slide.dimensions
        self.mpp = self._infer_mpp()

    def _infer_mpp(self) -> float | None:
        props = self.slide.properties
        for key in ("openslide.mpp-x", "aperio.MPP"):
            if key in props:
                try:
                    return float(props[key])
                except ValueError:
                    continue
        # Fallback from objective power: 20x ~ 0.5 um/px, 40x ~ 0.25 um/px.
        power = props.get("aperio.AppMag") or props.get("openslide.objective-power")
        if power is not None:
            try:
                p = float(power)
                return 10.0 / p if p > 0 else None
            except ValueError:
                return None
        return None

    def get_thumbnail(self, max_dim: int) -> np.ndarray:
        w, h = self.dimensions
        scale = max_dim / max(w, h)
        thumb = self.slide.get_thumbnail((int(w * scale), int(h * scale)))
        return np.asarray(thumb.convert("RGB"))

    def level_for_mpp(self, target_mpp: float) -> int:
        if self.mpp is None:
            return 0
        best_level, best_diff = 0, float("inf")
        for lvl in range(self.slide.level_count):
            downsample = self.slide.level_downsamples[lvl]
            lvl_mpp = self.mpp * downsample
            diff = abs(lvl_mpp - target_mpp)
            if diff < best_diff:
                best_diff, best_level = diff, lvl
        return best_level

    def read_region_20x(self, x_thumb: int, y_thumb: int, size_px: int) -> np.ndarray:
        """Read a `size_px` square at the pyramid level nearest 0.5 um/px.

        Despite the names, x_thumb/y_thumb are level-0 pixel coordinates, not thumbnail ones.
        """
        lvl = self.level_for_mpp(0.5)
        region = self.slide.read_region((int(x_thumb), int(y_thumb)), lvl, (size_px, size_px))
        return np.asarray(region.convert("RGB"))

    def sample_tissue_tile_coords(
        self,
        tissue_mask: np.ndarray,
        n_tiles: int,
        tile_px: int,
    ) -> list[tuple[int, int]]:
        """Sample up to n_tiles level-0 tile origins uniformly over thumbnail tissue pixels.

        Uses a fixed seed (0) so QC tiles are reproducible for a given mask.
        """
        ys, xs = np.where(tissue_mask)
        if len(xs) == 0:
            return []
        rng = np.random.RandomState(0)
        idx = rng.choice(len(xs), size=min(n_tiles, len(xs)), replace=False)
        w0, h0 = self.dimensions
        th, tw = tissue_mask.shape
        scale_x, scale_y = w0 / tw, h0 / th
        coords = []
        for i in idx:
            x0 = int(xs[i] * scale_x - tile_px / 2)
            y0 = int(ys[i] * scale_y - tile_px / 2)
            coords.append((max(0, x0), max(0, y0)))
        return coords

    def read_region_at_mpp(
        self,
        center_x_px: float,
        center_y_px: float,
        physical_size_um: float,
        out_px: int,
        source_mpp: float | None = None,
    ) -> np.ndarray:
        """Read a `physical_size_um`-wide square centered at (center_x_px, center_y_px).

        The region is read at level 0 and resized to out_px x out_px. Used for both CPTAC and HEST
        patches.
        """
        from PIL import Image

        mpp = source_mpp or self.mpp or 0.25
        native_span = physical_size_um / mpp
        half = native_span / 2.0
        x0, y0 = int(round(center_x_px - half)), int(round(center_y_px - half))
        size = max(1, int(round(native_span)))
        x0, y0 = _clamp_region(x0, y0, size, self.dimensions[0], self.dimensions[1])
        region = self.slide.read_region((x0, y0), 0, (size, size)).convert("RGB")
        if size != out_px:
            region = region.resize((out_px, out_px), Image.BILINEAR)
        return np.asarray(region, dtype=np.uint8)


def sample_patch_grid(
    reader: OpenSlideReader,
    tissue_mask_thumb: np.ndarray,
    cfg: dict,
    rng_seed: int,
) -> list[tuple[int, int]]:
    """Regular grid of patch centers (level-0 px) over tissue, subsampled to the target count.

    The grid stride is one patch footprint (patch_px * mpp / target_mpp), so candidates do not
    overlap; subsampling spreads patches instead of keeping adjacent redundant ones.
    """
    patching = cfg["patching"]
    patch_px, target_mpp = patching["patch_px"], patching["target_mpp"]
    w0, h0 = reader.dimensions
    stride_px = int(round(patch_px * (reader.mpp or target_mpp) / target_mpp))
    th, tw = tissue_mask_thumb.shape
    scale_x, scale_y = tw / w0, th / h0

    # Tissue test uses the 3x3 thumbnail neighborhood around each grid point.
    xs = np.arange(0, w0 - stride_px, stride_px)
    ys = np.arange(0, h0 - stride_px, stride_px)
    candidates = []
    for y in ys:
        ty = int(y * scale_y)
        row = tissue_mask_thumb[max(0, ty - 1) : ty + 2, :]
        for x in xs:
            tx = int(x * scale_x)
            patch_tissue = row[:, max(0, tx - 1) : tx + 2]
            if patch_tissue.size and patch_tissue.mean() >= patching["min_tissue_fraction"]:
                candidates.append((int(x + stride_px // 2), int(y + stride_px // 2)))

    # Only the upper bound of target_patches_per_wsi is enforced.
    lo, hi = patching["target_patches_per_wsi"]
    rng = np.random.RandomState(rng_seed)
    if len(candidates) > hi:
        idx = rng.choice(len(candidates), size=hi, replace=False)
        candidates = [candidates[i] for i in idx]
    return candidates


def patch_tissue_fraction(patch_rgb: np.ndarray) -> float:
    mask = tissue_mask_from_thumbnail(patch_rgb)
    return float(mask.mean())


def write_patch_shard(
    out_path: str | Path,
    patch_ids: list[str],
    patches: np.ndarray,
    meta: dict,
) -> None:
    """Write one HDF5 shard per WSI instead of millions of PNG files.

    Chunked one image at a time so single-patch reads stay cheap.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.create_dataset(
            "patches",
            data=patches,
            dtype="uint8",
            compression="gzip",
            compression_opts=4,
            chunks=(1,) + patches.shape[1:],
        )
        f.create_dataset("patch_id", data=np.array(patch_ids, dtype="S"))
        for k, v in meta.items():
            f.create_dataset(k, data=np.asarray(v))
