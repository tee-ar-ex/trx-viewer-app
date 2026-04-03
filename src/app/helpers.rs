use std::path::Path;

use trx_rs::Format;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DroppedPathKind {
    OpenTrx,
    ImportTractogram(Format),
    OpenNifti,
    OpenGifti,
    Unsupported,
}

pub(super) fn classify_dropped_path(path: &Path) -> DroppedPathKind {
    match trx_rs::detect_format(path) {
        Ok(Format::Trx) => DroppedPathKind::OpenTrx,
        Ok(format @ (Format::Tck | Format::Vtk | Format::TinyTrack)) => {
            DroppedPathKind::ImportTractogram(format)
        }
        Err(_) => {
            let ext = path
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("")
                .to_lowercase();
            let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
            match ext.as_str() {
                "gz" if stem.ends_with(".nii") => DroppedPathKind::OpenNifti,
                "nii" => DroppedPathKind::OpenNifti,
                "gii" | "gifti" => DroppedPathKind::OpenGifti,
                _ => DroppedPathKind::Unsupported,
            }
        }
    }
}

pub(super) fn tri_axis_value(p: glam::Vec3, axis_index: usize) -> f32 {
    match axis_index {
        0 => p.z,
        1 => p.y,
        _ => p.x,
    }
}

pub(super) fn intersect_edge_with_slice(
    p0: glam::Vec3,
    p1: glam::Vec3,
    axis_index: usize,
    slice_pos: f32,
    eps: f32,
) -> Option<glam::Vec3> {
    let c0 = tri_axis_value(p0, axis_index);
    let c1 = tri_axis_value(p1, axis_index);
    let d0 = c0 - slice_pos;
    let d1 = c1 - slice_pos;

    // Coplanar edge: skip to avoid degenerate full-triangle artifacts.
    if d0.abs() <= eps && d1.abs() <= eps {
        return None;
    }
    if d0.abs() <= eps {
        return Some(p0);
    }
    if d1.abs() <= eps {
        return Some(p1);
    }
    if d0 * d1 > 0.0 {
        return None;
    }
    let t = d0 / (d0 - d1);
    Some(p0 + (p1 - p0) * t)
}
