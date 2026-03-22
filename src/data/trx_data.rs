use std::path::Path;

use bytemuck::Pod;
use glam::Vec3;
use trx_rs::{AnyTrxFile, DType, TrxFile, TrxScalar};

/// Available coloring modes for streamline visualization.
#[derive(Clone, Debug, PartialEq)]
pub enum ColorMode {
    /// RGB from absolute local tangent direction.
    DirectionRgb,
    /// Scalar per-vertex field mapped to colormap.
    Dpv(String),
    /// Scalar per-streamline field expanded to all vertices.
    Dps(String),
    /// Distinct color per group.
    Group,
    /// Single uniform color for all vertices.
    Uniform([f32; 4]),
}

/// GPU-ready streamline data extracted from a TRX file.
pub struct TrxGpuData {
    /// Raw positions in RAS+ space.
    pub positions: Vec<[f32; 3]>,
    /// Streamline offsets (length = nb_streamlines + 1).
    pub offsets: Vec<u64>,
    /// Bounding box min/max in RAS+ space.
    pub bbox_min: Vec3,
    pub bbox_max: Vec3,
    /// Metadata counts.
    pub nb_streamlines: usize,
    pub nb_vertices: usize,
    /// Available DPV field names.
    pub dpv_names: Vec<String>,
    /// Available DPS field names.
    pub dps_names: Vec<String>,
    /// Available group names and their streamline indices.
    pub groups: Vec<(String, Vec<u32>)>,
    /// Cached DPV data: each entry is a flat Vec<f32> (one value per vertex).
    pub dpv_data: Vec<(String, Vec<f32>)>,
    /// Cached DPS data: each entry is a flat Vec<f32> (one value per streamline).
    pub dps_data: Vec<(String, Vec<f32>)>,
    /// Current vertex colors.
    pub colors: Vec<[f32; 4]>,
    /// Full line-segment index buffer (all streamlines).
    pub all_indices: Vec<u32>,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct StreamlineVertex {
    pub position: [f32; 3],
    pub color: [f32; 4],
}

impl TrxGpuData {
    /// Load a TRX file and produce GPU-ready data with direction-RGB coloring.
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let any = AnyTrxFile::load(path)?;

        // Convert all positions to f32
        let positions: Vec<[f32; 3]> = match any.positions_ref() {
            trx_rs::PositionsRef::F32(p) => p.to_vec(),
            trx_rs::PositionsRef::F64(p) => p
                .iter()
                .map(|v| [v[0] as f32, v[1] as f32, v[2] as f32])
                .collect(),
            trx_rs::PositionsRef::F16(p) => p
                .iter()
                .map(|v| {
                    [
                        half::f16::to_f32(v[0]),
                        half::f16::to_f32(v[1]),
                        half::f16::to_f32(v[2]),
                    ]
                })
                .collect(),
        };

        let nb_vertices = positions.len();
        let nb_streamlines = any.nb_streamlines();

        // Get offsets
        let offsets: Vec<u64> = any.with_typed(
            |trx| trx.offsets().to_vec(),
            |trx| trx.offsets().to_vec(),
            |trx| trx.offsets().to_vec(),
        );

        // Compute bounding box
        let mut bbox_min = Vec3::splat(f32::INFINITY);
        let mut bbox_max = Vec3::splat(f32::NEG_INFINITY);
        for p in &positions {
            let v = Vec3::from(*p);
            bbox_min = bbox_min.min(v);
            bbox_max = bbox_max.max(v);
        }

        // Extract DPV/DPS/groups using with_typed dispatch
        let (dpv_data, dps_data, groups) = any.with_typed(
            |trx| extract_metadata(trx, nb_vertices, nb_streamlines),
            |trx| extract_metadata(trx, nb_vertices, nb_streamlines),
            |trx| extract_metadata(trx, nb_vertices, nb_streamlines),
        );

        // Build full index buffer
        let all_indices = build_line_indices(&offsets);

        // Default: direction-RGB colors
        let colors = compute_direction_colors(&positions, &offsets);

        Ok(Self {
            positions,
            offsets,
            bbox_min,
            bbox_max,
            nb_streamlines,
            nb_vertices,
            dpv_names: dpv_data.iter().map(|(n, _)| n.clone()).collect(),
            dps_names: dps_data.iter().map(|(n, _)| n.clone()).collect(),
            groups,
            dpv_data,
            dps_data,
            colors,
            all_indices,
        })
    }

    /// Recompute vertex colors based on the given color mode.
    pub fn recolor(&mut self, mode: &ColorMode) {
        self.colors = match mode {
            ColorMode::DirectionRgb => compute_direction_colors(&self.positions, &self.offsets),
            ColorMode::Dpv(name) => {
                if let Some((_, data)) = self.dpv_data.iter().find(|(n, _)| n == name) {
                    scalar_to_colors(data)
                } else {
                    vec![[0.5, 0.5, 0.5, 1.0]; self.nb_vertices]
                }
            }
            ColorMode::Dps(name) => {
                if let Some((_, data)) = self.dps_data.iter().find(|(n, _)| n == name) {
                    expand_dps_to_vertices(data, &self.offsets, self.nb_vertices)
                } else {
                    vec![[0.5, 0.5, 0.5, 1.0]; self.nb_vertices]
                }
            }
            ColorMode::Group => self.compute_group_colors(),
            ColorMode::Uniform(c) => vec![*c; self.nb_vertices],
        };
    }

    /// Build interleaved vertex data from current positions + colors.
    pub fn vertices(&self) -> Vec<StreamlineVertex> {
        self.positions
            .iter()
            .zip(self.colors.iter())
            .map(|(pos, col)| StreamlineVertex {
                position: *pos,
                color: *col,
            })
            .collect()
    }

    /// Build index buffer for only the visible groups' streamlines.
    /// If no groups exist, returns all indices.
    pub fn indices_for_visible_groups(&self, visible: &[bool]) -> Vec<u32> {
        if self.groups.is_empty() {
            return self.all_indices.clone();
        }

        let mut indices = Vec::new();
        for (i, (_, members)) in self.groups.iter().enumerate() {
            if i < visible.len() && !visible[i] {
                continue;
            }
            for &streamline_idx in members {
                let si = streamline_idx as usize;
                if si + 1 < self.offsets.len() {
                    let start = self.offsets[si] as u32;
                    let end = self.offsets[si + 1] as u32;
                    for j in start..end.saturating_sub(1) {
                        indices.push(j);
                        indices.push(j + 1);
                    }
                }
            }
        }
        indices
    }

    fn compute_group_colors(&self) -> Vec<[f32; 4]> {
        let mut colors = vec![[0.5f32, 0.5, 0.5, 1.0]; self.nb_vertices];

        // Distinct colors for each group
        let palette: &[[f32; 4]] = &[
            [1.0, 0.2, 0.2, 1.0], // red
            [0.2, 0.7, 1.0, 1.0], // blue
            [0.2, 1.0, 0.3, 1.0], // green
            [1.0, 0.8, 0.1, 1.0], // yellow
            [1.0, 0.4, 0.8, 1.0], // pink
            [0.6, 0.3, 1.0, 1.0], // purple
            [1.0, 0.6, 0.2, 1.0], // orange
            [0.3, 1.0, 0.8, 1.0], // cyan
        ];

        for (gi, (_, members)) in self.groups.iter().enumerate() {
            let color = palette[gi % palette.len()];
            for &streamline_idx in members {
                let si = streamline_idx as usize;
                if si + 1 < self.offsets.len() {
                    let start = self.offsets[si] as usize;
                    let end = self.offsets[si + 1] as usize;
                    for v in start..end.min(self.nb_vertices) {
                        colors[v] = color;
                    }
                }
            }
        }
        colors
    }

    pub fn center(&self) -> Vec3 {
        (self.bbox_min + self.bbox_max) * 0.5
    }

    pub fn extent(&self) -> f32 {
        (self.bbox_max - self.bbox_min).length()
    }
}

/// Read a single-column data array as f32, handling dtype conversion.
fn read_scalar_as_f32<P: TrxScalar + Pod>(
    trx: &TrxFile<P>,
    name: &str,
    dtype: &DType,
    is_dpv: bool,
) -> Option<Vec<f32>> {
    macro_rules! try_read {
        ($t:ty, $convert:expr) => {{
            let view = if is_dpv {
                trx.dpv::<$t>(name).ok()?
            } else {
                trx.dps::<$t>(name).ok()?
            };
            if view.ncols() != 1 {
                return None;
            }
            Some(view.rows().map(|r| $convert(r[0])).collect())
        }};
    }

    match dtype {
        DType::Float32 => try_read!(f32, |v: f32| v),
        DType::Float64 => try_read!(f64, |v: f64| v as f32),
        DType::Float16 => try_read!(half::f16, |v: half::f16| v.to_f32()),
        DType::UInt8 => try_read!(u8, |v: u8| v as f32),
        DType::UInt16 => try_read!(u16, |v: u16| v as f32),
        DType::UInt32 => try_read!(u32, |v: u32| v as f32),
        DType::Int8 => try_read!(i8, |v: i8| v as f32),
        DType::Int16 => try_read!(i16, |v: i16| v as f32),
        DType::Int32 => try_read!(i32, |v: i32| v as f32),
        _ => None,
    }
}

/// Extract DPV, DPS, and group data from a typed TRX file.
fn extract_metadata<P: TrxScalar + Pod>(
    trx: &TrxFile<P>,
    nb_vertices: usize,
    nb_streamlines: usize,
) -> (
    Vec<(String, Vec<f32>)>,
    Vec<(String, Vec<f32>)>,
    Vec<(String, Vec<u32>)>,
) {
    let mut dpv_data = Vec::new();
    for name in trx.dpv_names() {
        if let Some(arr) = trx.dpv.get(name) {
            if arr.ncols == 1 {
                if let Some(flat) = read_scalar_as_f32(trx, name, &arr.dtype, true) {
                    if flat.len() == nb_vertices {
                        dpv_data.push((name.to_string(), flat));
                    }
                }
            }
        }
    }

    let mut dps_data = Vec::new();
    for name in trx.dps_names() {
        if let Some(arr) = trx.dps.get(name) {
            if arr.ncols == 1 {
                if let Some(flat) = read_scalar_as_f32(trx, name, &arr.dtype, false) {
                    if flat.len() == nb_streamlines {
                        dps_data.push((name.to_string(), flat));
                    }
                }
            }
        }
    }

    let mut groups = Vec::new();
    for name in trx.group_names() {
        if let Ok(members) = trx.group(name) {
            groups.push((name.to_string(), members.to_vec()));
        }
    }

    (dpv_data, dps_data, groups)
}

/// Compute direction-RGB colors: abs(normalized local tangent) for each vertex.
fn compute_direction_colors(positions: &[[f32; 3]], offsets: &[u64]) -> Vec<[f32; 4]> {
    let n = positions.len();
    let mut colors = vec![[0.5f32, 0.5, 0.5, 1.0]; n];

    for win in offsets.windows(2) {
        let start = win[0] as usize;
        let end = win[1] as usize;
        if end - start < 2 {
            continue;
        }

        for i in start..end {
            let tangent = if i == start {
                Vec3::from(positions[i + 1]) - Vec3::from(positions[i])
            } else if i == end - 1 {
                Vec3::from(positions[i]) - Vec3::from(positions[i - 1])
            } else {
                Vec3::from(positions[i + 1]) - Vec3::from(positions[i - 1])
            };

            let t = tangent.normalize_or_zero().abs();
            colors[i] = [t.x, t.y, t.z, 1.0];
        }
    }

    colors
}

/// Build line-segment indices for PrimitiveTopology::LineList.
fn build_line_indices(offsets: &[u64]) -> Vec<u32> {
    let mut indices = Vec::new();
    for win in offsets.windows(2) {
        let start = win[0] as u32;
        let end = win[1] as u32;
        for i in start..end.saturating_sub(1) {
            indices.push(i);
            indices.push(i + 1);
        }
    }
    indices
}

/// Map scalar values to a blue-red colormap.
fn scalar_to_colors(values: &[f32]) -> Vec<[f32; 4]> {
    let mut min_v = f32::INFINITY;
    let mut max_v = f32::NEG_INFINITY;
    for &v in values {
        if v.is_finite() {
            min_v = min_v.min(v);
            max_v = max_v.max(v);
        }
    }
    let range = (max_v - min_v).max(1e-10);

    values
        .iter()
        .map(|&v| {
            let t = ((v - min_v) / range).clamp(0.0, 1.0);
            // Blue (0) → White (0.5) → Red (1) colormap
            if t < 0.5 {
                let s = t * 2.0;
                [s, s, 1.0, 1.0]
            } else {
                let s = (1.0 - t) * 2.0;
                [1.0, s, s, 1.0]
            }
        })
        .collect()
}

/// Expand per-streamline scalar values to per-vertex colors.
fn expand_dps_to_vertices(
    dps_values: &[f32],
    offsets: &[u64],
    nb_vertices: usize,
) -> Vec<[f32; 4]> {
    // First expand to per-vertex scalars
    let mut per_vertex = vec![0.0f32; nb_vertices];
    for (si, &val) in dps_values.iter().enumerate() {
        if si + 1 < offsets.len() {
            let start = offsets[si] as usize;
            let end = offsets[si + 1] as usize;
            for vi in start..end.min(nb_vertices) {
                per_vertex[vi] = val;
            }
        }
    }
    scalar_to_colors(&per_vertex)
}
