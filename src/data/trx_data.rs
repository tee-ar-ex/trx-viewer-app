use std::collections::HashSet;
use std::collections::HashMap;
use std::path::Path;

use bytemuck::Pod;
use glam::Vec3;
use trx_rs::{AnyTrxFile, DType, TrxFile, TrxScalar};
use crate::data::gifti_data::GiftiSurfaceData;

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
    /// Per-streamline axis-aligned bounding boxes: [min_x, min_y, min_z, max_x, max_y, max_z].
    pub aabbs: Vec<[f32; 6]>,
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

        // Compute per-streamline AABBs for spatial queries
        let aabbs = build_streamline_aabbs(&positions, &offsets);

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
            aabbs,
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

    /// Build index buffer applying all active filters: group visibility, max count,
    /// ordering (for random subsetting), and optional sphere query.
    pub fn build_index_buffer(
        &self,
        visible_groups: &[bool],
        max_count: usize,
        ordering: &[u32],
        sphere_indices: Option<&HashSet<u32>>,
        surface_indices: Option<&HashSet<u32>>,
    ) -> Vec<u32> {
        let selected = self.filtered_streamline_indices(
            visible_groups,
            max_count,
            ordering,
            sphere_indices,
            surface_indices,
        );

        // Step 4: Build line-segment index pairs
        let mut indices = Vec::with_capacity(selected.len() * 100); // rough estimate
        for &si in &selected {
            let s = si as usize;
            if s + 1 < self.offsets.len() {
                let start = self.offsets[s] as u32;
                let end = self.offsets[s + 1] as u32;
                for j in start..end.saturating_sub(1) {
                    indices.push(j);
                    indices.push(j + 1);
                }
            }
        }
        indices
    }

    /// Return the selected streamline indices after applying active filters.
    pub fn filtered_streamline_indices(
        &self,
        visible_groups: &[bool],
        max_count: usize,
        ordering: &[u32],
        sphere_indices: Option<&HashSet<u32>>,
        surface_indices: Option<&HashSet<u32>>,
    ) -> Vec<u32> {
        // Step 1: Collect visible streamline indices
        let visible_set: Vec<u32> = if self.groups.is_empty() {
            (0..self.nb_streamlines as u32).collect()
        } else {
            let mut v = Vec::new();
            for (i, (_, members)) in self.groups.iter().enumerate() {
                if i < visible_groups.len() && !visible_groups[i] {
                    continue;
                }
                v.extend_from_slice(members);
            }
            v
        };

        // Step 2: Spatial filter intersections
        let filtered: Vec<u32> = visible_set
            .into_iter()
            .filter(|idx| sphere_indices.is_none_or(|set| set.contains(idx)))
            .filter(|idx| surface_indices.is_none_or(|set| set.contains(idx)))
            .collect();

        // Step 3: Apply ordering and max count
        if ordering.len() == self.nb_streamlines {
            let filtered_set: HashSet<u32> = filtered.into_iter().collect();
            ordering
                .iter()
                .copied()
                .filter(|idx| filtered_set.contains(idx))
                .take(max_count)
                .collect()
        } else {
            filtered.into_iter().take(max_count).collect()
        }
    }

    /// Query streamlines intersecting a sphere. Returns matching streamline indices.
    /// Uses AABB broad phase then per-vertex distance check.
    pub fn query_sphere(&self, center: Vec3, radius: f32) -> HashSet<u32> {
        let r2 = radius * radius;
        let sphere_min = center - Vec3::splat(radius);
        let sphere_max = center + Vec3::splat(radius);
        let mut result = HashSet::new();

        for si in 0..self.nb_streamlines {
            let aabb = &self.aabbs[si];
            // AABB broad phase
            if aabb[3] < sphere_min.x || aabb[0] > sphere_max.x
                || aabb[4] < sphere_min.y || aabb[1] > sphere_max.y
                || aabb[5] < sphere_min.z || aabb[2] > sphere_max.z
            {
                continue;
            }
            // Vertex-level narrow phase
            let start = self.offsets[si] as usize;
            let end = self.offsets[si + 1] as usize;
            for vi in start..end {
                let p = Vec3::from(self.positions[vi]);
                if (p - center).length_squared() <= r2 {
                    result.insert(si as u32);
                    break;
                }
            }
        }
        result
    }

    /// Query streamlines that pass within `depth_mm` of the surface.
    /// Uses streamline AABB and surface AABB broad phase, then nearest surface-vertex distance.
    pub fn query_near_surface(
        &self,
        surface: &GiftiSurfaceData,
        depth_mm: f32,
    ) -> HashSet<u32> {
        let mut result = HashSet::new();
        if depth_mm <= 0.0 || surface.vertices.is_empty() {
            return result;
        }
        let depth2 = depth_mm * depth_mm;
        let smin = surface.bbox_min - Vec3::splat(depth_mm);
        let smax = surface.bbox_max + Vec3::splat(depth_mm);
        let grid = SurfaceSpatialGrid::build(&surface.vertices, depth_mm.max(0.5));

        for si in 0..self.nb_streamlines {
            let aabb = self.aabbs[si];
            if !aabb_overlaps_expanded_surface(aabb, smin, smax) {
                continue;
            }
            let start = self.offsets[si] as usize;
            let end = self.offsets[si + 1] as usize;
            for vi in start..end {
                let p = Vec3::from(self.positions[vi]);
                if p.x < smin.x || p.x > smax.x || p.y < smin.y || p.y > smax.y || p.z < smin.z || p.z > smax.z {
                    continue;
                }
                if let Some((_, d2)) = grid.nearest_vertex(&surface.vertices, p) {
                    if d2 <= depth2 {
                        result.insert(si as u32);
                        break;
                    }
                }
            }
        }
        result
    }

    /// Project selected streamlines onto surface vertices.
    /// Returns (density_count, mean_dps_value_or_nan) per surface vertex.
    pub fn project_selected_to_surface(
        &self,
        surface: &GiftiSurfaceData,
        selected_streamlines: &[u32],
        depth_mm: f32,
        dps_values: Option<&[f32]>,
    ) -> (Vec<f32>, Vec<f32>) {
        let n = surface.vertices.len();
        let mut density = vec![0.0f32; n];
        let mut dps_sum = vec![0.0f32; n];
        let mut dps_count = vec![0u32; n];
        if depth_mm <= 0.0 || n == 0 {
            return (density, vec![f32::NAN; n]);
        }

        let depth2 = depth_mm * depth_mm;
        let smin = surface.bbox_min - Vec3::splat(depth_mm);
        let smax = surface.bbox_max + Vec3::splat(depth_mm);
        let grid = SurfaceSpatialGrid::build(&surface.vertices, depth_mm.max(0.5));

        for &si_u32 in selected_streamlines {
            let si = si_u32 as usize;
            if si + 1 >= self.offsets.len() {
                continue;
            }
            let dps_val = dps_values.and_then(|d| d.get(si)).copied().unwrap_or(0.0);
            let start = self.offsets[si] as usize;
            let end = self.offsets[si + 1] as usize;
            for vi in start..end {
                let p = Vec3::from(self.positions[vi]);
                if p.x < smin.x || p.x > smax.x || p.y < smin.y || p.y > smax.y || p.z < smin.z || p.z > smax.z {
                    continue;
                }
                if let Some((nearest_idx, d2)) = grid.nearest_vertex(&surface.vertices, p) {
                    if d2 <= depth2 {
                        density[nearest_idx] += 1.0;
                        if dps_values.is_some() {
                            dps_sum[nearest_idx] += dps_val;
                            dps_count[nearest_idx] += 1;
                        }
                    }
                }
            }
        }

        let mean_dps: Vec<f32> = dps_sum
            .iter()
            .zip(dps_count.iter())
            .map(|(&sum, &cnt)| if cnt > 0 { sum / cnt as f32 } else { f32::NAN })
            .collect();

        (density, mean_dps)
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

/// Compute per-streamline axis-aligned bounding boxes.
fn build_streamline_aabbs(positions: &[[f32; 3]], offsets: &[u64]) -> Vec<[f32; 6]> {
    let nb = offsets.len().saturating_sub(1);
    let mut aabbs = Vec::with_capacity(nb);
    for win in offsets.windows(2) {
        let start = win[0] as usize;
        let end = win[1] as usize;
        let mut min = [f32::INFINITY; 3];
        let mut max = [f32::NEG_INFINITY; 3];
        for vi in start..end {
            let p = &positions[vi];
            for d in 0..3 {
                min[d] = min[d].min(p[d]);
                max[d] = max[d].max(p[d]);
            }
        }
        aabbs.push([min[0], min[1], min[2], max[0], max[1], max[2]]);
    }
    aabbs
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

fn aabb_overlaps_expanded_surface(aabb: [f32; 6], smin: Vec3, smax: Vec3) -> bool {
    !(aabb[3] < smin.x
        || aabb[0] > smax.x
        || aabb[4] < smin.y
        || aabb[1] > smax.y
        || aabb[5] < smin.z
        || aabb[2] > smax.z)
}

struct SurfaceSpatialGrid {
    cell_size: f32,
    buckets: HashMap<(i32, i32, i32), Vec<usize>>,
}

impl SurfaceSpatialGrid {
    fn build(vertices: &[[f32; 3]], cell_size: f32) -> Self {
        let mut buckets: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();
        for (i, v) in vertices.iter().enumerate() {
            let key = Self::key(Vec3::from(*v), cell_size);
            buckets.entry(key).or_default().push(i);
        }
        Self { cell_size, buckets }
    }

    fn key(p: Vec3, cell_size: f32) -> (i32, i32, i32) {
        (
            (p.x / cell_size).floor() as i32,
            (p.y / cell_size).floor() as i32,
            (p.z / cell_size).floor() as i32,
        )
    }

    fn nearest_vertex(&self, vertices: &[[f32; 3]], p: Vec3) -> Option<(usize, f32)> {
        let (cx, cy, cz) = Self::key(p, self.cell_size);
        let mut best: Option<(usize, f32)> = None;
        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let key = (cx + dx, cy + dy, cz + dz);
                    let Some(list) = self.buckets.get(&key) else {
                        continue;
                    };
                    for &idx in list {
                        let d2 = (Vec3::from(vertices[idx]) - p).length_squared();
                        match best {
                            Some((_, best_d2)) if d2 >= best_d2 => {}
                            _ => best = Some((idx, d2)),
                        }
                    }
                }
            }
        }
        best
    }
}
