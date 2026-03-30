use glam::Vec3;
use std::path::Path;

use crate::data::trx_data::ColorMode;
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

/// Returns `Some((min, max))` when the color mode is scalar and auto-range is off,
/// otherwise `None` so `recolor` will auto-detect the range from the data.
pub(super) fn scalar_range_opt(
    mode: &ColorMode,
    auto: bool,
    min: f32,
    max: f32,
) -> Option<(f32, f32)> {
    if auto {
        return None;
    }
    match mode {
        ColorMode::Dpv(_) | ColorMode::Dps(_) => Some((min, max)),
        _ => None,
    }
}

pub(super) fn robust_range(values: &[f32]) -> (f32, f32) {
    let mut finite: Vec<f32> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.is_empty() {
        return (0.0, 1.0);
    }
    finite.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = finite.len();
    let lo_idx = ((n as f32) * 0.02).floor() as usize;
    let hi_idx = ((n as f32) * 0.98).floor() as usize;
    let lo = finite[lo_idx.min(n - 1)];
    let hi = finite[hi_idx.min(n - 1)].max(lo + 1e-6);
    (lo, hi)
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

impl super::TrxViewerApp {
    /// Draw the sphere query as a circle on a slice view.
    pub(super) fn draw_sphere_circle(
        &self,
        ui: &egui::Ui,
        rect: egui::Rect,
        axis_index: usize,
        view_proj: glam::Mat4,
        slice_pos: f32,
    ) {
        // Get the sphere center coordinate along this slice's normal axis
        let center_on_axis = match axis_index {
            0 => self.sphere_center.z, // axial
            1 => self.sphere_center.y, // coronal
            _ => self.sphere_center.x, // sagittal
        };

        let d = (slice_pos - center_on_axis).abs();
        if d >= self.sphere_radius {
            return;
        }

        // Circle radius on this slice plane
        let circle_r = (self.sphere_radius * self.sphere_radius - d * d).sqrt();

        // Project sphere center to screen
        let clip = view_proj * self.sphere_center.extend(1.0);
        let ndc_x = clip.x / clip.w;
        let ndc_y = clip.y / clip.w;
        let sx = rect.left() + (ndc_x + 1.0) * 0.5 * rect.width();
        let sy = rect.top() + (1.0 - ndc_y) * 0.5 * rect.height();

        // Convert world-space radius to screen pixels
        // Use a point offset by circle_r in the first in-plane axis
        let offset_world = match axis_index {
            0 => self.sphere_center + Vec3::new(circle_r, 0.0, 0.0),
            1 => self.sphere_center + Vec3::new(circle_r, 0.0, 0.0),
            _ => self.sphere_center + Vec3::new(0.0, circle_r, 0.0),
        };
        let clip2 = view_proj * offset_world.extend(1.0);
        let ndc_x2 = clip2.x / clip2.w;
        let ndc_y2 = clip2.y / clip2.w;
        let sx2 = rect.left() + (ndc_x2 + 1.0) * 0.5 * rect.width();
        let sy2 = rect.top() + (1.0 - ndc_y2) * 0.5 * rect.height();
        let screen_r = ((sx2 - sx).powi(2) + (sy2 - sy).powi(2)).sqrt();

        let painter = ui.painter_at(rect);
        let circle_color = egui::Color32::from_rgba_unmultiplied(255, 255, 0, 200);
        painter.circle_stroke(
            egui::pos2(sx, sy),
            screen_r,
            egui::Stroke::new(2.0, circle_color),
        );
    }

    /// Draw three axis-aligned circles in the 3D view to indicate the sphere query position.
    pub(super) fn draw_sphere_3d(&self, ui: &egui::Ui, rect: egui::Rect, view_proj: glam::Mat4) {
        let painter = ui.painter_at(rect);
        let color = egui::Color32::from_rgba_unmultiplied(255, 255, 0, 200);
        let stroke = egui::Stroke::new(1.5, color);
        let n = 48usize;
        let c = self.sphere_center;
        let r = self.sphere_radius;

        let project = |p: glam::Vec3| -> egui::Pos2 {
            let clip = view_proj * p.extend(1.0);
            let nx = clip.x / clip.w;
            let ny = clip.y / clip.w;
            egui::pos2(
                rect.left() + (nx + 1.0) * 0.5 * rect.width(),
                rect.top() + (1.0 - ny) * 0.5 * rect.height(),
            )
        };

        // Three rings: XY (axial plane), XZ (coronal), YZ (sagittal)
        let ring_points = |axis_a: glam::Vec3, axis_b: glam::Vec3| -> Vec<egui::Pos2> {
            (0..=n)
                .map(|k| {
                    let t = k as f32 / n as f32 * std::f32::consts::TAU;
                    project(c + axis_a * (r * t.cos()) + axis_b * (r * t.sin()))
                })
                .collect()
        };

        for pts in [
            ring_points(glam::Vec3::X, glam::Vec3::Y), // XY plane
            ring_points(glam::Vec3::X, glam::Vec3::Z), // XZ plane
            ring_points(glam::Vec3::Y, glam::Vec3::Z), // YZ plane
        ] {
            for w in pts.windows(2) {
                painter.line_segment([w[0], w[1]], stroke);
            }
        }
        // Small crosshair at center
        let cp = project(c);
        let arm = 6.0;
        painter.line_segment(
            [egui::pos2(cp.x - arm, cp.y), egui::pos2(cp.x + arm, cp.y)],
            stroke,
        );
        painter.line_segment(
            [egui::pos2(cp.x, cp.y - arm), egui::pos2(cp.x, cp.y + arm)],
            stroke,
        );
    }

    /// Draw anatomical orientation labels (R/L/A/P/S/I) on a slice view.
    pub(super) fn draw_orientation_labels(
        &self,
        ui: &egui::Ui,
        rect: egui::Rect,
        _axis_index: usize,
        view_proj: glam::Mat4,
    ) {
        let center = self.volume_center;
        let axis_len = (self.volume_extent * 0.2).max(10.0);

        // Define the 6 anatomical directions as offsets from center.
        let directions: &[(Vec3, &str)] = &[
            (Vec3::X * axis_len, "R"),  // +X = Right
            (-Vec3::X * axis_len, "L"), // -X = Left
            (Vec3::Y * axis_len, "A"),  // +Y = Anterior
            (-Vec3::Y * axis_len, "P"), // -Y = Posterior
            (Vec3::Z * axis_len, "S"),  // +Z = Superior
            (-Vec3::Z * axis_len, "I"), // -Z = Inferior
        ];

        let project = |world: Vec3| -> egui::Pos2 {
            let clip = view_proj * world.extend(1.0);
            if clip.w.abs() < 1e-6 {
                return rect.center();
            }
            let ndc_x = clip.x / clip.w;
            let ndc_y = clip.y / clip.w;
            egui::pos2(
                rect.left() + (ndc_x + 1.0) * 0.5 * rect.width(),
                rect.top() + (1.0 - ndc_y) * 0.5 * rect.height(),
            )
        };

        let painter = ui.painter_at(rect);
        let label_color = egui::Color32::from_rgb(220, 220, 220);
        let font = egui::FontId::proportional(14.0);
        let margin = 16.0;
        let center_screen = project(center);

        // Place labels by projecting a small offset and extending from center to the viewport edge.
        for &(offset, label) in directions {
            let p = project(center + offset);
            let delta = egui::vec2(p.x - center_screen.x, p.y - center_screen.y);
            let len2 = delta.length_sq();
            // Skip look-axis directions that collapse to the center in this view.
            if len2 < 1e-6 {
                continue;
            }
            let dir = delta / len2.sqrt();
            let tx = if dir.x.abs() > 1e-6 {
                ((rect.width() * 0.5 - margin) / dir.x.abs()).abs()
            } else {
                f32::INFINITY
            };
            let ty = if dir.y.abs() > 1e-6 {
                ((rect.height() * 0.5 - margin) / dir.y.abs()).abs()
            } else {
                f32::INFINITY
            };
            let t = tx.min(ty);
            let label_pos = egui::pos2(center_screen.x + dir.x * t, center_screen.y + dir.y * t);

            painter.text(
                label_pos,
                egui::Align2::CENTER_CENTER,
                label,
                font.clone(),
                label_color,
            );
        }
    }

    /// Draw 3D orientation axes in the corner of the 3D viewport.
    pub(super) fn draw_3d_axes(&self, ui: &egui::Ui, rect: egui::Rect, view_proj: glam::Mat4) {
        let painter = ui.painter_at(rect);

        // Place axes in bottom-left corner
        let origin_screen = egui::pos2(rect.left() + 50.0, rect.bottom() - 50.0);
        let axis_length = 30.0;

        let axes = [
            (Vec3::X, "R", egui::Color32::RED),
            (Vec3::Y, "A", egui::Color32::GREEN),
            (Vec3::Z, "S", egui::Color32::from_rgb(80, 120, 255)),
        ];

        for (dir, label, color) in axes {
            // Project the direction vector (just the rotation, no translation)
            let clip0 = view_proj * Vec3::ZERO.extend(1.0);
            let clip1 = view_proj * dir.extend(1.0);
            // Direction in NDC
            let ndc0 = egui::vec2(clip0.x / clip0.w, clip0.y / clip0.w);
            let ndc1 = egui::vec2(clip1.x / clip1.w, clip1.y / clip1.w);
            let dir_ndc = ndc1 - ndc0;
            let dir_screen = egui::vec2(dir_ndc.x, -dir_ndc.y); // flip Y
            let dir_norm = if dir_screen.length() > 0.001 {
                dir_screen / dir_screen.length()
            } else {
                egui::vec2(0.0, 0.0)
            };

            let end = origin_screen + dir_norm * axis_length;
            painter.line_segment([origin_screen, end], egui::Stroke::new(2.0, color));
            painter.text(
                end + dir_norm * 10.0,
                egui::Align2::CENTER_CENTER,
                label,
                egui::FontId::proportional(12.0),
                color,
            );
        }
    }

    /// Draw crosshair lines on a 2D slice view showing the other two slice positions.
    pub(super) fn draw_crosshairs(
        &self,
        ui: &egui::Ui,
        rect: egui::Rect,
        axis_index: usize,
        view_proj: glam::Mat4,
    ) {
        // Get the world-space positions of the other two slices
        let (other_a, other_b) = match axis_index {
            // Axial view: show coronal (Y) and sagittal (X) positions
            0 => (self.slice_world_position(2), self.slice_world_position(1)),
            // Coronal view: show sagittal (X) and axial (Z) positions
            1 => (self.slice_world_position(2), self.slice_world_position(0)),
            // Sagittal view: show coronal (Y) and axial (Z) positions
            _ => (self.slice_world_position(1), self.slice_world_position(0)),
        };

        let slice_pos = self.slice_world_position(axis_index);

        // Create world-space points on the crosshair lines and project them
        // For each crosshair line, we create two points at the extremes of the view
        let far = 10000.0;
        let (h_p1, h_p2, v_p1, v_p2) = match axis_index {
            0 => {
                // Axial: horizontal = coronal(Y), vertical = sagittal(X)
                let y = other_b; // coronal Y position
                let x = other_a; // sagittal X position
                (
                    glam::Vec3::new(-far, y, slice_pos),
                    glam::Vec3::new(far, y, slice_pos),
                    glam::Vec3::new(x, -far, slice_pos),
                    glam::Vec3::new(x, far, slice_pos),
                )
            }
            1 => {
                // Coronal: horizontal = axial(Z), vertical = sagittal(X)
                let z = other_b; // axial Z position
                let x = other_a; // sagittal X position
                (
                    glam::Vec3::new(-far, slice_pos, z),
                    glam::Vec3::new(far, slice_pos, z),
                    glam::Vec3::new(x, slice_pos, -far),
                    glam::Vec3::new(x, slice_pos, far),
                )
            }
            _ => {
                // Sagittal: horizontal = axial(Z), vertical = coronal(Y)
                let z = other_b; // axial Z position
                let y = other_a; // coronal Y position
                (
                    glam::Vec3::new(slice_pos, -far, z),
                    glam::Vec3::new(slice_pos, far, z),
                    glam::Vec3::new(slice_pos, y, -far),
                    glam::Vec3::new(slice_pos, y, far),
                )
            }
        };

        let project = |world: glam::Vec3| -> egui::Pos2 {
            let clip = view_proj * world.extend(1.0);
            let ndc_x = clip.x / clip.w;
            let ndc_y = clip.y / clip.w;
            // NDC [-1,1] → screen rect
            let sx = rect.left() + (ndc_x + 1.0) * 0.5 * rect.width();
            let sy = rect.top() + (1.0 - ndc_y) * 0.5 * rect.height(); // flip Y
            egui::pos2(sx, sy)
        };

        let crosshair_color = egui::Color32::from_rgba_unmultiplied(255, 255, 0, 128);
        let stroke = egui::Stroke::new(1.0, crosshair_color);
        let painter = ui.painter_at(rect);

        // Horizontal line (clipped to rect)
        painter.line_segment([project(h_p1), project(h_p2)], stroke);
        // Vertical line (clipped to rect)
        painter.line_segment([project(v_p1), project(v_p2)], stroke);
    }

    pub(super) fn slice_world_position(&self, axis_index: usize) -> f32 {
        if let Some(nf) = self.nifti_files.first() {
            let vol = &nf.volume;
            let idx = self.slice_indices[axis_index] as f32;
            let world = match axis_index {
                0 => vol.voxel_to_world(Vec3::new(0.0, 0.0, idx)),
                1 => vol.voxel_to_world(Vec3::new(0.0, idx, 0.0)),
                2 => vol.voxel_to_world(Vec3::new(idx, 0.0, 0.0)),
                _ => Vec3::ZERO,
            };
            match axis_index {
                0 => world.z,
                1 => world.y,
                2 => world.x,
                _ => 0.0,
            }
        } else {
            self.slice_world_offsets[axis_index]
        }
    }

    pub(super) fn draw_mesh_intersections(
        &self,
        ui: &egui::Ui,
        rect: egui::Rect,
        axis_index: usize,
        view_proj: glam::Mat4,
        slice_pos: f32,
    ) {
        let any_bundle_mesh = self
            .trx_files
            .iter()
            .any(|t| t.show_bundle_mesh && !t.bundle_meshes_cpu.is_empty());
        if self.gifti_surfaces.is_empty() && !any_bundle_mesh {
            return;
        }
        let painter = ui.painter_at(rect);
        let eps = 1e-4f32;

        let project = |world: glam::Vec3| -> egui::Pos2 {
            let clip = view_proj * world.extend(1.0);
            let ndc_x = clip.x / clip.w;
            let ndc_y = clip.y / clip.w;
            egui::pos2(
                rect.left() + (ndc_x + 1.0) * 0.5 * rect.width(),
                rect.top() + (1.0 - ndc_y) * 0.5 * rect.height(),
            )
        };

        for surface in &self.gifti_surfaces {
            if !surface.visible || surface.opacity <= 0.01 {
                continue;
            }

            // Surface-level early out by axis-aligned bounds.
            let (smin, smax) = match axis_index {
                0 => (surface.data.bbox_min.z, surface.data.bbox_max.z),
                1 => (surface.data.bbox_min.y, surface.data.bbox_max.y),
                _ => (surface.data.bbox_min.x, surface.data.bbox_max.x),
            };
            if slice_pos < smin - eps || slice_pos > smax + eps {
                continue;
            }

            let color = egui::Color32::from_rgba_unmultiplied(
                (surface.color[0].clamp(0.0, 1.0) * 255.0) as u8,
                (surface.color[1].clamp(0.0, 1.0) * 255.0) as u8,
                (surface.color[2].clamp(0.0, 1.0) * 255.0) as u8,
                (surface.opacity.clamp(0.0, 1.0) * 255.0) as u8,
            );
            let stroke = egui::Stroke::new(1.25, color);

            for tri in surface.data.indices.chunks_exact(3) {
                let ia = tri[0] as usize;
                let ib = tri[1] as usize;
                let ic = tri[2] as usize;
                let a = glam::Vec3::from(surface.data.vertices[ia]);
                let b = glam::Vec3::from(surface.data.vertices[ib]);
                let c = glam::Vec3::from(surface.data.vertices[ic]);

                let tmin = tri_axis_value(a, axis_index)
                    .min(tri_axis_value(b, axis_index))
                    .min(tri_axis_value(c, axis_index));
                let tmax = tri_axis_value(a, axis_index)
                    .max(tri_axis_value(b, axis_index))
                    .max(tri_axis_value(c, axis_index));
                if slice_pos < tmin - eps || slice_pos > tmax + eps {
                    continue;
                }

                let mut pts = Vec::with_capacity(3);
                for (p0, p1) in [(a, b), (b, c), (c, a)] {
                    if let Some(p) = intersect_edge_with_slice(p0, p1, axis_index, slice_pos, eps) {
                        if !pts
                            .iter()
                            .any(|q: &glam::Vec3| (*q - p).length_squared() <= eps * eps)
                        {
                            pts.push(p);
                        }
                    }
                }
                if pts.len() < 2 {
                    continue;
                }
                // For rare 3-point cases (vertex on plane), keep the longest segment.
                let (p0, p1) = if pts.len() == 2 {
                    (pts[0], pts[1])
                } else {
                    let mut best = (pts[0], pts[1]);
                    let mut best_d2 = (pts[1] - pts[0]).length_squared();
                    for i in 0..pts.len() {
                        for j in (i + 1)..pts.len() {
                            let d2 = (pts[j] - pts[i]).length_squared();
                            if d2 > best_d2 {
                                best = (pts[i], pts[j]);
                                best_d2 = d2;
                            }
                        }
                    }
                    best
                };

                painter.line_segment([project(p0), project(p1)], stroke);
            }
        }

        // ── Bundle mesh contours ─────────────────────────────────────────────
        for trx in &self.trx_files {
            if !trx.show_bundle_mesh {
                continue;
            }
            let bundle_mesh_opacity = trx.bundle_mesh_opacity;
            for mesh in &trx.bundle_meshes_cpu {
                for tri in mesh.indices.chunks_exact(3) {
                    let va = &mesh.vertices[tri[0] as usize];
                    let vb = &mesh.vertices[tri[1] as usize];
                    let vc = &mesh.vertices[tri[2] as usize];
                    let a = glam::Vec3::from(va.position);
                    let b = glam::Vec3::from(vb.position);
                    let c = glam::Vec3::from(vc.position);

                    let tmin = tri_axis_value(a, axis_index)
                        .min(tri_axis_value(b, axis_index))
                        .min(tri_axis_value(c, axis_index));
                    let tmax = tri_axis_value(a, axis_index)
                        .max(tri_axis_value(b, axis_index))
                        .max(tri_axis_value(c, axis_index));
                    if slice_pos < tmin - eps || slice_pos > tmax + eps {
                        continue;
                    }

                    // Find intersections, interpolating color along each edge.
                    let mut pts: Vec<(glam::Vec3, [f32; 4])> = Vec::with_capacity(2);
                    for (p0, c0, p1, c1) in [
                        (a, va.color, b, vb.color),
                        (b, vb.color, c, vc.color),
                        (c, vc.color, a, va.color),
                    ] {
                        let d0 = tri_axis_value(p0, axis_index) - slice_pos;
                        let d1 = tri_axis_value(p1, axis_index) - slice_pos;
                        if d0.abs() <= eps && d1.abs() <= eps {
                            continue;
                        }
                        let t = if d0.abs() <= eps {
                            0.0
                        } else if d1.abs() <= eps {
                            1.0
                        } else if d0 * d1 < 0.0 {
                            d0 / (d0 - d1)
                        } else {
                            continue;
                        };
                        let pos = p0 + (p1 - p0) * t;
                        let col = [
                            c0[0] + (c1[0] - c0[0]) * t,
                            c0[1] + (c1[1] - c0[1]) * t,
                            c0[2] + (c1[2] - c0[2]) * t,
                            c0[3] + (c1[3] - c0[3]) * t,
                        ];
                        if !pts
                            .iter()
                            .any(|(q, _)| (*q - pos).length_squared() <= eps * eps)
                        {
                            pts.push((pos, col));
                        }
                    }
                    if pts.len() < 2 {
                        continue;
                    }
                    let (p0, col0, p1, col1) = if pts.len() == 2 {
                        (pts[0].0, pts[0].1, pts[1].0, pts[1].1)
                    } else {
                        let mut best = (pts[0].0, pts[0].1, pts[1].0, pts[1].1);
                        let mut best_d2 = (pts[1].0 - pts[0].0).length_squared();
                        for i in 0..pts.len() {
                            for j in (i + 1)..pts.len() {
                                let d2 = (pts[j].0 - pts[i].0).length_squared();
                                if d2 > best_d2 {
                                    best = (pts[i].0, pts[i].1, pts[j].0, pts[j].1);
                                    best_d2 = d2;
                                }
                            }
                        }
                        best
                    };

                    let opacity_u8 = (bundle_mesh_opacity.clamp(0.0, 1.0) * 255.0) as u8;
                    let to_color = |c: [f32; 4]| {
                        egui::Color32::from_rgba_unmultiplied(
                            (c[0].clamp(0.0, 1.0) * 255.0) as u8,
                            (c[1].clamp(0.0, 1.0) * 255.0) as u8,
                            (c[2].clamp(0.0, 1.0) * 255.0) as u8,
                            opacity_u8,
                        )
                    };

                    // Draw with gradient by blending colors at the two endpoints.
                    let mid = (p0 + p1) * 0.5;
                    let mid_col = [
                        (col0[0] + col1[0]) * 0.5,
                        (col0[1] + col1[1]) * 0.5,
                        (col0[2] + col1[2]) * 0.5,
                        (col0[3] + col1[3]) * 0.5,
                    ];
                    painter.line_segment(
                        [project(p0), project(mid)],
                        egui::Stroke::new(1.5, to_color(col0)),
                    );
                    painter.line_segment(
                        [project(mid), project(p1)],
                        egui::Stroke::new(1.5, to_color(mid_col)),
                    );
                    let _ = (col1, mid_col);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_foreign_tractograms_for_import() {
        assert_eq!(
            classify_dropped_path(Path::new("bundle.tck.gz")),
            DroppedPathKind::ImportTractogram(Format::Tck)
        );
        assert_eq!(
            classify_dropped_path(Path::new("bundle.vtk")),
            DroppedPathKind::ImportTractogram(Format::Vtk)
        );
        assert_eq!(
            classify_dropped_path(Path::new("bundle.tt.gz")),
            DroppedPathKind::ImportTractogram(Format::TinyTrack)
        );
    }
}
