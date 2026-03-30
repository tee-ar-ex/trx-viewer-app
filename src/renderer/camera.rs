use glam::{Mat4, Vec3};

use crate::renderer::slice_renderer::SliceAxis;

/// Orbit camera for the 3D viewport.
///
/// Uses RAS+ neuroimaging convention: X=Right, Y=Anterior, Z=Superior (up).
pub struct OrbitCamera {
    pub center: Vec3,
    pub distance: f32,
    /// Rotation around the Z (superior) axis. 0 = looking from +Y (anterior).
    pub yaw: f32,
    /// Elevation above the horizontal plane. 0 = level, positive = from above.
    pub pitch: f32,
    pub fov_y: f32,
    pub near: f32,
    pub far: f32,
    pub invert_pitch: bool,
}

impl OrbitCamera {
    pub fn new(center: Vec3, distance: f32) -> Self {
        Self {
            center,
            distance,
            // Default: slightly from the right-anterior, slightly above
            yaw: 0.6,
            pitch: 0.4,
            fov_y: std::f32::consts::FRAC_PI_4,
            near: 0.1,
            far: 10000.0,
            invert_pitch: false,
        }
    }

    pub fn eye(&self) -> Vec3 {
        // Spherical coordinates with Z-up
        let horiz = self.distance * self.pitch.cos();
        let x = horiz * self.yaw.sin();
        let y = horiz * self.yaw.cos();
        let z = self.distance * self.pitch.sin();
        self.center + Vec3::new(x, y, z)
    }

    pub fn view_direction(&self) -> Vec3 {
        (self.center - self.eye()).normalize_or_zero()
    }

    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.eye(), self.center, Vec3::Z)
    }

    pub fn projection_matrix(&self, aspect: f32) -> Mat4 {
        Mat4::perspective_rh(self.fov_y, aspect, self.near, self.far)
    }

    pub fn view_projection(&self, aspect: f32) -> Mat4 {
        self.projection_matrix(aspect) * self.view_matrix()
    }

    pub fn handle_drag(&mut self, delta_x: f32, delta_y: f32) {
        let sensitivity = 0.005;
        self.yaw += delta_x * sensitivity;
        let pitch_sign = if self.invert_pitch { -1.0 } else { 1.0 };
        self.pitch += pitch_sign * delta_y * sensitivity;
        let limit = std::f32::consts::FRAC_PI_2 - 0.01;
        self.pitch = self.pitch.clamp(-limit, limit);
    }

    pub fn pan_screen(&mut self, delta_x: f32, delta_y: f32) {
        let eye = self.eye();
        let forward = (self.center - eye).normalize_or_zero();
        let right = forward.cross(Vec3::Z).normalize_or_zero();
        let up = right.cross(forward).normalize_or_zero();
        let scale = self.distance * 0.0015;
        self.center -= right * delta_x * scale;
        self.center += up * delta_y * scale;
    }

    pub fn handle_zoom_drag(&mut self, delta_x: f32, delta_y: f32) {
        let zoom_delta = (-delta_x - delta_y) * 0.01;
        self.handle_scroll(zoom_delta);
    }

    pub fn handle_scroll(&mut self, delta: f32) {
        self.distance *= 1.0 - delta * 0.1;
        self.distance = self.distance.max(0.1);
    }
}

/// Orthographic camera for 2D slice views.
///
/// Uses a rotation angle to orient the view according to neuroimaging conventions.
/// center[0] corresponds to the screen-horizontal axis, center[1] to screen-vertical.
pub struct OrthoSliceCamera {
    pub axis: SliceAxis,
    /// Center of view in screen space: [horizontal, vertical].
    pub center: [f32; 2],
    /// Half-extent of the view (zoom level).
    pub half_extent: f32,
    /// In-plane rotation angle (radians, positive = CCW from viewer's perspective).
    pub rotation: f32,
}

impl OrthoSliceCamera {
    pub fn new(axis: SliceAxis, volume_center: Vec3, volume_extent: f32) -> Self {
        // Rotation angles to match standard neuroimaging conventions:
        //   Axial:    90° CW  = -π/2
        //   Coronal:  90° CCW = +π/2
        //   Sagittal: 90° CW  = -π/2
        let (center, rotation) = match axis {
            // Axial: looking down -Z, screen-right = +X (right), screen-up = +Y (anterior)
            SliceAxis::Axial => ([volume_center.x, volume_center.y], 0.0f32),
            // Coronal: looking along -Y, screen-right = -X (radiological), screen-up = +Z (superior)
            SliceAxis::Coronal => ([volume_center.x, volume_center.z], 0.0f32),
            // Sagittal: looking along +X, screen-right = -Y (posterior), screen-up = +Z (superior)
            SliceAxis::Sagittal => ([volume_center.y, volume_center.z], 0.0f32),
        };
        Self {
            axis,
            center,
            half_extent: volume_extent * 0.5,
            rotation,
        }
    }

    /// Compute view-projection matrix for this slice view.
    /// The slice plane is positioned at the given RAS+ coordinate on the normal axis.
    pub fn view_projection(&self, aspect: f32, slice_position: f32) -> Mat4 {
        let hx = self.half_extent * aspect;
        let hy = self.half_extent;
        // Keep coronal in neurological orientation (R on screen-right), while
        // axial and sagittal already have the desired left-right orientation.
        let flip_lr = matches!(self.axis, SliceAxis::Coronal);
        let (left, right) = if flip_lr { (hx, -hx) } else { (-hx, hx) };

        let projection = Mat4::orthographic_rh(left, right, -hy, hy, -10000.0, 10000.0);

        // View matrix: look along the slice normal axis
        let (eye, target, up) = match self.axis {
            // Axial: looking down -Z (superior to inferior), X=right, Y=up(anterior)
            SliceAxis::Axial => (
                Vec3::new(self.center[0], self.center[1], slice_position + 1.0),
                Vec3::new(self.center[0], self.center[1], slice_position),
                Vec3::Y,
            ),
            // Coronal: looking along -Y (anterior to posterior), X=right, Z=up
            SliceAxis::Coronal => (
                Vec3::new(self.center[0], slice_position + 1.0, self.center[1]),
                Vec3::new(self.center[0], slice_position, self.center[1]),
                Vec3::Z,
            ),
            // Sagittal: looking along +X (left to right), Y=forward, Z=up
            SliceAxis::Sagittal => (
                Vec3::new(slice_position - 1.0, self.center[0], self.center[1]),
                Vec3::new(slice_position, self.center[0], self.center[1]),
                Vec3::Z,
            ),
        };

        let view = Mat4::look_at_rh(eye, target, up);

        // Apply in-plane rotation (around screen Z axis, i.e., the look direction)
        let rot = Mat4::from_rotation_z(self.rotation);
        rot * projection * view
    }

    /// Convert a screen position to RAS+ world coordinates on the slice plane.
    pub fn screen_to_world(
        &self,
        screen_pos: egui::Pos2,
        rect: egui::Rect,
        aspect: f32,
        slice_position: f32,
    ) -> Vec3 {
        // Screen → NDC
        let ndc_x = (screen_pos.x - rect.left()) / rect.width() * 2.0 - 1.0;
        let ndc_y = 1.0 - (screen_pos.y - rect.top()) / rect.height() * 2.0; // flip Y

        // Un-rotate NDC back to the pre-rotation camera frame
        let cos_r = self.rotation.cos();
        let sin_r = self.rotation.sin();
        // Inverse of rotation by self.rotation is rotation by -self.rotation
        let ux = cos_r * ndc_x + sin_r * ndc_y;
        let uy = -sin_r * ndc_x + cos_r * ndc_y;
        // NDC → in-plane world coordinates.
        // Sign depends on which world axis maps to screen-right for each view:
        //   Axial:    right = +X → wx = center[0] + ndc_x * hx
        //   Coronal:  projection is flipped, so right = +X → wx = center[0] + ndc_x * hx
        //   Sagittal: right = -Y → wx = center[0] - ndc_x * hx (center[0] = volume_center.y)
        let hx = self.half_extent * aspect;
        let hy = self.half_extent;
        let wy = self.center[1] + uy * hy;
        match self.axis {
            SliceAxis::Axial => {
                let wx = self.center[0] + ux * hx;
                Vec3::new(wx, wy, slice_position)
            }
            SliceAxis::Coronal => {
                let wx = self.center[0] + ux * hx;
                Vec3::new(wx, slice_position, wy)
            }
            SliceAxis::Sagittal => {
                let wx = self.center[0] - ux * hx;
                Vec3::new(slice_position, wx, wy)
            }
        }
    }
}
