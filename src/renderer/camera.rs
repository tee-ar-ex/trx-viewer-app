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
        self.pitch += delta_y * sensitivity;
        let limit = std::f32::consts::FRAC_PI_2 - 0.01;
        self.pitch = self.pitch.clamp(-limit, limit);
    }

    pub fn handle_scroll(&mut self, delta: f32) {
        self.distance *= 1.0 - delta * 0.1;
        self.distance = self.distance.max(0.1);
    }
}

/// Orthographic camera for 2D slice views.
pub struct OrthoSliceCamera {
    pub axis: SliceAxis,
    /// Center of view in RAS+ space (the 2 in-plane coordinates).
    pub center: [f32; 2],
    /// Half-extent of the view (zoom level).
    pub half_extent: f32,
}

impl OrthoSliceCamera {
    pub fn new(axis: SliceAxis, volume_center: Vec3, volume_extent: f32) -> Self {
        // Extract the 2 in-plane coordinates of the volume center
        let center = match axis {
            SliceAxis::Axial => [volume_center.x, volume_center.y],
            SliceAxis::Coronal => [volume_center.x, volume_center.z],
            SliceAxis::Sagittal => [volume_center.y, volume_center.z],
        };
        Self {
            axis,
            center,
            half_extent: volume_extent * 0.5,
        }
    }

    /// Compute view-projection matrix for this slice view.
    /// The slice plane is positioned at the given RAS+ coordinate on the normal axis.
    pub fn view_projection(&self, aspect: f32, slice_position: f32) -> Mat4 {
        let hx = self.half_extent * aspect;
        let hy = self.half_extent;

        let projection = Mat4::orthographic_rh(
            -hx,
            hx,
            -hy,
            hy,
            -10000.0,
            10000.0,
        );

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
        projection * view
    }

    pub fn handle_drag(&mut self, delta_x: f32, delta_y: f32, viewport_width: f32) {
        let scale = self.half_extent * 2.0 / viewport_width.max(1.0);
        self.center[0] -= delta_x * scale;
        self.center[1] -= delta_y * scale;
    }

    pub fn handle_zoom(&mut self, delta: f32) {
        self.half_extent *= 1.0 - delta * 0.1;
        self.half_extent = self.half_extent.max(1.0);
    }
}
