use std::path::Path;

use glam::{Mat4, Vec3, Vec4};
use nifti::{IntoNdArray, NiftiObject, ReaderOptions};

/// NIfTI volume data ready for GPU upload.
pub struct NiftiVolume {
    /// 3D voxel data normalized to [0, 1] as f32, stored in [i][j][k] order.
    pub data: Vec<f32>,
    /// Volume dimensions (i, j, k).
    pub dims: [usize; 3],
    /// Voxel-to-RAS+ affine (4x4, column-major for glam).
    pub voxel_to_ras: Mat4,
    /// Intensity range before normalization.
    pub intensity_range: (f32, f32),
}

impl NiftiVolume {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let obj = ReaderOptions::new().read_file(path)?;
        let header = obj.header();

        // Build affine from sform rows (srow_x, srow_y, srow_z)
        // NIfTI stores row-major: each srow is a row of the 4x4 affine
        let sx = header.srow_x;
        let sy = header.srow_y;
        let sz = header.srow_z;
        let voxel_to_ras = Mat4::from_cols(
            Vec4::new(sx[0], sy[0], sz[0], 0.0),
            Vec4::new(sx[1], sy[1], sz[1], 0.0),
            Vec4::new(sx[2], sy[2], sz[2], 0.0),
            Vec4::new(sx[3], sy[3], sz[3], 1.0),
        );

        // Get volume dimensions from header dim array
        let dims = [
            header.dim[1] as usize,
            header.dim[2] as usize,
            header.dim[3] as usize,
        ];

        // Convert volume to f32 ndarray
        let volume = obj.into_volume();
        let array = volume.into_ndarray::<f32>()?;

        // Find intensity range
        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;
        for &v in array.iter() {
            if v.is_finite() {
                min_val = min_val.min(v);
                max_val = max_val.max(v);
            }
        }

        // Normalize to [0, 1]
        let range = max_val - min_val;
        let data: Vec<f32> = if range > 0.0 {
            array
                .iter()
                .map(|&v| ((v - min_val) / range).clamp(0.0, 1.0))
                .collect()
        } else {
            vec![0.0; array.len()]
        };

        Ok(Self {
            data,
            dims,
            voxel_to_ras,
            intensity_range: (min_val, max_val),
        })
    }

    /// Transform a voxel coordinate to RAS+ world space.
    pub fn voxel_to_world(&self, voxel: Vec3) -> Vec3 {
        let v = self.voxel_to_ras * voxel.extend(1.0);
        Vec3::new(v.x, v.y, v.z)
    }

    /// Compute the 4 corner positions (in RAS+ space) of an axial slice at voxel index k.
    ///
    /// The NIfTI affine maps integer voxel indices to the *center* of each voxel.
    /// So the outer edges of the slice quad span from voxel -0.5 to dim-0.5 in
    /// each in-plane axis, ensuring the quad covers the full extent of all voxels.
    /// The slice itself sits at voxel center k (not offset).
    pub fn axial_slice_corners(&self, k: usize) -> [Vec3; 4] {
        let kf = k as f32;
        let i0 = -0.5;
        let i1 = self.dims[0] as f32 - 0.5;
        let j0 = -0.5;
        let j1 = self.dims[1] as f32 - 0.5;
        [
            self.voxel_to_world(Vec3::new(i0, j0, kf)),
            self.voxel_to_world(Vec3::new(i1, j0, kf)),
            self.voxel_to_world(Vec3::new(i1, j1, kf)),
            self.voxel_to_world(Vec3::new(i0, j1, kf)),
        ]
    }

    /// Compute the 4 corner positions of a coronal slice at voxel index j.
    pub fn coronal_slice_corners(&self, j: usize) -> [Vec3; 4] {
        let jf = j as f32;
        let i0 = -0.5;
        let i1 = self.dims[0] as f32 - 0.5;
        let k0 = -0.5;
        let k1 = self.dims[2] as f32 - 0.5;
        [
            self.voxel_to_world(Vec3::new(i0, jf, k0)),
            self.voxel_to_world(Vec3::new(i1, jf, k0)),
            self.voxel_to_world(Vec3::new(i1, jf, k1)),
            self.voxel_to_world(Vec3::new(i0, jf, k1)),
        ]
    }

    /// Compute the 4 corner positions of a sagittal slice at voxel index i.
    pub fn sagittal_slice_corners(&self, i: usize) -> [Vec3; 4] {
        let if_ = i as f32;
        let j0 = -0.5;
        let j1 = self.dims[1] as f32 - 0.5;
        let k0 = -0.5;
        let k1 = self.dims[2] as f32 - 0.5;
        [
            self.voxel_to_world(Vec3::new(if_, j0, k0)),
            self.voxel_to_world(Vec3::new(if_, j1, k0)),
            self.voxel_to_world(Vec3::new(if_, j1, k1)),
            self.voxel_to_world(Vec3::new(if_, j0, k1)),
        ]
    }
}
