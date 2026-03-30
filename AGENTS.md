# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Build and Run

```bash
cargo build --release
cargo run --release -- path/to/file.trx path/to/volume.nii.gz
```

Requires the `trx-rs` crate as a sibling directory (`../trx-rs`).

### macOS App Bundle

```bash
cargo build --release
cp target/release/trx-viewer-app "target/release/TRX Viewer.app/Contents/MacOS/TRX Viewer"
touch "target/release/TRX Viewer.app"
```

## Architecture

This is a GPU-accelerated TRX tractography viewer built on egui + wgpu. All coordinates use the **RAS+ neuroimaging convention** (X=Right, Y=Anterior, Z=Superior/up).

### Data flow

Files load via `TrxGpuData::load()` / `NiftiVolume::load()` → GPU resources created (`StreamlineResources` / `SliceResources`) → inserted into `egui_wgpu::CallbackResources` → rendered each frame via `CallbackTrait` implementations (`Scene3DCallback`, `SliceViewCallback`).

### Split position/color buffers

Streamline vertex data is split into separate GPU buffers: positions (uploaded once, 12 bytes/vertex) and colors (re-uploaded on recolor, 16 bytes/vertex). This is critical for performance — recoloring 100k+ streamlines only rewrites the color buffer.

### Multi-viewport bind groups

`SliceResources` maintains 4 uniform buffers/bind groups (index 0=3D, 1=axial, 2=coronal, 3=sagittal) so each viewport gets its own view-projection matrix while sharing one pipeline and one 3D texture.

### Dirty flags

State changes are deferred: `slices_dirty`, `colors_dirty`, `groups_dirty` flags trigger GPU buffer updates at the start of the next `update()` call, avoiding redundant uploads.

### NIfTI affine handling

The NIfTI sform matrix (`srow_x/y/z`) maps voxel indices to RAS+ world coordinates. Quad corners use half-voxel offsets (`-0.5` to `dim-0.5`) to cover full voxel footprints and align with TRX positions.

### DPV/DPS dtype handling

TRX ancillary data arrays can be any dtype (f16, f32, u8, etc.). The `read_scalar_as_f32()` function checks the `DataArray.dtype` field before casting to avoid bytemuck panics — never call `trx.dpv::<f32>()` without verifying the dtype first.

### Depth testing

Enabled via `depth_buffer: 32` in eframe `NativeOptions`, which makes egui-wgpu create a `Depth32Float` attachment on the render pass. Both streamline and slice pipelines include matching `DepthStencilState`.

### Device limits

The wgpu device requests `max_buffer_size: 1 GB` (via `WgpuSetupCreateNew::device_descriptor`) to handle large tractography datasets. The default 256 MB limit is insufficient for 100k+ streamline files.

### Bundle surface meshes

`src/data/bundle_mesh.rs` builds a surface mesh from streamline point clouds via voxel density + marching cubes (`mcubes` crate). Pipeline:

1. **Voxelise** positions into a density grid; accumulate per-voxel RGB sums for color.
2. **Gaussian blur** the density field (separable 3-pass, configurable σ). The MC iso-threshold is scaled by k₀³ (the 3D Gaussian center weight) so the slider always reads in "raw points/voxel" units regardless of σ.
3. **Marching cubes** (`MarchingCubes::new(..., blurred_density, mc_threshold).generate(MeshSide::OutsideOnly)`). Output vertices are in world space.
4. **Color interpolation**: trilinear sample of the (unblurred) voxel color grid at each vertex position.
5. **Largest connected component**: `mcubes` emits non-shared vertices, so adjacency is determined by quantized (0.1 mm) vertex *positions*, not indices.

Sources: `BundleMeshSource::All` / `Selection` / `PerGroup`. Mesh is rebuilt on a background thread with 150 ms debounce; result arrives via `mpsc::channel` as `Vec<(BundleMesh, String)>`. CPU copies are kept in `bundle_meshes_cpu` for slice-panel contour drawing.

**Slice contours** are drawn in `draw_mesh_intersections` via egui painter (CPU triangle-plane intersection, same approach as GIFTI surfaces). Colors are linearly interpolated between the two edge endpoints at the cut point.

**GPU**: `MeshResources` holds a `bundle_pipeline` (`TriangleList`, two-sided, per-vertex RGB) and a `Vec<BundleGpuSurface>` (one per source slot). Uniform struct is 96 bytes: `view_proj` (64) + `camera_pos` (12) + `opacity` (4) + `ambient` (4) + `_pad` (12).
