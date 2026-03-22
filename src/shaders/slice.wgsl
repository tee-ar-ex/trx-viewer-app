struct Uniforms {
    view_proj: mat4x4<f32>,
    window_center: f32,
    window_width: f32,
    _pad0: f32,
    _pad1: f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var volume_texture: texture_3d<f32>;
@group(0) @binding(2) var volume_sampler: sampler;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coord: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coord: vec3<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = uniforms.view_proj * vec4<f32>(in.position, 1.0);
    out.tex_coord = in.tex_coord;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let intensity = textureSample(volume_texture, volume_sampler, in.tex_coord).r;
    // Apply windowing
    let lo = uniforms.window_center - uniforms.window_width * 0.5;
    let hi = uniforms.window_center + uniforms.window_width * 0.5;
    let val = clamp((intensity - lo) / max(hi - lo, 0.001), 0.0, 1.0);
    return vec4<f32>(val, val, val, 1.0);
}
