struct Uniforms {
    view_proj:  mat4x4<f32>,
    camera_pos: vec3<f32>,
    opacity:    f32,
    ambient:    f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal:   vec3<f32>,
    @location(2) color:    vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos:    vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) color:        vec4<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = uniforms.view_proj * vec4<f32>(in.position, 1.0);
    out.world_pos     = in.position;
    out.world_normal  = normalize(in.normal);
    out.color         = in.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let V = normalize(uniforms.camera_pos - in.world_pos);
    var N = normalize(in.world_normal);
    // Two-sided lighting.
    if dot(N, V) < 0.0 { N = -N; }
    // Headlight.
    let diff = max(dot(N, V), 0.0);
    let lit  = in.color.rgb * (uniforms.ambient + (1.0 - uniforms.ambient) * diff);
    return vec4<f32>(clamp(lit, vec3<f32>(0.0), vec3<f32>(1.0)), uniforms.opacity);
}
