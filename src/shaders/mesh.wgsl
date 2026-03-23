struct Uniforms {
    view_proj: mat4x4<f32>,
    color: vec4<f32>,
    camera_pos: vec3<f32>,
    shininess: f32,
    specular_strength: f32,
    ambient_strength: f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = uniforms.view_proj * vec4<f32>(in.position, 1.0);
    out.world_pos = in.position;
    out.world_normal = normalize(in.normal);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let N = normalize(in.world_normal);
    let V = normalize(uniforms.camera_pos - in.world_pos);
    // Headlight-style illumination keeps meshes readable under interaction.
    let L = V;
    let H = normalize(L + V);

    let diff = max(dot(N, L), 0.0);
    let spec = pow(max(dot(N, H), 0.0), uniforms.shininess) * uniforms.specular_strength;
    let ambient = uniforms.ambient_strength;

    let lit = uniforms.color.rgb * (ambient + diff) + vec3<f32>(spec);
    return vec4<f32>(clamp(lit, vec3<f32>(0.0), vec3<f32>(1.0)), uniforms.color.a);
}
