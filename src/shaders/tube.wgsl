struct Uniforms {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    render_style: u32,
    slab_axis: u32,
    slab_min: f32,
    slab_max: f32,
    tube_radius: f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) p0: vec3<f32>,
    @location(1) p1: vec3<f32>,
    @location(2) color: vec4<f32>,
    @location(3) uv: vec2<f32>,  // (lateral: -1..1, end: -1=p0 side, +1=p1 side)
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) world_pos: vec3<f32>,
    @location(2) tangent: vec3<f32>,
    // Capsule parameterisation:
    //   cap_uv.x = lateral (same as in.uv.x, -1..1)
    //   cap_uv.y = t_ext:  0..1 = cylinder, <0 = p0 cap, >1 = p1 cap
    @location(3) cap_uv: vec2<f32>,
    @location(4) cap_scale: f32,   // tube_radius / seg_len
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    let seg = in.p1 - in.p0;
    let seg_len = length(seg);
    let t_hat = select(normalize(seg), vec3<f32>(1.0, 0.0, 0.0), seg_len < 0.0001);
    let cap_scale = uniforms.tube_radius / max(seg_len, 0.0001);

    // Remap uv.y from [-1,1] to [-cap_scale, 1+cap_scale]
    // so the billboard extends one tube_radius past each endpoint.
    let raw_t = (in.uv.y + 1.0) * 0.5;                              // 0..1
    let t_ext  = raw_t * (1.0 + 2.0 * cap_scale) - cap_scale;       // -cap_scale..(1+cap_scale)
    let pos_along = in.p0 + t_hat * (t_ext * seg_len);

    // Right vector: perpendicular to tube and view direction
    let to_cam = normalize(uniforms.camera_pos - pos_along);
    let cross_len = length(cross(t_hat, to_cam));
    var right: vec3<f32>;
    if cross_len < 0.001 {
        right = normalize(cross(t_hat, vec3<f32>(0.0, 0.0, 1.0)));
    } else {
        right = cross(t_hat, to_cam) / cross_len;
    }

    let world_pos = pos_along + right * in.uv.x * uniforms.tube_radius;

    out.clip_position = uniforms.view_proj * vec4<f32>(world_pos, 1.0);
    out.color         = in.color;
    out.world_pos     = world_pos;
    out.tangent       = t_hat;
    out.cap_uv        = vec2<f32>(in.uv.x, t_ext);
    out.cap_scale     = cap_scale;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Slab clipping
    if uniforms.slab_axis < 3u {
        var coord: f32;
        if uniforms.slab_axis == 0u      { coord = in.world_pos.x; }
        else if uniforms.slab_axis == 1u { coord = in.world_pos.y; }
        else                             { coord = in.world_pos.z; }
        if coord < uniforms.slab_min || coord > uniforms.slab_max { discard; }
    }

    let lateral    = in.cap_uv.x;
    let t_ext      = in.cap_uv.y;
    let cap_scale  = in.cap_scale;

    let t_hat    = normalize(in.tangent);
    let view_dir = normalize(uniforms.camera_pos - in.world_pos);
    let right    = normalize(cross(t_hat, view_dir));

    var normal: vec3<f32>;

    if t_ext >= 0.0 && t_ext <= 1.0 {
        // ── Cylinder region ──────────────────────────────────────────────────
        if lateral * lateral > 1.0 { discard; }
        // Reconstruct outward-facing cylinder normal: lateral component + depth component
        let n_cam = sqrt(max(0.0, 1.0 - lateral * lateral));
        normal = normalize(right * lateral + view_dir * n_cam);
    } else {
        // ── Spherical cap region ──────────────────────────────────────────────
        // along_cap: normalised distance from the nearest endpoint (0 = at endpoint)
        let past_p0 = max(0.0, -t_ext);
        let past_p1 = max(0.0, t_ext - 1.0);
        let along_cap = (past_p0 + past_p1) / cap_scale;  // 0..1 from base to tip

        if lateral * lateral + along_cap * along_cap > 1.0 { discard; }

        // Cap sign: +1 for p1 cap (normal faces +tangent), -1 for p0 cap
        let cap_sign = select(-1.0, 1.0, t_ext > 1.0);
        let n_cam    = sqrt(max(0.0, 1.0 - lateral * lateral - along_cap * along_cap));

        normal = normalize(right * lateral + t_hat * (along_cap * cap_sign) + view_dir * n_cam);
    }

    // Phong shading (shared between cylinder and cap)
    let light_dir   = normalize(vec3<f32>(0.6, 0.8, 1.0));
    let diffuse     = max(0.0, dot(normal, light_dir));
    let reflect_dir = reflect(-light_dir, normal);
    let specular    = pow(max(0.0, dot(reflect_dir, view_dir)), 32.0);

    let lit = 0.3 + 0.55 * diffuse + 0.3 * specular;
    return vec4<f32>(in.color.rgb * lit, in.color.a);
}
