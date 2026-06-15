// WGSL Fragment Shader - learnable_heat_render.wgsl
// =============================================================================

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
}

struct KnotPoint {
    lightness: f32,
    saturation: f32,
    hue: f32,
    _pad: f32,
}

@group(0) @binding(0) var<uniform> knots: array<KnotPoint, 5>; // 5-knot learnable colormap
@group(0) @binding(1) var<uniform> min_max_T: vec2<f32>;       // x = minT, y = maxT
@group(1) @binding(0) var t_heat: texture_2d<f32>;             // Temperature texture from compute pipeline
@group(1) @binding(1) var s_heat: sampler;

// Fast conversion from Okhsl to RGB on the GPU
fn okhsl_to_rgb(h: f32, s: f32, l: f32) -> vec3<f32> {
    let c = (1.0f - abs(2.0f * l - 1.0f)) * s;
    let x = c * (1.0f - abs(mod_val((h / 60.0f), 2.0f) - 1.0f));
    let m = l - c / 2.0f;
    
    var rgb = vec3<f32>(0.0f);
    let h_segment = u32(floor(h / 60.0f)) % 6u;
    switch h_segment {
        case 0u: { rgb = vec3<f32>(c, x, 0.0f); }
        case 1u: { rgb = vec3<f32>(x, c, 0.0f); }
        case 2u: { rgb = vec3<f32>(0.0f, c, x); }
        case 3u: { rgb = vec3<f32>(0.0f, x, c); }
        case 4u: { rgb = vec3<f32>(x, 0.0f, c); }
        default: { rgb = vec3<f32>(c, 0.0f, x); }
    }
    return rgb + vec3<f32>(m);
}

fn mod_val(x: f32, y: f32) -> f32 {
    return x - y * floor(x / y);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let T = textureSample(t_heat, s_heat, in.tex_coords).r;
    let range = clamp((T - min_max_T.x) / (min_max_T.y - min_max_T.x), 0.0f, 1.0f);
    
    // Linearly interpolate between colormap knots on the GPU
    let segment = range * 4.0f;
    let idx = u32(floor(segment));
    let ratio = segment - floor(segment);
    
    let k0 = knots[idx];
    let k1 = knots[idx + 1u];
    
    let lightness  = k0.lightness  * (1.0f - ratio) + k1.lightness  * ratio;
    let saturation = k0.saturation * (1.0f - ratio) + k1.saturation * ratio;
    
    // Shortest-path Hue interpolation in WGSL
    let h0 = k0.hue;
    let h1 = k1.hue;
    var diff = h1 - h0;
    diff = diff - 360.0f * floor((diff + 180.0f) / 360.0f);
    let hue = mod_val(h0 + diff * ratio, 360.0f);
    
    let rgb = okhsl_to_rgb(hue, saturation, lightness);
    return vec4<f32>(rgb, 1.0f);
}
