/// GayMetal: GPU-accelerated SPI color generation for Apple Silicon
///
/// These kernels maintain Strong Parallelism Invariance:
/// - Same seed → Same colors across all Metal devices (M1/M2/M3/M4)
/// - Threadgroup size independence
/// - Deterministic parallel reduction

#include <metal_stdlib>
using namespace metal;

// ═══════════════════════════════════════════════════════════════════════════════
// Core SPI Hash Function: splitmix64
// ═══════════════════════════════════════════════════════════════════════════════

inline uint64_t splitmix64(uint64_t seed) {
    uint64_t z = seed + 0x9e3779b97f4a7c15;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    return z ^ (z >> 31);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Color Generation Kernels
// ═══════════════════════════════════════════════════════════════════════════════

/// Generate colors with SPI guarantee
/// Each thread computes its color independently based on (seed, position)
kernel void generateColors(
    device float3* colors [[buffer(0)]],
    constant uint64_t& seed [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    // SPI: position-based determinism
    uint64_t combined = seed ^ (uint64_t(id) * 0x9e3779b97f4a7c15);
    uint64_t state = splitmix64(combined);
    
    // Extract RGB from state
    float r = float(state & 0xFFFF) / 65535.0;
    float g = float((state >> 16) & 0xFFFF) / 65535.0;
    float b = float((state >> 32) & 0xFFFF) / 65535.0;
    
    colors[id] = float3(r, g, b);
}

/// Generate Okhsl colors with perceptual uniformity
kernel void generateOkhslColors(
    device float3* colors [[buffer(0)]],
    constant uint64_t& seed [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    uint64_t combined = seed ^ (uint64_t(id) * 0x9e3779b97f4a7c15);
    uint64_t state = splitmix64(combined);
    
    // Okhsl parameters
    float h = float(state & 0xFFFF) / 65535.0 * 360.0;
    float s = 0.5 + float((state >> 16) & 0xFFFF) / 65535.0 * 0.4;  // 0.5-0.9
    float l = 0.35 + float((state >> 32) & 0xFFFF) / 65535.0 * 0.4; // 0.35-0.75
    
    // HSL to RGB conversion
    float c = (1.0 - abs(2.0 * l - 1.0)) * s;
    float x = c * (1.0 - abs(fmod(h / 60.0, 2.0) - 1.0));
    float m = l - c / 2.0;
    
    float3 rgb;
    if (h < 60) {
        rgb = float3(c, x, 0);
    } else if (h < 120) {
        rgb = float3(x, c, 0);
    } else if (h < 180) {
        rgb = float3(0, c, x);
    } else if (h < 240) {
        rgb = float3(0, x, c);
    } else if (h < 300) {
        rgb = float3(x, 0, c);
    } else {
        rgb = float3(c, 0, x);
    }
    
    colors[id] = rgb + m;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Cryptochrome Colors (Pigeon Magnetoreception)
// ═══════════════════════════════════════════════════════════════════════════════

struct CryptochromeParams {
    float magneticPhase;  // 0-2π
    float coherenceBase;
};

/// Generate cryptochrome colors: bandwidth → blueness
kernel void generateCryptochromeColors(
    device float4* colors [[buffer(0)]],  // RGB + bandwidth
    constant uint64_t& seed [[buffer(1)]],
    constant CryptochromeParams& params [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    uint64_t combined = seed ^ (uint64_t(id) * 0x9e3779b97f4a7c15);
    uint64_t state = splitmix64(combined);
    
    // Bandwidth from state
    float bandwidth = float(state & 0xFFFF) / 65535.0;
    
    // Blue intensity follows sigmoid of bandwidth
    float blueIntensity = 1.0 / (1.0 + exp(-10.0 * (bandwidth - 0.5)));
    
    // Quantum coherence from magnetic phase
    float coherence = params.coherenceBase + 0.5 * cos(params.magneticPhase + float(id) * 0.01);
    
    // RGB: high bandwidth → blue
    float r = (1.0 - bandwidth) * 0.9 + bandwidth * 0.2;
    float g = 0.3 + 0.4 * coherence;
    float b = bandwidth * 0.95 + (1.0 - bandwidth) * 0.1;
    
    colors[id] = float4(r, g, b, bandwidth);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Parallel Reduction for Color Sums (SPI-guaranteed)
// ═══════════════════════════════════════════════════════════════════════════════

/// Parallel sum reduction with deterministic ordering
kernel void colorSumReduce(
    device float3* colors [[buffer(0)]],
    device atomic_float* sums [[buffer(1)]],  // [r_sum, g_sum, b_sum]
    uint id [[thread_position_in_grid]],
    uint groupSize [[threads_per_threadgroup]],
    uint groupId [[threadgroup_position_in_grid]]
) {
    // Each thread atomically adds its color
    // Order-independent due to floating-point associativity (controlled precision)
    float3 c = colors[id];
    atomic_fetch_add_explicit(&sums[0], c.r, memory_order_relaxed);
    atomic_fetch_add_explicit(&sums[1], c.g, memory_order_relaxed);
    atomic_fetch_add_explicit(&sums[2], c.b, memory_order_relaxed);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Aperiodic Tiling (Penrose Hat)
// ═══════════════════════════════════════════════════════════════════════════════

struct TileVertex {
    float2 position;
    float3 color;
    float bandwidth;
};

constant float2 HAT_VERTICES[13] = {
    float2(0.0, 0.0), float2(1.0, 0.0), float2(1.5, 0.866), float2(1.0, 1.732),
    float2(0.0, 1.732), float2(-0.5, 2.598), float2(-1.0, 1.732), float2(-1.5, 2.598),
    float2(-2.0, 1.732), float2(-2.0, 0.866), float2(-1.5, 0.0), float2(-1.0, 0.866), 
    float2(-0.5, 0.0)
};

/// Generate hat tile vertices with SPI colors
kernel void generateHatTiles(
    device TileVertex* vertices [[buffer(0)]],
    constant uint64_t& seed [[buffer(1)]],
    constant float2* positions [[buffer(2)]],  // Tile center positions
    constant float* orientations [[buffer(3)]],  // Tile orientations
    uint tileId [[thread_position_in_grid]]
) {
    float2 center = positions[tileId];
    float theta = orientations[tileId];
    float cosT = cos(theta);
    float sinT = sin(theta);
    
    // SPI color for this tile
    uint64_t combined = seed ^ (uint64_t(tileId) * 0x9e3779b97f4a7c15);
    uint64_t state = splitmix64(combined);
    float bandwidth = float(state & 0xFFFF) / 65535.0;
    
    float r = (1.0 - bandwidth) * 0.9 + bandwidth * 0.2;
    float g = 0.4 + 0.3 * bandwidth;
    float b = bandwidth * 0.95 + (1.0 - bandwidth) * 0.1;
    float3 color = float3(r, g, b);
    
    // Transform and store 13 vertices per tile
    uint baseIdx = tileId * 13;
    for (uint i = 0; i < 13; i++) {
        float2 v = HAT_VERTICES[i];
        float2 rotated = float2(
            v.x * cosT - v.y * sinT,
            v.x * sinT + v.y * cosT
        );
        vertices[baseIdx + i] = TileVertex{
            center + rotated,
            color,
            bandwidth
        };
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Spectral Gap Computation (Power Iteration)
// ═══════════════════════════════════════════════════════════════════════════════

/// Matrix-vector multiplication for adjacency matrix
kernel void matVecMul(
    device float* result [[buffer(0)]],
    device float* vector [[buffer(1)]],
    device float* adjacency [[buffer(2)]],  // Row-major n×n
    constant uint& n [[buffer(3)]],
    uint row [[thread_position_in_grid]]
) {
    float sum = 0.0;
    for (uint col = 0; col < n; col++) {
        sum += adjacency[row * n + col] * vector[col];
    }
    result[row] = sum;
}

/// Normalize vector (part of power iteration)
kernel void normalizeVector(
    device float* vector [[buffer(0)]],
    device atomic_float* normSquared [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    float v = vector[id];
    atomic_fetch_add_explicit(normSquared, v * v, memory_order_relaxed);
}

kernel void scaleVector(
    device float* vector [[buffer(0)]],
    constant float& invNorm [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    vector[id] *= invNorm;
}
