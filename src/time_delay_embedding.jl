# src/time_delay_embedding.jl
# =============================================================================
# Time Delay Embeddings for Chaotic Entropy Dynamics & Color Trajectories
# =============================================================================

"""
    TimeDelayEmbedding

This module implements Time Delay Embeddings based on Takens' Embedding Theorem.
It enables reconstructing a multi-dimensional state space attractor from a 
one-dimensional time series of scalar measurements (e.g. confidence, entropy, or
color HSL/IntrinsicHSL channels from physical entropy sources).

It includes algorithms for:
1. Reconstructing phase space coordinates: y_i = [x_i, x_{i+τ}, ..., x_{i+(d-1)τ}]
2. Computing Autocorrelation for delay estimation
3. Computing Average Mutual Information (AMI) for optimal delay estimation
4. False Nearest Neighbors (FNN) algorithm for optimal dimension estimation
5. Rosenstein's algorithm for Maximum Lyapunov Exponent estimation
6. ColoredTick-specific sequence extraction & embedding reconstruction

Seed 1069 balanced ternary: [+1, -1, -1, +1, +1, +1, +1]
"""
module TimeDelayEmbedding

using Statistics
using LinearAlgebra
using Colors: RGB, HSL
using ..Gay: ColoredTick, TritTick, entropy_mix, color_at, to_intrinsic_hsl, IntrinsicHSL

export DelayEmbedding, reconstruct_phase_space,
       autocorrelation, average_mutual_information,
       false_nearest_neighbors, find_optimal_delay_acf_zero,
       find_optimal_delay_acf_decay, find_optimal_delay_ami,
       find_optimal_dimension, lyapunov_divergence_curve,
       estimate_lyapunov_exponent, embed_colored_ticks

# ═══════════════════════════════════════════════════════════════════════════════
# Core Types
# ═══════════════════════════════════════════════════════════════════════════════

"""
    DelayEmbedding{T}

A struct representing a reconstructed multi-dimensional phase space embedding.

Fields:
- `points`: Matrix{T} of size `(M, dimension)` where `M` is the number of points in the reconstruction,
  and each row corresponds to a coordinate in the reconstructed state space.
- `delay`: The lag time parameter `τ` (in ticks).
- `dimension`: The embedding dimension `d`.
- `series_name`: Symbol identifier of the underlying scalar source channel.
"""
struct DelayEmbedding{T}
    points::Matrix{T}
    delay::Int
    dimension::Int
    series_name::Symbol
end

# ═══════════════════════════════════════════════════════════════════════════════
# Phase Space Reconstruction
# ═══════════════════════════════════════════════════════════════════════════════

"""
    reconstruct_phase_space(series::AbstractVector{T}, delay::Int, dim::Int) -> Matrix{T}

Reconstruct a multi-dimensional phase space from a 1D time series.
Returns a matrix of size `(M, dim)` where `M = length(series) - (dim - 1) * delay`.
"""
function reconstruct_phase_space(series::AbstractVector{T}, delay::Int, dim::Int) where {T<:Real}
    if delay < 1
        throw(ArgumentError("Delay must be at least 1, got $delay"))
    end
    if dim < 1
        throw(ArgumentError("Dimension must be at least 1, got $dim"))
    end
    
    N = length(series)
    M = N - (dim - 1) * delay
    if M <= 0
        throw(ArgumentError("Series length ($N) is too short for dimension $dim and delay $delay. Required length > $((dim-1)*delay)."))
    end
    
    points = Matrix{T}(undef, M, dim)
    for i in 1:M
        for j in 1:dim
            points[i, j] = series[i + (j - 1) * delay]
        end
    end
    return points
end

# ═══════════════════════════════════════════════════════════════════════════════
# Delay Selection (Autocorrelation & Mutual Information)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    autocorrelation(series::AbstractVector{T}, max_lag::Int) -> Vector{Float64}

Compute the sample Autocorrelation Function (ACF) of a 1D series up to `max_lag`.
"""
function autocorrelation(series::AbstractVector{T}, max_lag::Int) where {T<:Real}
    N = length(series)
    if N == 0
        return Float64[]
    end
    if max_lag < 0
        throw(ArgumentError("max_lag must be non-negative, got $max_lag"))
    end
    
    mean_val = mean(series)
    var_val = var(series; corrected=false)
    
    if var_val ≈ 0.0
        return ones(max_lag + 1)
    end
    
    acf = Vector{Float64}(undef, max_lag + 1)
    acf[1] = 1.0  # Lag 0 is always 1.0
    
    for k in 1:max_lag
        if k >= N
            acf[k+1] = 0.0
            continue
        end
        s = 0.0
        for i in 1:(N - k)
            s += (series[i] - mean_val) * (series[i + k] - mean_val)
        end
        acf[k+1] = (s / N) / var_val
    end
    return acf
end

"""
    find_optimal_delay_acf_zero(series::AbstractVector{<:Real}; max_lag::Int) -> Int

Estimate the optimal delay `τ` as the first zero-crossing of the autocorrelation function.
"""
function find_optimal_delay_acf_zero(series::AbstractVector{<:Real}; max_lag::Int=min(100, length(series) ÷ 2))
    acf = autocorrelation(series, max_lag)
    for k in 1:(length(acf)-1)
        if acf[k+1] <= 0.0 && acf[k] > 0.0
            return k  # delay equals the index lag
        end
    end
    # Fallback: Find the first local minimum
    for k in 2:(length(acf)-1)
        if acf[k+1] > acf[k] && acf[k-1] > acf[k]
            return k - 1
        end
    end
    return 1 # Fallback to 1
end

"""
    find_optimal_delay_acf_decay(series::AbstractVector{<:Real}; threshold::Float64, max_lag::Int) -> Int

Estimate the optimal delay `τ` as the first lag where the autocorrelation falls below a threshold (e.g. 1/e).
"""
function find_optimal_delay_acf_decay(series::AbstractVector{<:Real}; threshold::Float64=1/exp(1), max_lag::Int=min(100, length(series) ÷ 2))
    acf = autocorrelation(series, max_lag)
    for k in 1:(length(acf)-1)
        if acf[k+1] < threshold
            return k
        end
    end
    return 1 # Fallback to 1
end

"""
    average_mutual_information(series::AbstractVector{T}, max_lag::Int; bins::Int) -> Vector{Float64}

Compute the Average Mutual Information (AMI) of a series with its lagged versions.
Uses uniform grid binning to compute joint and marginal probability distributions.
"""
function average_mutual_information(series::AbstractVector{T}, max_lag::Int; bins::Int=10) where {T<:Real}
    N = length(series)
    if N <= 1
        return zeros(max_lag + 1)
    end
    if max_lag < 0
        throw(ArgumentError("max_lag must be non-negative, got $max_lag"))
    end
    
    min_val, max_val = extrema(series)
    if min_val ≈ max_val
        return zeros(max_lag + 1)
    end
    
    # Grid bin mapping: [min_val, max_val] -> [1, bins]
    bin_idx(x) = clamp(floor(Int, (x - min_val) / (max_val - min_val) * bins) + 1, 1, bins)
    
    ami = Vector{Float64}(undef, max_lag + 1)
    
    for k in 0:max_lag
        if k >= N - 1
            ami[k+1] = 0.0
            continue
        end
        
        joint_counts = zeros(Int, bins, bins)
        x_counts = zeros(Int, bins)
        y_counts = zeros(Int, bins)
        M = N - k
        
        for i in 1:M
            bx = bin_idx(series[i])
            by = bin_idx(series[i+k])
            joint_counts[bx, by] += 1
            x_counts[bx] += 1
            y_counts[by] += 1
        end
        
        mi = 0.0
        for bx in 1:bins
            for by in 1:bins
                c = joint_counts[bx, by]
                if c > 0
                    px = x_counts[bx] / M
                    py = y_counts[by] / M
                    pxy = c / M
                    mi += pxy * log2(pxy / (px * py))
                end
            end
        end
        ami[k+1] = mi
    end
    return ami
end

"""
    find_optimal_delay_ami(series::AbstractVector{<:Real}; bins::Int, max_lag::Int) -> Int

Estimate the optimal delay `τ` as the first local minimum of the Average Mutual Information function.
"""
function find_optimal_delay_ami(series::AbstractVector{<:Real}; bins::Int=10, max_lag::Int=min(100, length(series) ÷ 2))
    ami = average_mutual_information(series, max_lag; bins=bins)
    # Search for first local minimum
    for k in 2:(length(ami)-1)
        if ami[k+1] > ami[k] && ami[k-1] > ami[k]
            return k - 1
        end
    end
    # Fallback to autocorrelation decay
    return find_optimal_delay_acf_decay(series; max_lag=max_lag)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Dimension Selection (False Nearest Neighbors)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    false_nearest_neighbors(series::AbstractVector{T}, delay::Int, max_dim::Int; R_tol, A_tol) -> Vector{Float64}

Run the False Nearest Neighbors (FNN) algorithm for `d` from 1 to `max_dim - 1`.
Returns the fraction of false neighbors for each candidate dimension.
"""
function false_nearest_neighbors(series::AbstractVector{T}, delay::Int, max_dim::Int;
                                 R_tol::Float64=15.0, A_tol::Float64=2.0) where {T<:Real}
    N = length(series)
    if N <= delay
        return zeros(max_dim - 1)
    end
    
    σ = std(series)
    if σ ≈ 0.0
        return zeros(max_dim - 1)
    end
    
    fnn_fracs = Vector{Float64}(undef, max_dim - 1)
    
    for d in 1:(max_dim - 1)
        pts = reconstruct_phase_space(series, delay, d)
        M = size(pts, 1)
        
        # We can only project points that have a coordinate in the d+1 dimension
        M_next = N - d * delay
        if M_next <= 1
            fnn_fracs[d] = 0.0
            continue
        end
        
        false_neighbors_count = 0
        valid_points_count = 0
        
        for i in 1:M_next
            # Find nearest neighbor of pts[i, :] in d dimensions
            min_dist = Inf
            nn_idx = -1
            
            for j in 1:M_next
                if j == i
                    continue
                end
                
                dist_sq = 0.0
                for k in 1:d
                    diff = pts[i, k] - pts[j, k]
                    dist_sq += diff * diff
                end
                
                if dist_sq < min_dist
                    min_dist = dist_sq
                    nn_idx = j
                end
            end
            
            if nn_idx == -1
                continue
            end
            
            R_d = sqrt(min_dist)
            valid_points_count += 1
            
            # Distance in the d+1 coordinate
            coord_diff = abs(series[i + d * delay] - series[nn_idx + d * delay])
            
            # Criterion 1: Increase in distance is too large relative to R_d
            is_false = false
            if R_d > 1e-10
                if (coord_diff / R_d) > R_tol
                    is_false = true
                end
            end
            
            # Criterion 2: Distance in d+1 is too large relative to standard deviation
            R_dplus1 = sqrt(R_d^2 + coord_diff^2)
            if (R_dplus1 / σ) > A_tol
                is_false = true
            end
            
            if is_false
                false_neighbors_count += 1
            end
        end
        
        fnn_fracs[d] = valid_points_count > 0 ? false_neighbors_count / valid_points_count : 0.0
    end
    
    return fnn_fracs
end

"""
    find_optimal_dimension(series::AbstractVector{<:Real}, delay::Int; max_dim::Int, threshold::Float64) -> Int

Determine the optimal embedding dimension using the False Nearest Neighbors threshold.
"""
function find_optimal_dimension(series::AbstractVector{<:Real}, delay::Int; max_dim::Int=8, threshold::Float64=0.01)
    fnn = false_nearest_neighbors(series, delay, max_dim)
    for d in 1:length(fnn)
        if fnn[d] < threshold
            return d + 1
        end
    end
    # Fallback to the dimension minimizing False Nearest Neighbors
    _, min_idx = findmin(fnn)
    return min_idx + 1
end

# ═══════════════════════════════════════════════════════════════════════════════
# Chaos Metrics (Lyapunov Exponent Curve Fitting)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    lyapunov_divergence_curve(points::Matrix{T}, delay::Int; theiler, max_steps) -> Vector{Float64}

Calculate the average logarithm of divergence of nearest neighbors as a function of step.
Uses a Theiler window to prevent matching temporally adjacent points.
"""
function lyapunov_divergence_curve(points::Matrix{T}, delay::Int; theiler::Int=delay, max_steps::Int=20) where {T<:Real}
    M, dim = size(points)
    if M <= theiler + max_steps
        return Float64[]
    end
    
    divergence = zeros(Float64, max_steps + 1)
    counts = zeros(Int, max_steps + 1)
    
    for i in 1:M
        # Find nearest neighbor of points[i, :] outside theiler window
        min_dist = Inf
        nn_idx = -1
        
        for j in 1:M
            if abs(j - i) <= theiler
                continue
            end
            
            dist_sq = 0.0
            for k in 1:dim
                diff = points[i, k] - points[j, k]
                dist_sq += diff * diff
            end
            
            if dist_sq < min_dist && dist_sq > 1e-12
                min_dist = dist_sq
                nn_idx = j
            end
        end
        
        if nn_idx == -1
            continue
        end
        
        # Track separation over steps
        for l in 0:max_steps
            if i + l <= M && nn_idx + l <= M
                dist_sq_l = 0.0
                for k in 1:dim
                    diff = points[i + l, k] - points[nn_idx + l, k]
                    dist_sq_l += diff * diff
                end
                if dist_sq_l > 1e-15
                    divergence[l+1] += log(sqrt(dist_sq_l))
                    counts[l+1] += 1
                end
            end
        end
    end
    
    for l in 1:(max_steps + 1)
        if counts[l] > 0
            divergence[l] /= counts[l]
        else
            divergence[l] = 0.0
        end
    end
    
    return divergence
end

"""
    estimate_lyapunov_exponent(points::Matrix{T}, delay::Int; theiler, max_steps) -> Float64

Estimate the Maximum Lyapunov Exponent (MLE) by fitting a line to the initial region
of the nearest-neighbor divergence curve. Positive values indicate chaotic dynamics.
"""
function estimate_lyapunov_exponent(points::Matrix{T}, delay::Int; theiler::Int=delay, max_steps::Int=20) where {T<:Real}
    div_curve = lyapunov_divergence_curve(points, delay; theiler=theiler, max_steps=max_steps)
    if isempty(div_curve)
        return 0.0
    end
    
    # Fit line to first half of the curve (linear expansion phase)
    fit_end = max(3, max_steps ÷ 2)
    xs = 0:fit_end
    ys = div_curve[1:(fit_end+1)]
    
    mx = mean(xs)
    my = mean(ys)
    
    num = sum((x - mx) * (y - my) for (x, y) in zip(xs, ys))
    den = sum((x - mx)^2 for x in xs)
    
    if den ≈ 0.0
        return 0.0
    end
    
    return num / den
end

# ═══════════════════════════════════════════════════════════════════════════════
# ColoredTick Multi-channel Embedding
# ═══════════════════════════════════════════════════════════════════════════════

"""
    embed_colored_ticks(ticks::Vector{ColoredTick}, channel::Symbol; base_seed, delay, dim) -> DelayEmbedding{Float64}

Extract a specified telemetry channel from a sequence of `ColoredTick`s and reconstruct
its multi-dimensional phase space embedding.

Valid channels:
- `:confidence`: Confidence levels (0.0 to 1.0)
- `:entropy`: Normalised entropy values
- `:trit`: Measured trit value (-1.0, 0.0, 1.0)
- `:hue`: Hue of the entropy-mixed color (0.0 to 360.0)
- `:saturation`: Saturation of the entropy-mixed color (0.0 to 1.0)
- `:lightness`: Lightness of the entropy-mixed color (0.0 to 1.0)
- `:intrinsic_saturation`: Schrödinger intrinsic saturation of the mixed color (0.0 to 1.0)

If `delay` or `dim` are omitted, they are estimated automatically.
"""
function embed_colored_ticks(ticks::Vector{ColoredTick}, channel::Symbol;
                             base_seed::UInt64=UInt64(1069),
                             delay::Union{Nothing, Int}=nothing,
                             dim::Union{Nothing, Int}=nothing)
    N = length(ticks)
    if N == 0
        throw(ArgumentError("Cannot embed empty vector of ColoredTicks"))
    end
    
    # 1. Extract 1D series
    series = Vector{Float64}(undef, N)
    
    if channel == :confidence
        for i in 1:N
            series[i] = Float64(ticks[i].confidence)
        end
    elseif channel == :entropy
        for i in 1:N
            # Normalise 64-bit unsigned integer to [0, 1]
            series[i] = Float64(ticks[i].entropy) / Float64(typemax(UInt64))
        end
    elseif channel == :trit
        for i in 1:N
            series[i] = Float64(ticks[i].measured_trit)
        end
    elseif channel in (:hue, :saturation, :lightness, :intrinsic_saturation)
        for i in 1:N
            # Compute mixed color identity
            s = entropy_mix(base_seed, ticks[i])
            rgb = color_at(ticks[i].tick; seed=s)
            
            if channel == :hue
                hsl = HSL(rgb)
                series[i] = Float64(hsl.h)
            elseif channel == :saturation
                hsl = HSL(rgb)
                series[i] = Float64(hsl.s)
            elseif channel == :lightness
                hsl = HSL(rgb)
                series[i] = Float64(hsl.l)
            elseif channel == :intrinsic_saturation
                ihsl = to_intrinsic_hsl(rgb; A=25.0)
                series[i] = Float64(ihsl.s)
            end
        end
    else
        throw(ArgumentError("Invalid channel selection: $channel. Choose from :confidence, :entropy, :trit, :hue, :saturation, :lightness, :intrinsic_saturation"))
    end
    
    # 2. Determine delay τ if needed
    τ = if delay === nothing
        find_optimal_delay_ami(series; max_lag=min(50, N ÷ 3))
    else
        delay
    end
    
    # 3. Determine dimension d if needed
    d = if dim === nothing
        find_optimal_dimension(series, τ; max_dim=8, threshold=0.01)
    else
        dim
    end
    
    # 4. Reconstruct phase space
    points = reconstruct_phase_space(series, τ, d)
    
    return DelayEmbedding(points, τ, d, channel)
end

end # module TimeDelayEmbedding
