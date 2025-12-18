module SeedSonification

const CHROMATIC_FREQS = [261.63, 277.18, 293.66, 311.13, 329.63, 349.23, 
                         369.99, 392.00, 415.30, 440.00, 466.16, 493.88]  # C4-B4

hue_to_frequency(hue::Float64) = CHROMATIC_FREQS[1 + mod(floor(Int, hue / 30), 12)]

polarity_to_waveform(p::UInt8) = p == 0 ? :sine : p == 1 ? :triangle : :sawtooth

xor_to_rhythm(fp::UInt64) = [((fp >> i) & 1) == 1 for i in 0:7]

struct SeedSonificationData
    seed::UInt64
    frequency::Float64
    waveform::Symbol
    rhythm::Vector{Bool}
end

function sm64(s::UInt64)
    z = s + 0x9e3779b97f4a7c15
    z = (z ⊻ (z >> 30)) * 0xbf58476d1ce4e5b9
    z = (z ⊻ (z >> 27)) * 0x94d049bb133111eb
    z ⊻ (z >> 31)
end

function sonify_seed(seed::UInt64)
    hash = sm64(seed)
    hue = (hash % 360) * 1.0
    polarity = UInt8((hash >> 8) % 3)
    frequency = hue_to_frequency(hue)
    waveform = polarity_to_waveform(polarity)
    rhythm = xor_to_rhythm(hash)
    SeedSonificationData(seed, frequency, waveform, rhythm)
end

function world_seed_sonification()
    seeds = [UInt64(1069), UInt64(5980), UInt64(6939)]
    results = [sonify_seed(s) for s in seeds]
    for r in results
        println("Seed $(r.seed): $(r.frequency)Hz $(r.waveform) rhythm=$(r.rhythm)")
    end
    results
end

export hue_to_frequency, polarity_to_waveform, xor_to_rhythm
export SeedSonificationData, sm64, sonify_seed, world_seed_sonification

end
