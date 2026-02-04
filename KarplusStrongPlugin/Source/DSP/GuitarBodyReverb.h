#pragma once

#include <array>
#include <cmath>

/**
 * Guitar Body Reverb
 *
 * Simulates the resonance of an acoustic guitar body using:
 * - Parallel comb filters tuned to body resonance frequencies
 * - Series allpass filters for diffusion
 * - Carefully tuned decay times for natural sound
 */
class GuitarBodyReverb {
public:
    GuitarBodyReverb() {
        setSampleRate(48000.0f);
        clear();
    }

    void setSampleRate(float sampleRate) {
        sampleRate_ = sampleRate;

        // Guitar body resonance frequencies (typical dreadnought)
        // These create the characteristic "woody" tone
        float bodyFreqs[] = { 95.0f, 189.0f, 285.0f, 392.0f };

        // Calculate comb filter delay times from frequencies
        for (int i = 0; i < numCombs; ++i) {
            combDelayTimes_[i] = static_cast<int>(sampleRate / bodyFreqs[i]);
            // Slightly detune for richness
            if (i % 2 == 1) combDelayTimes_[i] += 3;
        }

        // Allpass delay times for diffusion (prime-ish numbers for density)
        allpassDelayTimes_[0] = static_cast<int>(sampleRate * 0.0051f);  // ~5.1ms
        allpassDelayTimes_[1] = static_cast<int>(sampleRate * 0.0077f);  // ~7.7ms
        allpassDelayTimes_[2] = static_cast<int>(sampleRate * 0.011f);   // ~11ms

        // Set decay times - shorter for guitar body (tight, not washy)
        for (int i = 0; i < numCombs; ++i) {
            // Lower frequencies decay slower (body resonance)
            float decayTime = 0.15f + (i * 0.05f);  // 150-300ms decay
            combFeedback_[i] = std::pow(0.001f, combDelayTimes_[i] / (decayTime * sampleRate));
        }
    }

    void clear() {
        for (auto& comb : combBuffers_) comb.fill(0.0f);
        for (auto& ap : allpassBuffers_) ap.fill(0.0f);
        for (auto& idx : combIndices_) idx = 0;
        for (auto& idx : allpassIndices_) idx = 0;
        for (auto& lp : combLowpass_) lp = 0.0f;
        lowpassState_ = 0.0f;
        prevOutput_ = 0.0f;
    }

    void setMix(float wet) {
        wetMix_ = wet;
        dryMix_ = 1.0f - wet * 0.5f;  // Don't fully cut dry signal
    }

    void setDecay(float decay) {
        // Adjust feedback based on decay parameter (0-1)
        for (int i = 0; i < numCombs; ++i) {
            float baseDecay = 0.1f + decay * 0.25f;  // 100-350ms
            combFeedback_[i] = std::pow(0.001f, combDelayTimes_[i] / (baseDecay * sampleRate_));
        }
    }

    float process(float input) {
        // Pre-filter: gentle lowpass to simulate body absorption
        lowpassState_ = lowpassState_ * 0.7f + input * 0.3f;
        float filtered = lowpassState_;

        // Parallel comb filters (body resonances)
        float combOut = 0.0f;
        for (int i = 0; i < numCombs; ++i) {
            combOut += processComb(i, filtered);
        }
        combOut *= 0.25f;  // Scale down parallel sum

        // Series allpass filters (diffusion)
        float diffused = combOut;
        for (int i = 0; i < numAllpass; ++i) {
            diffused = processAllpass(i, diffused);
        }

        // High-frequency damping on reverb tail
        diffused = diffused * 0.85f + prevOutput_ * 0.15f;
        prevOutput_ = diffused;

        // Mix
        return input * dryMix_ + diffused * wetMix_;
    }

    // Process stereo with slight decorrelation
    void processStereo(float inputL, float inputR, float& outL, float& outR) {
        float mono = (inputL + inputR) * 0.5f;
        float reverb = process(mono);

        // Add subtle stereo width to reverb
        float width = reverb * 0.1f;
        outL = inputL * dryMix_ + (reverb + width) * wetMix_;
        outR = inputR * dryMix_ + (reverb - width) * wetMix_;
    }

private:
    static constexpr int numCombs = 4;
    static constexpr int numAllpass = 3;
    static constexpr int maxDelay = 4096;

    float sampleRate_ = 48000.0f;
    float wetMix_ = 0.3f;
    float dryMix_ = 0.85f;
    float lowpassState_ = 0.0f;
    float prevOutput_ = 0.0f;

    // Comb filters
    std::array<std::array<float, maxDelay>, numCombs> combBuffers_;
    std::array<int, numCombs> combDelayTimes_;
    std::array<int, numCombs> combIndices_;
    std::array<float, numCombs> combFeedback_;
    std::array<float, numCombs> combLowpass_;

    // Allpass filters
    std::array<std::array<float, maxDelay>, numAllpass> allpassBuffers_;
    std::array<int, numAllpass> allpassDelayTimes_;
    std::array<int, numAllpass> allpassIndices_;

    float processComb(int idx, float input) {
        auto& buffer = combBuffers_[idx];
        int delayTime = combDelayTimes_[idx];
        int& writeIdx = combIndices_[idx];

        int readIdx = (writeIdx - delayTime + maxDelay) % maxDelay;
        float delayed = buffer[readIdx];

        // Lowpass in feedback loop (simulates high-freq absorption)
        combLowpass_[idx] = combLowpass_[idx] * 0.6f + delayed * 0.4f;

        float output = combLowpass_[idx];
        buffer[writeIdx] = input + output * combFeedback_[idx];
        writeIdx = (writeIdx + 1) % maxDelay;

        return output;
    }

    float processAllpass(int idx, float input) {
        auto& buffer = allpassBuffers_[idx];
        int delayTime = allpassDelayTimes_[idx];
        int& writeIdx = allpassIndices_[idx];

        int readIdx = (writeIdx - delayTime + maxDelay) % maxDelay;
        float delayed = buffer[readIdx];

        constexpr float g = 0.5f;  // Allpass coefficient
        float output = -g * input + delayed;
        buffer[writeIdx] = input + g * delayed;
        writeIdx = (writeIdx + 1) % maxDelay;

        return output;
    }
};
