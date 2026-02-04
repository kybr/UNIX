#pragma once

#include "DelayLine.h"
#include "MeanFilter.h"
#include <cmath>
#include <random>

/**
 * Velocity Curve Types
 *
 * Different curves for mapping MIDI velocity to amplitude:
 * - Linear: Direct 1:1 mapping (velocity/127)
 * - Soft: Square root curve, gentler response for expressive playing
 * - Hard: Squared curve, aggressive response for punchy playing
 */
enum class VelocityCurve {
    Linear,
    Soft,
    Hard
};

/**
 * Karplus-Strong String Synthesis
 *
 * Physical modeling synthesis using a delay line with lowpass feedback.
 * The delay line length determines pitch, and the mean filter creates
 * the characteristic plucked string decay.
 *
 * Improvements in this version:
 * - Accurate pitch calculation accounting for filter group delay
 * - Velocity curve control for expressive playing
 * - Smooth pitch modulation for vibrato/pitch bend
 * - Brightness control for initial excitation filtering
 */
class KarplusStrong {
    DelayLine delay_;
    MeanFilter filter_;
    float sampleRate_;
    float frequency_;
    float baseDelayLength_;  // Base delay for current note
    int delayLength_;
    float fractionalDelay_;
    float damping_;
    float brightness_;
    bool active_;
    float pitchModulation_;  // Current pitch modulation in semitones
    VelocityCurve velocityCurve_;

    std::mt19937 rng_;
    std::uniform_real_distribution<float> dist_;

    float randomNoise(float amplitude) {
        return amplitude * dist_(rng_);
    }

    /**
     * Apply velocity curve to convert MIDI velocity to amplitude.
     */
    float applyVelocityCurve(float velocity) const {
        switch (velocityCurve_) {
            case VelocityCurve::Soft:
                // Square root curve: gentler response, easier to play quietly
                return std::sqrt(velocity);
            case VelocityCurve::Hard:
                // Squared curve: more aggressive, punchy response
                return velocity * velocity;
            case VelocityCurve::Linear:
            default:
                return velocity;
        }
    }

public:
    KarplusStrong(float sampleRate = 48000.0f)
        : delay_(4096)
        , sampleRate_(sampleRate)
        , frequency_(440.0f)
        , baseDelayLength_(100.0f)
        , delayLength_(100)
        , fractionalDelay_(0.0f)
        , damping_(0.3f)
        , brightness_(0.7f)
        , active_(false)
        , pitchModulation_(0.0f)
        , velocityCurve_(VelocityCurve::Linear)
        , rng_(std::random_device{}())
        , dist_(-1.0f, 1.0f) {
        frequency(440.0f);
    }

    void setSampleRate(float sampleRate) {
        sampleRate_ = sampleRate;
        frequency(frequency_);
    }

    /**
     * Set the fundamental frequency.
     *
     * The delay length is calculated accounting for the group delay of the
     * mean filter (~0.5 samples) to improve pitch accuracy.
     */
    void frequency(float hertz) {
        frequency_ = hertz;
        // Account for mean filter group delay (approximately 0.5 samples)
        // This improves pitch accuracy, especially for higher notes
        constexpr float filterGroupDelay = 0.5f;
        baseDelayLength_ = sampleRate_ / hertz - filterGroupDelay;
        updateDelayFromModulation();
    }

    /**
     * Smooth pitch modulation without recalculating base delay.
     * Used for vibrato and pitch bend during a note.
     *
     * @param semitones Pitch offset in semitones (positive = higher pitch)
     */
    void setPitchModulation(float semitones) {
        pitchModulation_ = semitones;
        updateDelayFromModulation();
    }

    void setDamping(float d) {
        damping_ = d;
    }

    void setBrightness(float b) {
        brightness_ = b;
    }

    /**
     * Set velocity curve type.
     */
    void setVelocityCurve(VelocityCurve curve) {
        velocityCurve_ = curve;
    }

    VelocityCurve getVelocityCurve() const {
        return velocityCurve_;
    }

    /**
     * Trigger a new note (pluck the string).
     *
     * Fills the delay line with filtered noise to excite the resonator.
     *
     * @param amplitude Pluck amplitude (typically from MIDI velocity 0-1)
     */
    void pluck(float amplitude = 1.0f) {
        // Apply velocity curve
        float scaledAmplitude = applyVelocityCurve(amplitude);

        int maxDelay = static_cast<int>(baseDelayLength_) + 10;
        if (maxDelay < 20) maxDelay = 20;
        delay_.resize(maxDelay);
        delay_.clear();
        filter_.reset();
        pitchModulation_ = 0.0f;

        // Fill with filtered noise based on brightness
        // Brightness controls high-frequency content of initial excitation
        float prevSample = 0.0f;
        int fillLength = static_cast<int>(baseDelayLength_) + 1;
        for (int i = 0; i < fillLength; ++i) {
            float noise = randomNoise(scaledAmplitude);
            // Low-pass filter the noise: brightness 1.0 = no filter, 0.0 = heavy filter
            float filtered = brightness_ * noise + (1.0f - brightness_) * prevSample;
            delay_.write(filtered);
            prevSample = filtered;
        }
        active_ = true;
    }

    /**
     * Generate next output sample.
     */
    float operator()() {
        if (!active_) return 0.0f;

        float totalDelay = delayLength_ + fractionalDelay_;
        float delayed = delay_.readLinear(totalDelay);
        float filtered = filter_(delayed);

        // Apply damping: scales output and determines decay rate
        // damping_ range 0-1, scaled to musical range (0-5% attenuation per sample)
        filtered *= (1.0f - damping_ * 0.05f);

        delay_.write(filtered);

        // Detect when string has decayed to silence
        if (std::abs(filtered) < 1e-7f) {
            active_ = false;
        }

        return filtered;
    }

    bool isActive() const {
        return active_;
    }

    float getFrequency() const {
        return frequency_;
    }

    void stop() {
        active_ = false;
        delay_.clear();
        filter_.reset();
    }

private:
    void updateDelayFromModulation() {
        // Convert semitone modulation to delay length change
        // Pitch up = shorter delay, pitch down = longer delay
        float pitchRatio = std::pow(2.0f, -pitchModulation_ / 12.0f);
        float modulatedDelay = baseDelayLength_ * pitchRatio;

        delayLength_ = static_cast<int>(modulatedDelay);
        fractionalDelay_ = modulatedDelay - delayLength_;

        // Clamp to minimum delay to prevent instability
        if (delayLength_ < 2) {
            delayLength_ = 2;
            fractionalDelay_ = 0.0f;
        }
    }
};
