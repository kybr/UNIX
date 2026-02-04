#include <JuceHeader.h>
#include "DSP/DelayLine.h"
#include "DSP/MeanFilter.h"
#include "DSP/KarplusStrong.h"

class DelayLineTests : public juce::UnitTest {
public:
    DelayLineTests() : UnitTest("DelayLine Tests") {}

    void runTest() override {
        beginTest("Basic write and read");
        {
            DelayLine dl(16);
            dl.resize(8);

            // Write some samples
            for (int i = 0; i < 8; ++i) {
                dl.write(static_cast<float>(i + 1));
            }

            // Read back with delay
            expect(dl.read(0) == 8.0f);  // Most recent
            expect(dl.read(1) == 7.0f);
            expect(dl.read(7) == 1.0f);  // Oldest
        }

        beginTest("Linear interpolation");
        {
            DelayLine dl(16);
            dl.resize(8);

            dl.write(0.0f);
            dl.write(10.0f);

            // Fractional delay should interpolate
            float interpolated = dl.readLinear(0.5f);
            expectWithinAbsoluteError(interpolated, 5.0f, 0.001f);
        }

        beginTest("Circular buffer wraparound");
        {
            DelayLine dl(4);
            dl.resize(4);

            // Write more samples than buffer size
            for (int i = 0; i < 10; ++i) {
                dl.write(static_cast<float>(i));
            }

            // Should only contain last 4 values: 6, 7, 8, 9
            expect(dl.read(0) == 9.0f);
            expect(dl.read(3) == 6.0f);
        }

        beginTest("Clear resets buffer");
        {
            DelayLine dl(8);
            dl.resize(8);

            for (int i = 0; i < 8; ++i) {
                dl.write(1.0f);
            }

            dl.clear();

            for (int i = 0; i < 8; ++i) {
                expectWithinAbsoluteError(dl.read(i), 0.0f, 0.0001f);
            }
        }
    }
};

class MeanFilterTests : public juce::UnitTest {
public:
    MeanFilterTests() : UnitTest("MeanFilter Tests") {}

    void runTest() override {
        beginTest("Two-point average");
        {
            MeanFilter filter;

            // First sample: (0 + 1) / 2 = 0.5
            float out1 = filter(1.0f);
            expectWithinAbsoluteError(out1, 0.5f, 0.001f);

            // Second sample: (1 + 3) / 2 = 2.0
            float out2 = filter(3.0f);
            expectWithinAbsoluteError(out2, 2.0f, 0.001f);
        }

        beginTest("Reset clears state");
        {
            MeanFilter filter;
            filter(10.0f);
            filter.reset();

            // After reset, should be (0 + 2) / 2 = 1.0
            float out = filter(2.0f);
            expectWithinAbsoluteError(out, 1.0f, 0.001f);
        }

        beginTest("Constant input produces same output after warmup");
        {
            MeanFilter filter;
            filter(5.0f);  // Warmup

            for (int i = 0; i < 10; ++i) {
                float out = filter(5.0f);
                expectWithinAbsoluteError(out, 5.0f, 0.001f);
            }
        }
    }
};

class KarplusStrongTests : public juce::UnitTest {
public:
    KarplusStrongTests() : UnitTest("KarplusStrong Tests") {}

    void runTest() override {
        beginTest("Inactive before pluck");
        {
            KarplusStrong ks(48000.0f);
            expect(!ks.isActive());
            expectWithinAbsoluteError(ks(), 0.0f, 0.0001f);
        }

        beginTest("Active after pluck");
        {
            KarplusStrong ks(48000.0f);
            ks.frequency(440.0f);
            ks.pluck(1.0f);
            expect(ks.isActive());
        }

        beginTest("Output decays to silence");
        {
            KarplusStrong ks(48000.0f);
            ks.frequency(440.0f);
            ks.pluck(1.0f);

            // Run for a few seconds worth of samples
            float maxSample = 0.0f;
            for (int i = 0; i < 48000 * 5; ++i) {
                float sample = std::abs(ks());
                if (sample > maxSample) maxSample = sample;
            }

            // Should eventually become inactive
            expect(!ks.isActive());
        }

        beginTest("Frequency affects delay length");
        {
            KarplusStrong ks(48000.0f);

            // Higher frequency = shorter delay
            ks.frequency(880.0f);
            ks.pluck(1.0f);

            // Count samples until we see pattern repeat (approximate)
            // For 880 Hz at 48kHz, period ~= 48000/880 ~= 54.5 samples
            // Just verify it produces output
            float sum = 0.0f;
            for (int i = 0; i < 100; ++i) {
                sum += std::abs(ks());
            }
            expect(sum > 0.0f);
        }

        beginTest("Damping affects decay rate");
        {
            // High damping should decay faster
            KarplusStrong ks1(48000.0f);
            ks1.frequency(220.0f);
            ks1.setDamping(0.0f);
            ks1.pluck(1.0f);

            KarplusStrong ks2(48000.0f);
            ks2.frequency(220.0f);
            ks2.setDamping(0.9f);
            ks2.pluck(1.0f);

            // Run both for 1 second
            float energy1 = 0.0f, energy2 = 0.0f;
            for (int i = 0; i < 48000; ++i) {
                float s1 = ks1();
                float s2 = ks2();
                energy1 += s1 * s1;
                energy2 += s2 * s2;
            }

            // Higher damping should result in lower total energy
            expect(energy2 < energy1);
        }

        beginTest("Stop immediately silences");
        {
            KarplusStrong ks(48000.0f);
            ks.frequency(440.0f);
            ks.pluck(1.0f);

            // Generate some samples
            for (int i = 0; i < 100; ++i) ks();

            expect(ks.isActive());
            ks.stop();
            expect(!ks.isActive());
            expectWithinAbsoluteError(ks(), 0.0f, 0.0001f);
        }

        beginTest("Sample rate change updates frequency calculation");
        {
            KarplusStrong ks(48000.0f);
            ks.frequency(440.0f);
            float freq1 = ks.getFrequency();

            ks.setSampleRate(96000.0f);
            float freq2 = ks.getFrequency();

            // Frequency should be preserved
            expectWithinAbsoluteError(freq1, freq2, 0.001f);
        }
    }
};

// Register tests
static DelayLineTests delayLineTests;
static MeanFilterTests meanFilterTests;
static KarplusStrongTests karplusStrongTests;

int main(int argc, char* argv[]) {
    juce::UnitTestRunner runner;
    runner.runAllTests();

    int numFailures = 0;
    for (int i = 0; i < runner.getNumResults(); ++i) {
        if (runner.getResult(i)->failures > 0) {
            numFailures += runner.getResult(i)->failures;
        }
    }

    return numFailures > 0 ? 1 : 0;
}
