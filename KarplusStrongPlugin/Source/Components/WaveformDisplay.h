#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_gui_basics/juce_gui_basics.h>

class KarplusStrongAudioProcessor;

class WaveformDisplay : public juce::Component, public juce::Timer {
public:
    WaveformDisplay(KarplusStrongAudioProcessor& processor);
    ~WaveformDisplay() override;

    void paint(juce::Graphics& g) override;
    void resized() override;
    void timerCallback() override;

private:
    KarplusStrongAudioProcessor& processor_;
    juce::AudioBuffer<float> waveformBuffer_;
    juce::Path waveformPath_;

    static constexpr int refreshRate_ = 30;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(WaveformDisplay)
};
