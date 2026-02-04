#pragma once

#include <juce_audio_utils/juce_audio_utils.h>

class VirtualKeyboard : public juce::Component {
public:
    VirtualKeyboard(juce::MidiKeyboardState& state);
    ~VirtualKeyboard() override;

    void paint(juce::Graphics& g) override;
    void resized() override;

    // Callbacks for wheel changes and body hit
    std::function<void(int)> onPitchWheelChanged;
    std::function<void(int)> onVibratoChanged;
    std::function<void()> onBodyHit;

private:
    juce::MidiKeyboardComponent keyboard_;

    // Pitch and Vibrato wheels
    juce::Slider pitchWheel_;
    juce::Slider vibratoWheel_;
    juce::Label pitchLabel_;
    juce::Label vibratoLabel_;

    // Body hit button
    juce::TextButton bodyHitButton_;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(VirtualKeyboard)
};
