#pragma once

#include <juce_gui_basics/juce_gui_basics.h>

class StringVisualization : public juce::Component, public juce::Timer {
public:
    StringVisualization();
    ~StringVisualization() override;

    void paint(juce::Graphics& g) override;
    void resized() override;
    void timerCallback() override;

    void triggerPluck(float frequency, float amplitude);

private:
    float frequency_ = 220.0f;
    float amplitude_ = 0.0f;
    float phase_ = 0.0f;
    float decayRate_ = 0.98f;

    static constexpr int numHarmonics_ = 5;
    static constexpr int refreshRate_ = 60;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(StringVisualization)
};
