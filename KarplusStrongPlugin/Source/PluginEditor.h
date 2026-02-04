#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_audio_utils/juce_audio_utils.h>
#include "PluginProcessor.h"
#include "Components/WaveformDisplay.h"
#include "Components/StringVisualization.h"
#include "Components/VirtualKeyboard.h"

class KarplusStrongAudioProcessorEditor : public juce::AudioProcessorEditor {
public:
    explicit KarplusStrongAudioProcessorEditor(KarplusStrongAudioProcessor&);
    ~KarplusStrongAudioProcessorEditor() override;

    void paint(juce::Graphics&) override;
    void resized() override;

private:
    KarplusStrongAudioProcessor& audioProcessor_;

    // Custom look and feel
    class KarplusLookAndFeel : public juce::LookAndFeel_V4 {
    public:
        KarplusLookAndFeel();
        void drawRotarySlider(juce::Graphics& g, int x, int y, int width, int height,
                              float sliderPosProportional, float rotaryStartAngle,
                              float rotaryEndAngle, juce::Slider& slider) override;
        void drawLabel(juce::Graphics& g, juce::Label& label) override;
    };

    KarplusLookAndFeel lookAndFeel_;

    // Visualizations
    StringVisualization stringViz_;
    WaveformDisplay waveformDisplay_;

    // MIDI keyboard
    juce::MidiKeyboardState keyboardState_;
    VirtualKeyboard virtualKeyboard_;

    // Parameter sliders
    juce::Slider dampingSlider_;
    juce::Slider brightnessSlider_;
    juce::Slider attackSlider_;
    juce::Slider decaySlider_;
    juce::Slider sustainSlider_;
    juce::Slider releaseSlider_;
    juce::Slider gainSlider_;

    // Labels
    juce::Label dampingLabel_;
    juce::Label brightnessLabel_;
    juce::Label attackLabel_;
    juce::Label decayLabel_;
    juce::Label sustainLabel_;
    juce::Label releaseLabel_;
    juce::Label gainLabel_;
    juce::Label toneLabel_;
    juce::Label envelopeLabel_;
    juce::Label outputLabel_;

    // Attachments
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> dampingAttachment_;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> brightnessAttachment_;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> attackAttachment_;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> decayAttachment_;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> sustainAttachment_;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> releaseAttachment_;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> gainAttachment_;

    void setupSlider(juce::Slider& slider, juce::Label& label, const juce::String& text);

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(KarplusStrongAudioProcessorEditor)
};
