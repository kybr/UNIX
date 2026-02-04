#include "VirtualKeyboard.h"

VirtualKeyboard::VirtualKeyboard(juce::MidiKeyboardState& state)
    : keyboard_(state, juce::MidiKeyboardComponent::horizontalKeyboard) {

    // Set range to C2-C5 (3 octaves)
    keyboard_.setAvailableRange(36, 72);
    keyboard_.setKeyWidth(28.0f);

    // Custom colors for dark theme
    keyboard_.setColour(juce::MidiKeyboardComponent::whiteNoteColourId, juce::Colour(0xffe0e0e0));
    keyboard_.setColour(juce::MidiKeyboardComponent::blackNoteColourId, juce::Colour(0xff1a1a2e));
    keyboard_.setColour(juce::MidiKeyboardComponent::keySeparatorLineColourId, juce::Colour(0xff0f3460));
    keyboard_.setColour(juce::MidiKeyboardComponent::mouseOverKeyOverlayColourId, juce::Colour(0xff00ff88).withAlpha(0.3f));
    keyboard_.setColour(juce::MidiKeyboardComponent::keyDownOverlayColourId, juce::Colour(0xff00ff88).withAlpha(0.6f));

    addAndMakeVisible(keyboard_);

    // Pitch wheel setup - vertical slider, centered at 8192
    pitchWheel_.setSliderStyle(juce::Slider::LinearVertical);
    pitchWheel_.setRange(0, 16383, 1);
    pitchWheel_.setValue(8192);  // Center position
    pitchWheel_.setTextBoxStyle(juce::Slider::NoTextBox, false, 0, 0);
    pitchWheel_.setColour(juce::Slider::thumbColourId, juce::Colour(0xff00ff88));
    pitchWheel_.setColour(juce::Slider::trackColourId, juce::Colour(0xff0f3460));
    pitchWheel_.setColour(juce::Slider::backgroundColourId, juce::Colour(0xff1a1a2e));
    pitchWheel_.onValueChange = [this]() {
        if (onPitchWheelChanged)
            onPitchWheelChanged(static_cast<int>(pitchWheel_.getValue()));
    };
    // Spring back to center on mouse release
    pitchWheel_.onDragEnd = [this]() {
        pitchWheel_.setValue(8192);
    };
    addAndMakeVisible(pitchWheel_);

    // Vibrato wheel setup - vertical slider, 0-127
    vibratoWheel_.setSliderStyle(juce::Slider::LinearVertical);
    vibratoWheel_.setRange(0, 127, 1);
    vibratoWheel_.setValue(0);
    vibratoWheel_.setTextBoxStyle(juce::Slider::NoTextBox, false, 0, 0);
    vibratoWheel_.setColour(juce::Slider::thumbColourId, juce::Colour(0xff00ff88));
    vibratoWheel_.setColour(juce::Slider::trackColourId, juce::Colour(0xff0f3460));
    vibratoWheel_.setColour(juce::Slider::backgroundColourId, juce::Colour(0xff1a1a2e));
    vibratoWheel_.onValueChange = [this]() {
        if (onVibratoChanged)
            onVibratoChanged(static_cast<int>(vibratoWheel_.getValue()));
    };
    addAndMakeVisible(vibratoWheel_);

    // Labels
    pitchLabel_.setText("BEND", juce::dontSendNotification);
    pitchLabel_.setFont(juce::Font(9.0f));
    pitchLabel_.setColour(juce::Label::textColourId, juce::Colour(0xffe0e0e0));
    pitchLabel_.setJustificationType(juce::Justification::centred);
    addAndMakeVisible(pitchLabel_);

    vibratoLabel_.setText("VIB", juce::dontSendNotification);
    vibratoLabel_.setFont(juce::Font(9.0f));
    vibratoLabel_.setColour(juce::Label::textColourId, juce::Colour(0xffe0e0e0));
    vibratoLabel_.setJustificationType(juce::Justification::centred);
    addAndMakeVisible(vibratoLabel_);

    // Body hit button
    bodyHitButton_.setButtonText("BODY");
    bodyHitButton_.setColour(juce::TextButton::buttonColourId, juce::Colour(0xff0f3460));
    bodyHitButton_.setColour(juce::TextButton::buttonOnColourId, juce::Colour(0xff00ff88));
    bodyHitButton_.setColour(juce::TextButton::textColourOffId, juce::Colour(0xffe0e0e0));
    bodyHitButton_.setColour(juce::TextButton::textColourOnId, juce::Colour(0xff1a1a2e));
    bodyHitButton_.onClick = [this]() {
        if (onBodyHit)
            onBodyHit();
    };
    addAndMakeVisible(bodyHitButton_);
}

VirtualKeyboard::~VirtualKeyboard() {
}

void VirtualKeyboard::paint(juce::Graphics& g) {
    auto bounds = getLocalBounds().toFloat();

    // Background
    g.setColour(juce::Colour(0xff16213e));
    g.fillRoundedRectangle(bounds, 8.0f);

    // Border
    g.setColour(juce::Colour(0xff0f3460));
    g.drawRoundedRectangle(bounds.reduced(1.0f), 8.0f, 2.0f);
}

void VirtualKeyboard::resized() {
    auto bounds = getLocalBounds().reduced(8);

    // Controls on the left side
    int wheelWidth = 30;
    int wheelSpacing = 4;
    int labelHeight = 14;
    int buttonHeight = 35;
    int buttonSpacing = 5;

    // Calculate total width needed for left controls
    int controlsWidth = wheelWidth * 2 + wheelSpacing + 15;
    auto controlsArea = bounds.removeFromLeft(controlsWidth);

    // Top padding
    controlsArea.removeFromTop(3);

    // Body hit button at top
    auto buttonArea = controlsArea.removeFromTop(buttonHeight);
    buttonArea = buttonArea.reduced(2, 0);
    bodyHitButton_.setBounds(buttonArea);

    controlsArea.removeFromTop(buttonSpacing);

    // Labels at bottom
    auto labelArea = controlsArea.removeFromBottom(labelHeight);

    // Pitch wheel (left)
    auto pitchArea = controlsArea.removeFromLeft(wheelWidth);
    pitchWheel_.setBounds(pitchArea);
    pitchLabel_.setBounds(labelArea.removeFromLeft(wheelWidth));

    controlsArea.removeFromLeft(wheelSpacing);
    labelArea.removeFromLeft(wheelSpacing);

    // Vibrato wheel (right of pitch)
    auto vibArea = controlsArea.removeFromLeft(wheelWidth);
    vibratoWheel_.setBounds(vibArea);
    vibratoLabel_.setBounds(labelArea.removeFromLeft(wheelWidth));

    // Keyboard takes the rest
    bounds.removeFromLeft(3);  // Small gap between controls and keyboard
    keyboard_.setBounds(bounds);
}
