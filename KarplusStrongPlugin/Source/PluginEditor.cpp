#include "PluginEditor.h"

// ============================================================================
// Custom LookAndFeel
// ============================================================================

KarplusStrongAudioProcessorEditor::KarplusLookAndFeel::KarplusLookAndFeel() {
    setColour(juce::Slider::rotarySliderFillColourId, juce::Colour(0xff00ff88));
    setColour(juce::Slider::rotarySliderOutlineColourId, juce::Colour(0xff0f3460));
    setColour(juce::Slider::thumbColourId, juce::Colour(0xffe0e0e0));
    setColour(juce::Label::textColourId, juce::Colour(0xffe0e0e0));
    setColour(juce::TextButton::buttonColourId, juce::Colour(0xff0f3460));
    setColour(juce::TextButton::textColourOnId, juce::Colour(0xff00ff88));
    setColour(juce::TextButton::textColourOffId, juce::Colour(0xffe0e0e0));
}

void KarplusStrongAudioProcessorEditor::KarplusLookAndFeel::drawRotarySlider(
    juce::Graphics& g, int x, int y, int width, int height,
    float sliderPosProportional, float rotaryStartAngle, float rotaryEndAngle,
    juce::Slider&) {

    auto bounds = juce::Rectangle<int>(x, y, width, height).toFloat().reduced(5.0f);
    auto radius = juce::jmin(bounds.getWidth(), bounds.getHeight()) / 2.0f;
    auto centreX = bounds.getCentreX();
    auto centreY = bounds.getCentreY();
    auto rx = centreX - radius;
    auto ry = centreY - radius;
    auto rw = radius * 2.0f;
    auto angle = rotaryStartAngle + sliderPosProportional * (rotaryEndAngle - rotaryStartAngle);

    // Background arc
    g.setColour(juce::Colour(0xff0f3460));
    juce::Path backgroundArc;
    backgroundArc.addCentredArc(centreX, centreY, radius, radius, 0.0f,
                                 rotaryStartAngle, rotaryEndAngle, true);
    g.strokePath(backgroundArc, juce::PathStrokeType(4.0f, juce::PathStrokeType::curved,
                                                      juce::PathStrokeType::rounded));

    // Value arc
    g.setColour(juce::Colour(0xff00ff88));
    juce::Path valueArc;
    valueArc.addCentredArc(centreX, centreY, radius, radius, 0.0f,
                            rotaryStartAngle, angle, true);
    g.strokePath(valueArc, juce::PathStrokeType(4.0f, juce::PathStrokeType::curved,
                                                 juce::PathStrokeType::rounded));

    // Knob center
    g.setColour(juce::Colour(0xff16213e));
    g.fillEllipse(rx + radius * 0.3f, ry + radius * 0.3f, rw * 0.7f, rw * 0.7f);

    // Pointer
    juce::Path pointer;
    auto pointerLength = radius * 0.6f;
    auto pointerThickness = 3.0f;
    pointer.addRectangle(-pointerThickness * 0.5f, -radius * 0.8f, pointerThickness, pointerLength);
    pointer.applyTransform(juce::AffineTransform::rotation(angle).translated(centreX, centreY));

    g.setColour(juce::Colour(0xffe0e0e0));
    g.fillPath(pointer);
}

void KarplusStrongAudioProcessorEditor::KarplusLookAndFeel::drawLabel(juce::Graphics& g, juce::Label& label) {
    g.setColour(label.findColour(juce::Label::textColourId));
    g.setFont(label.getFont());

    auto textArea = label.getBorderSize().subtractedFrom(label.getLocalBounds());
    g.drawFittedText(label.getText(), textArea, label.getJustificationType(),
                     juce::jmax(1, static_cast<int>(textArea.getHeight() / label.getFont().getHeight())),
                     label.getMinimumHorizontalScale());
}

// ============================================================================
// Editor Implementation
// ============================================================================

KarplusStrongAudioProcessorEditor::KarplusStrongAudioProcessorEditor(KarplusStrongAudioProcessor& p)
    : AudioProcessorEditor(&p)
    , audioProcessor_(p)
    , stringViz_()
    , waveformDisplay_(p)
    , virtualKeyboard_(keyboardState_) {

    setLookAndFeel(&lookAndFeel_);

    // Set up sliders
    setupSlider(dampingSlider_, dampingLabel_, "Damping");
    setupSlider(brightnessSlider_, brightnessLabel_, "Brightness");
    setupSlider(attackSlider_, attackLabel_, "A");
    setupSlider(decaySlider_, decayLabel_, "D");
    setupSlider(sustainSlider_, sustainLabel_, "S");
    setupSlider(releaseSlider_, releaseLabel_, "R");
    setupSlider(gainSlider_, gainLabel_, "Gain");

    // Section labels
    auto setupSectionLabel = [this](juce::Label& label, const juce::String& text) {
        label.setText(text, juce::dontSendNotification);
        label.setFont(juce::Font(12.0f, juce::Font::bold));
        label.setColour(juce::Label::textColourId, juce::Colour(0xff00ff88));
        label.setJustificationType(juce::Justification::centred);
        addAndMakeVisible(label);
    };

    setupSectionLabel(toneLabel_, "TONE");
    setupSectionLabel(envelopeLabel_, "ENVELOPE");
    setupSectionLabel(outputLabel_, "OUTPUT");

    // Attachments
    dampingAttachment_ = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        audioProcessor_.getAPVTS(), "damping", dampingSlider_);
    brightnessAttachment_ = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        audioProcessor_.getAPVTS(), "brightness", brightnessSlider_);
    attackAttachment_ = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        audioProcessor_.getAPVTS(), "attack", attackSlider_);
    decayAttachment_ = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        audioProcessor_.getAPVTS(), "decay", decaySlider_);
    sustainAttachment_ = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        audioProcessor_.getAPVTS(), "sustain", sustainSlider_);
    releaseAttachment_ = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        audioProcessor_.getAPVTS(), "release", releaseSlider_);
    gainAttachment_ = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        audioProcessor_.getAPVTS(), "gain", gainSlider_);

    // Visualizations
    addAndMakeVisible(stringViz_);
    addAndMakeVisible(waveformDisplay_);

    // Keyboard
    addAndMakeVisible(virtualKeyboard_);

    // Connect keyboard state to processor
    keyboardState_.addListener(&audioProcessor_);

    // Set up callback for string visualization when notes are played
    audioProcessor_.onNotePlayed_ = [this](int midiNote, float velocity) {
        float freq = 440.0f * std::pow(2.0f, (midiNote - 69) / 12.0f);
        stringViz_.triggerPluck(freq, velocity);
    };

    // Wire up pitch wheel, vibrato, and body hit callbacks
    virtualKeyboard_.onPitchWheelChanged = [this](int value) {
        audioProcessor_.handlePitchWheel(value);
    };
    virtualKeyboard_.onVibratoChanged = [this](int value) {
        audioProcessor_.handleModWheel(value);
    };
    virtualKeyboard_.onBodyHit = [this]() {
        audioProcessor_.triggerBodyHit();
    };

    // Add tooltips to sliders
    dampingSlider_.setTooltip("Controls string decay rate (0 = long sustain, 1 = quick decay)");
    brightnessSlider_.setTooltip("Controls string brightness (0 = mellow, 1 = bright)");
    attackSlider_.setTooltip("Envelope attack time");
    decaySlider_.setTooltip("Envelope decay time");
    sustainSlider_.setTooltip("Envelope sustain level");
    releaseSlider_.setTooltip("Envelope release time");
    gainSlider_.setTooltip("Output gain in dB");

    // Make window resizable
    setResizable(true, true);
    setResizeLimits(600, 400, 1200, 800);
    setSize(800, 550);
}

KarplusStrongAudioProcessorEditor::~KarplusStrongAudioProcessorEditor() {
    keyboardState_.removeListener(&audioProcessor_);
    setLookAndFeel(nullptr);
}

void KarplusStrongAudioProcessorEditor::setupSlider(juce::Slider& slider, juce::Label& label,
                                                     const juce::String& text) {
    slider.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
    slider.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 50, 15);
    slider.setColour(juce::Slider::textBoxTextColourId, juce::Colour(0xffe0e0e0));
    slider.setColour(juce::Slider::textBoxBackgroundColourId, juce::Colours::transparentBlack);
    slider.setColour(juce::Slider::textBoxOutlineColourId, juce::Colours::transparentBlack);
    addAndMakeVisible(slider);

    label.setText(text, juce::dontSendNotification);
    label.setFont(juce::Font(11.0f));
    label.setColour(juce::Label::textColourId, juce::Colour(0xffe0e0e0));
    label.setJustificationType(juce::Justification::centred);
    addAndMakeVisible(label);
}

void KarplusStrongAudioProcessorEditor::paint(juce::Graphics& g) {
    // Background gradient
    juce::ColourGradient gradient(juce::Colour(0xff1a1a2e), 0, 0,
                                   juce::Colour(0xff16213e), 0, static_cast<float>(getHeight()),
                                   false);
    g.setGradientFill(gradient);
    g.fillAll();

    // Title bar - proportional height
    int titleHeight = juce::jmax(40, proportionOfHeight(0.07f));
    g.setColour(juce::Colour(0xff0f3460));
    g.fillRect(0, 0, getWidth(), titleHeight);

    g.setColour(juce::Colour(0xffe0e0e0));
    g.setFont(juce::Font(juce::jmax(14.0f, titleHeight * 0.45f), juce::Font::bold));
    g.drawText("KARPLUS-STRONG SYNTHESIZER", 15, 0, getWidth() - 30, titleHeight,
               juce::Justification::centredLeft);

    // Accent line
    g.setColour(juce::Colour(0xff00ff88));
    g.fillRect(0, titleHeight, getWidth(), 2);

    // Section dividers - proportional positions
    int vizAndControlsTop = titleHeight + 2 + proportionOfHeight(0.22f) + proportionOfHeight(0.02f) + 10;
    int controlsHeight = proportionOfHeight(0.38f);
    int controlsBottom = vizAndControlsTop + controlsHeight;

    int totalWidth = getWidth();
    int toneWidth = totalWidth * 28 / 100;
    int envWidth = totalWidth * 45 / 100;
    int divider1X = toneWidth;
    int divider2X = toneWidth + envWidth;

    g.setColour(juce::Colour(0xff0f3460));
    g.drawLine(static_cast<float>(divider1X), static_cast<float>(vizAndControlsTop + 20),
               static_cast<float>(divider1X), static_cast<float>(controlsBottom - 10), 1.0f);
    g.drawLine(static_cast<float>(divider2X), static_cast<float>(vizAndControlsTop + 20),
               static_cast<float>(divider2X), static_cast<float>(controlsBottom - 10), 1.0f);
}

void KarplusStrongAudioProcessorEditor::resized() {
    auto bounds = getLocalBounds();

    // Title area: 7% of height
    auto titleHeight = proportionOfHeight(0.07f);
    bounds.removeFromTop(juce::jmax(42, titleHeight));

    // Visualization area: 22% of height, EQUAL 50/50 split
    auto vizHeight = proportionOfHeight(0.22f);
    auto vizArea = bounds.removeFromTop(juce::jmax(100, vizHeight)).reduced(10);
    auto halfWidth = vizArea.getWidth() / 2;
    stringViz_.setBounds(vizArea.removeFromLeft(halfWidth).reduced(5));
    waveformDisplay_.setBounds(vizArea.reduced(5));

    // Spacing
    bounds.removeFromTop(proportionOfHeight(0.02f));

    // Controls area: 38% of height
    auto controlsHeight = proportionOfHeight(0.38f);
    auto controlsArea = bounds.removeFromTop(juce::jmax(130, controlsHeight));

    // Calculate proportional section widths
    int totalWidth = controlsArea.getWidth();
    int toneWidth = totalWidth * 28 / 100;      // 28%
    int envWidth = totalWidth * 45 / 100;       // 45%
    // Output section uses remaining width after tone and envelope

    // TONE section (left)
    auto toneArea = controlsArea.removeFromLeft(toneWidth);
    toneLabel_.setBounds(toneArea.removeFromTop(20));
    auto toneKnobs = toneArea.reduced(10, 0);

    int toneKnobWidth = toneKnobs.getWidth() / 2;
    auto dampingArea = toneKnobs.removeFromLeft(toneKnobWidth);
    dampingSlider_.setBounds(dampingArea.removeFromTop(dampingArea.getHeight() - 20));
    dampingLabel_.setBounds(dampingArea);

    auto brightnessArea = toneKnobs;
    brightnessSlider_.setBounds(brightnessArea.removeFromTop(brightnessArea.getHeight() - 20));
    brightnessLabel_.setBounds(brightnessArea);

    // ENVELOPE section (middle)
    auto envArea = controlsArea.removeFromLeft(envWidth);
    envelopeLabel_.setBounds(envArea.removeFromTop(20));
    auto envKnobs = envArea.reduced(10, 0);

    int envKnobWidth = envKnobs.getWidth() / 4;
    auto aArea = envKnobs.removeFromLeft(envKnobWidth);
    attackSlider_.setBounds(aArea.removeFromTop(aArea.getHeight() - 20));
    attackLabel_.setBounds(aArea);

    auto dArea = envKnobs.removeFromLeft(envKnobWidth);
    decaySlider_.setBounds(dArea.removeFromTop(dArea.getHeight() - 20));
    decayLabel_.setBounds(dArea);

    auto sArea = envKnobs.removeFromLeft(envKnobWidth);
    sustainSlider_.setBounds(sArea.removeFromTop(sArea.getHeight() - 20));
    sustainLabel_.setBounds(sArea);

    auto rArea = envKnobs;
    releaseSlider_.setBounds(rArea.removeFromTop(rArea.getHeight() - 20));
    releaseLabel_.setBounds(rArea);

    // OUTPUT section (right)
    auto outputArea = controlsArea;
    outputLabel_.setBounds(outputArea.removeFromTop(20));
    auto outputKnobs = outputArea.reduced(30, 0);

    gainSlider_.setBounds(outputKnobs.removeFromTop(outputKnobs.getHeight() - 20));
    gainLabel_.setBounds(outputKnobs);

    // Virtual keyboard: remaining space - fill it completely
    virtualKeyboard_.setBounds(bounds);
}
