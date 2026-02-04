#include "StringVisualization.h"

StringVisualization::StringVisualization() {
    startTimerHz(refreshRate_);
}

StringVisualization::~StringVisualization() {
    stopTimer();
}

void StringVisualization::paint(juce::Graphics& g) {
    auto bounds = getLocalBounds().toFloat();

    // Background
    g.setColour(juce::Colour(0xff16213e));
    g.fillRoundedRectangle(bounds, 8.0f);

    // Border
    g.setColour(juce::Colour(0xff0f3460));
    g.drawRoundedRectangle(bounds.reduced(1.0f), 8.0f, 2.0f);

    // String endpoints
    float padding = 15.0f;
    float stringY = bounds.getCentreY();
    float stringLeft = bounds.getX() + padding;
    float stringRight = bounds.getRight() - padding;
    float stringLength = stringRight - stringLeft;

    // Draw string anchor points
    g.setColour(juce::Colour(0xff0f3460));
    g.fillEllipse(stringLeft - 4, stringY - 4, 8, 8);
    g.fillEllipse(stringRight - 4, stringY - 4, 8, 8);

    // Draw vibrating string
    if (amplitude_ > 0.001f) {
        juce::Path stringPath;
        stringPath.startNewSubPath(stringLeft, stringY);

        int numPoints = 100;
        for (int i = 1; i < numPoints; ++i) {
            float x = stringLeft + (i / static_cast<float>(numPoints)) * stringLength;
            float normalizedX = i / static_cast<float>(numPoints);

            // Sum of harmonics with 1/n^2 amplitude
            float displacement = 0.0f;
            for (int n = 1; n <= numHarmonics_; ++n) {
                float harmonicAmp = amplitude_ / static_cast<float>(n * n);
                float harmonicPhase = phase_ * n;
                displacement += harmonicAmp * std::sin(juce::MathConstants<float>::pi * n * normalizedX)
                              * std::sin(harmonicPhase);
            }

            float y = stringY - displacement * bounds.getHeight() * 0.35f;
            stringPath.lineTo(x, y);
        }

        stringPath.lineTo(stringRight, stringY);

        // Glow effect
        g.setColour(juce::Colour(0xff00ff88).withAlpha(0.3f));
        g.strokePath(stringPath, juce::PathStrokeType(4.0f));

        g.setColour(juce::Colour(0xff00ff88));
        g.strokePath(stringPath, juce::PathStrokeType(2.0f));
    } else {
        // Static string
        g.setColour(juce::Colour(0xff0f3460));
        g.drawLine(stringLeft, stringY, stringRight, stringY, 2.0f);
    }

    // Label
    g.setColour(juce::Colour(0xffe0e0e0).withAlpha(0.5f));
    g.setFont(10.0f);
    g.drawText("STRING", bounds.reduced(5), juce::Justification::topLeft);
}

void StringVisualization::resized() {
}

void StringVisualization::timerCallback() {
    if (amplitude_ > 0.001f) {
        phase_ += frequency_ * 0.001f * juce::MathConstants<float>::twoPi;
        if (phase_ > juce::MathConstants<float>::twoPi * 1000.0f) {
            phase_ = 0.0f;
        }
        amplitude_ *= decayRate_;
        repaint();
    }
}

void StringVisualization::triggerPluck(float frequency, float amplitude) {
    frequency_ = frequency;
    amplitude_ = amplitude;
    phase_ = 0.0f;
}
