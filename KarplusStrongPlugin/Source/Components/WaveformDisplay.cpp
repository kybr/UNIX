#include "WaveformDisplay.h"
#include "../PluginProcessor.h"

WaveformDisplay::WaveformDisplay(KarplusStrongAudioProcessor& processor)
    : processor_(processor) {
    startTimerHz(refreshRate_);
}

WaveformDisplay::~WaveformDisplay() {
    stopTimer();
}

void WaveformDisplay::paint(juce::Graphics& g) {
    auto bounds = getLocalBounds().toFloat();

    // Background
    g.setColour(juce::Colour(0xff16213e));
    g.fillRoundedRectangle(bounds, 8.0f);

    // Border
    g.setColour(juce::Colour(0xff0f3460));
    g.drawRoundedRectangle(bounds.reduced(1.0f), 8.0f, 2.0f);

    // Draw waveform
    if (waveformPath_.isEmpty()) return;

    g.setColour(juce::Colour(0xff00ff88));
    g.strokePath(waveformPath_, juce::PathStrokeType(1.5f));

    // Center line
    g.setColour(juce::Colour(0xff0f3460));
    g.drawHorizontalLine(static_cast<int>(bounds.getCentreY()), bounds.getX() + 10, bounds.getRight() - 10);
}

void WaveformDisplay::resized() {
}

void WaveformDisplay::timerCallback() {
    if (processor_.getNextSampleBlock(waveformBuffer_)) {
        auto bounds = getLocalBounds().toFloat().reduced(10.0f);
        waveformPath_.clear();

        if (waveformBuffer_.getNumSamples() > 0) {
            auto* samples = waveformBuffer_.getReadPointer(0);
            int numSamples = waveformBuffer_.getNumSamples();

            float xScale = bounds.getWidth() / static_cast<float>(numSamples - 1);
            float yScale = bounds.getHeight() * 0.4f;
            float yCenter = bounds.getCentreY();

            waveformPath_.startNewSubPath(bounds.getX(), yCenter - samples[0] * yScale);

            for (int i = 1; i < numSamples; ++i) {
                float x = bounds.getX() + i * xScale;
                float y = yCenter - samples[i] * yScale;
                waveformPath_.lineTo(x, y);
            }
        }

        repaint();
    }
}
