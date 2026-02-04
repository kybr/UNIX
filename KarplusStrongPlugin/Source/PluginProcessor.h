#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_audio_utils/juce_audio_utils.h>
#include <random>
#include "DSP/KarplusStrong.h"
#include "DSP/GuitarBodyReverb.h"

class KarplusStrongVoice : public juce::SynthesiserVoice {
public:
    KarplusStrongVoice();

    bool canPlaySound(juce::SynthesiserSound* sound) override;
    void startNote(int midiNoteNumber, float velocity,
                   juce::SynthesiserSound* sound, int currentPitchWheelPosition) override;
    void stopNote(float velocity, bool allowTailOff) override;
    void pitchWheelMoved(int newPitchWheelValue) override;
    void controllerMoved(int controllerNumber, int newControllerValue) override;
    void renderNextBlock(juce::AudioBuffer<float>& outputBuffer,
                         int startSample, int numSamples) override;

    void setDamping(float damping);
    void setBrightness(float brightness);
    void setADSR(float attack, float decay, float sustain, float release);
    void setVibratoDepth(float depth);

private:
    KarplusStrong synth_;
    juce::ADSR envelope_;
    juce::ADSR::Parameters adsrParams_;
    float damping_ = 0.3f;
    float brightness_ = 0.7f;
    int currentNote_ = -1;
    float pitchBendSemitones_ = 0.0f;
    float baseFrequency_ = 440.0f;

    // Vibrato LFO
    float vibratoPhase_ = 0.0f;
    float vibratoDepth_ = 0.0f;  // In semitones (0-1 range from mod wheel)
    static constexpr float vibratoRate_ = 5.5f;  // Hz - typical guitar vibrato rate
};

class KarplusStrongSound : public juce::SynthesiserSound {
public:
    bool appliesToNote(int) override { return true; }
    bool appliesToChannel(int) override { return true; }
};

class KarplusStrongAudioProcessor : public juce::AudioProcessor,
                                    public juce::MidiKeyboardState::Listener {
public:
    KarplusStrongAudioProcessor();
    ~KarplusStrongAudioProcessor() override;

    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;

    bool isBusesLayoutSupported(const BusesLayout& layouts) const override;

    void processBlock(juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override;

    const juce::String getName() const override;

    bool acceptsMidi() const override;
    bool producesMidi() const override;
    bool isMidiEffect() const override;
    double getTailLengthSeconds() const override;

    int getNumPrograms() override;
    int getCurrentProgram() override;
    void setCurrentProgram(int index) override;
    const juce::String getProgramName(int index) override;
    void changeProgramName(int index, const juce::String& newName) override;

    void getStateInformation(juce::MemoryBlock& destData) override;
    void setStateInformation(const void* data, int sizeInBytes) override;

    juce::AudioProcessorValueTreeState& getAPVTS() { return apvts_; }

    // For waveform display
    void pushSampleToFifo(float sample);
    bool getNextSampleBlock(juce::AudioBuffer<float>& buffer);

    // Manual pluck (for UI button)
    void triggerPluck(int midiNote, float velocity);

    // Pitch bend and mod wheel from UI
    void handlePitchWheel(int value);
    void handleModWheel(int value);

    // Guitar body hit
    void triggerBodyHit();

    // MidiKeyboardState::Listener
    void handleNoteOn(juce::MidiKeyboardState*, int midiChannel, int midiNoteNumber, float velocity) override;
    void handleNoteOff(juce::MidiKeyboardState*, int midiChannel, int midiNoteNumber, float velocity) override;

    // Callback for UI notification when note is played
    std::function<void(int midiNote, float velocity)> onNotePlayed_;

private:
    juce::AudioProcessorValueTreeState apvts_;
    juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout();

    juce::Synthesiser synthesiser_;
    static constexpr int numVoices_ = 16;

    // Waveform display buffer - lock-free snapshot approach
    static constexpr int fifoSize_ = 1024;
    std::array<float, fifoSize_> fifoBuffer_;
    std::atomic<int> fifoWriteIndex_ { 0 };
    std::atomic<bool> fifoReady_ { false };

    // Cached parameters for change detection
    float cachedDamping_ = -1.0f;
    float cachedBrightness_ = -1.0f;
    float cachedAttack_ = -1.0f;
    float cachedDecay_ = -1.0f;
    float cachedSustain_ = -1.0f;
    float cachedRelease_ = -1.0f;

    // Body hit generator
    std::atomic<bool> bodyHitTriggered_ { false };
    float bodyHitEnvelope_ = 0.0f;
    float bodyHitPhase_ = 0.0f;
    float bodyHitFilter_ = 0.0f;
    std::mt19937 bodyHitRng_;

    // Guitar body reverb
    GuitarBodyReverb reverbL_;
    GuitarBodyReverb reverbR_;

    void updateVoiceParameters();
    void processBodyHit(juce::AudioBuffer<float>& buffer);
    void processReverb(juce::AudioBuffer<float>& buffer);

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(KarplusStrongAudioProcessor)
};
