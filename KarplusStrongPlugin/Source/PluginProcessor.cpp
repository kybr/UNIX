#include "PluginProcessor.h"
#include "PluginEditor.h"

// ============================================================================
// KarplusStrongVoice Implementation
// ============================================================================

KarplusStrongVoice::KarplusStrongVoice() {
    adsrParams_.attack = 0.001f;
    adsrParams_.decay = 0.1f;
    adsrParams_.sustain = 0.8f;
    adsrParams_.release = 2.0f;
    envelope_.setParameters(adsrParams_);
}

bool KarplusStrongVoice::canPlaySound(juce::SynthesiserSound* sound) {
    return dynamic_cast<KarplusStrongSound*>(sound) != nullptr;
}

void KarplusStrongVoice::startNote(int midiNoteNumber, float velocity,
                                    juce::SynthesiserSound*, int currentPitchWheelPosition) {
    currentNote_ = midiNoteNumber;
    baseFrequency_ = 440.0f * std::pow(2.0f, (midiNoteNumber - 69) / 12.0f);

    // Apply pitch bend
    pitchBendSemitones_ = ((currentPitchWheelPosition - 8192) / 8192.0f) * 2.0f;
    float freq = baseFrequency_ * std::pow(2.0f, pitchBendSemitones_ / 12.0f);

    synth_.setSampleRate(static_cast<float>(getSampleRate()));
    synth_.setDamping(damping_);
    synth_.setBrightness(brightness_);
    synth_.frequency(freq);
    synth_.pluck(velocity);

    envelope_.setSampleRate(getSampleRate());
    envelope_.setParameters(adsrParams_);
    envelope_.noteOn();
}

void KarplusStrongVoice::stopNote(float, bool allowTailOff) {
    if (allowTailOff) {
        envelope_.noteOff();
    } else {
        envelope_.reset();
        synth_.stop();
        clearCurrentNote();
    }
}

void KarplusStrongVoice::pitchWheelMoved(int newPitchWheelValue) {
    // Just update pitch bend semitones - the actual pitch modulation
    // is applied in renderNextBlock via setPitchModulation()
    // This avoids double-application of pitch bend
    pitchBendSemitones_ = ((newPitchWheelValue - 8192) / 8192.0f) * 2.0f;
}

void KarplusStrongVoice::controllerMoved(int, int) {
    // Could handle mod wheel, etc.
}

void KarplusStrongVoice::renderNextBlock(juce::AudioBuffer<float>& outputBuffer,
                                          int startSample, int numSamples) {
    if (!synth_.isActive() && !envelope_.isActive()) {
        clearCurrentNote();
        return;
    }

    float sampleRate = static_cast<float>(getSampleRate());
    float vibratoIncrement = vibratoRate_ / sampleRate;

    for (int i = startSample; i < startSample + numSamples; ++i) {
        // Calculate total pitch modulation (pitch bend + vibrato)
        float totalPitchMod = pitchBendSemitones_;

        if (vibratoDepth_ > 0.0f) {
            float vibratoMod = std::sin(vibratoPhase_ * 2.0f * juce::MathConstants<float>::pi);
            totalPitchMod += vibratoMod * vibratoDepth_;
            vibratoPhase_ += vibratoIncrement;
            if (vibratoPhase_ >= 1.0f)
                vibratoPhase_ -= 1.0f;
        }

        // Apply pitch modulation smoothly via delay line adjustment
        synth_.setPitchModulation(totalPitchMod);

        float sample = synth_() * envelope_.getNextSample();

        for (int channel = 0; channel < outputBuffer.getNumChannels(); ++channel) {
            outputBuffer.addSample(channel, i, sample);
        }

        if (!envelope_.isActive()) {
            synth_.stop();
            clearCurrentNote();
            break;
        }
    }
}

void KarplusStrongVoice::setDamping(float damping) {
    damping_ = damping;
    synth_.setDamping(damping);
}

void KarplusStrongVoice::setBrightness(float brightness) {
    brightness_ = brightness;
    synth_.setBrightness(brightness);
}

void KarplusStrongVoice::setADSR(float attack, float decay, float sustain, float release) {
    adsrParams_.attack = attack;
    adsrParams_.decay = decay;
    adsrParams_.sustain = sustain;
    adsrParams_.release = release;
    envelope_.setParameters(adsrParams_);
}

void KarplusStrongVoice::setVibratoDepth(float depth) {
    // depth 0-1 maps to 0-1 semitone vibrato depth
    vibratoDepth_ = depth;
}

// ============================================================================
// KarplusStrongAudioProcessor Implementation
// ============================================================================

KarplusStrongAudioProcessor::KarplusStrongAudioProcessor()
    : AudioProcessor(BusesProperties()
                     .withOutput("Output", juce::AudioChannelSet::stereo(), true)),
      apvts_(*this, nullptr, "Parameters", createParameterLayout()) {

    for (int i = 0; i < numVoices_; ++i) {
        synthesiser_.addVoice(new KarplusStrongVoice());
    }
    synthesiser_.addSound(new KarplusStrongSound());
}

KarplusStrongAudioProcessor::~KarplusStrongAudioProcessor() {
}

juce::AudioProcessorValueTreeState::ParameterLayout
KarplusStrongAudioProcessor::createParameterLayout() {
    std::vector<std::unique_ptr<juce::RangedAudioParameter>> params;

    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID("damping", 1), "Damping",
        juce::NormalisableRange<float>(0.0f, 1.0f, 0.01f), 0.3f));

    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID("brightness", 1), "Brightness",
        juce::NormalisableRange<float>(0.0f, 1.0f, 0.01f), 0.7f));

    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID("attack", 1), "Attack",
        juce::NormalisableRange<float>(0.001f, 1.0f, 0.001f, 0.4f), 0.001f));

    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID("decay", 1), "Decay",
        juce::NormalisableRange<float>(0.01f, 5.0f, 0.01f, 0.4f), 0.1f));

    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID("sustain", 1), "Sustain",
        juce::NormalisableRange<float>(0.0f, 1.0f, 0.01f), 0.8f));

    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID("release", 1), "Release",
        juce::NormalisableRange<float>(0.01f, 10.0f, 0.01f, 0.4f), 2.0f));

    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID("gain", 1), "Gain",
        juce::NormalisableRange<float>(-60.0f, 12.0f, 0.1f), 0.0f));

    return { params.begin(), params.end() };
}

const juce::String KarplusStrongAudioProcessor::getName() const {
    return JucePlugin_Name;
}

bool KarplusStrongAudioProcessor::acceptsMidi() const { return true; }
bool KarplusStrongAudioProcessor::producesMidi() const { return false; }
bool KarplusStrongAudioProcessor::isMidiEffect() const { return false; }
double KarplusStrongAudioProcessor::getTailLengthSeconds() const { return 10.0; }
int KarplusStrongAudioProcessor::getNumPrograms() { return 1; }
int KarplusStrongAudioProcessor::getCurrentProgram() { return 0; }
void KarplusStrongAudioProcessor::setCurrentProgram(int) {}
const juce::String KarplusStrongAudioProcessor::getProgramName(int) { return {}; }
void KarplusStrongAudioProcessor::changeProgramName(int, const juce::String&) {}

void KarplusStrongAudioProcessor::prepareToPlay(double sampleRate, int) {
    synthesiser_.setCurrentPlaybackSampleRate(sampleRate);

    for (int i = 0; i < synthesiser_.getNumVoices(); ++i) {
        if (auto* voice = dynamic_cast<KarplusStrongVoice*>(synthesiser_.getVoice(i))) {
            voice->setADSR(
                *apvts_.getRawParameterValue("attack"),
                *apvts_.getRawParameterValue("decay"),
                *apvts_.getRawParameterValue("sustain"),
                *apvts_.getRawParameterValue("release")
            );
        }
    }

    // Set up guitar body reverb
    reverbL_.setSampleRate(static_cast<float>(sampleRate));
    reverbR_.setSampleRate(static_cast<float>(sampleRate));
    reverbL_.setMix(0.25f);  // Subtle reverb
    reverbR_.setMix(0.25f);
    reverbL_.setDecay(0.4f);  // Medium body resonance
    reverbR_.setDecay(0.4f);
}

void KarplusStrongAudioProcessor::releaseResources() {
}

bool KarplusStrongAudioProcessor::isBusesLayoutSupported(const BusesLayout& layouts) const {
    if (layouts.getMainOutputChannelSet() != juce::AudioChannelSet::mono()
        && layouts.getMainOutputChannelSet() != juce::AudioChannelSet::stereo())
        return false;

    return true;
}

void KarplusStrongAudioProcessor::processBlock(juce::AudioBuffer<float>& buffer,
                                                juce::MidiBuffer& midiMessages) {
    juce::ScopedNoDenormals noDenormals;

    buffer.clear();

    updateVoiceParameters();

    synthesiser_.renderNextBlock(buffer, midiMessages, 0, buffer.getNumSamples());

    // Process body hit percussion
    processBodyHit(buffer);

    // Apply guitar body reverb (simulates acoustic resonance)
    processReverb(buffer);

    // Apply output gain
    float gainDb = *apvts_.getRawParameterValue("gain");
    float gain = std::pow(10.0f, gainDb / 20.0f);
    buffer.applyGain(gain);

    // Push samples to FIFO for waveform display
    auto* channelData = buffer.getReadPointer(0);
    for (int i = 0; i < buffer.getNumSamples(); ++i) {
        pushSampleToFifo(channelData[i]);
    }
}

void KarplusStrongAudioProcessor::updateVoiceParameters() {
    float damping = *apvts_.getRawParameterValue("damping");
    float brightness = *apvts_.getRawParameterValue("brightness");
    float attack = *apvts_.getRawParameterValue("attack");
    float decay = *apvts_.getRawParameterValue("decay");
    float sustain = *apvts_.getRawParameterValue("sustain");
    float release = *apvts_.getRawParameterValue("release");

    // Only update voices if parameters have changed
    bool dampingChanged = damping != cachedDamping_;
    bool brightnessChanged = brightness != cachedBrightness_;
    bool envelopeChanged = attack != cachedAttack_ || decay != cachedDecay_ ||
                           sustain != cachedSustain_ || release != cachedRelease_;

    if (!dampingChanged && !brightnessChanged && !envelopeChanged)
        return;

    // Update cache
    cachedDamping_ = damping;
    cachedBrightness_ = brightness;
    cachedAttack_ = attack;
    cachedDecay_ = decay;
    cachedSustain_ = sustain;
    cachedRelease_ = release;

    for (int i = 0; i < synthesiser_.getNumVoices(); ++i) {
        if (auto* voice = dynamic_cast<KarplusStrongVoice*>(synthesiser_.getVoice(i))) {
            if (dampingChanged)
                voice->setDamping(damping);
            if (brightnessChanged)
                voice->setBrightness(brightness);
            if (envelopeChanged)
                voice->setADSR(attack, decay, sustain, release);
        }
    }
}

void KarplusStrongAudioProcessor::pushSampleToFifo(float sample) {
    int idx = fifoWriteIndex_.load(std::memory_order_relaxed);
    fifoBuffer_[static_cast<size_t>(idx)] = sample;
    int nextIdx = (idx + 1) % fifoSize_;
    fifoWriteIndex_.store(nextIdx, std::memory_order_relaxed);

    // Signal ready when buffer wraps (we have a full buffer)
    if (nextIdx == 0)
        fifoReady_.store(true, std::memory_order_release);
}

bool KarplusStrongAudioProcessor::getNextSampleBlock(juce::AudioBuffer<float>& buffer) {
    if (!fifoReady_.load(std::memory_order_acquire))
        return false;

    buffer.setSize(1, fifoSize_, false, false, true);

    // Copy the buffer (slight visual tearing is acceptable for waveform display)
    for (int i = 0; i < fifoSize_; ++i)
        buffer.setSample(0, i, fifoBuffer_[static_cast<size_t>(i)]);

    fifoReady_.store(false, std::memory_order_release);
    return true;
}

void KarplusStrongAudioProcessor::triggerPluck(int midiNote, float velocity) {
    synthesiser_.noteOn(1, midiNote, velocity);
}

void KarplusStrongAudioProcessor::handlePitchWheel(int value) {
    // Send pitch wheel to all channels (the synth responds to all)
    for (int i = 0; i < synthesiser_.getNumVoices(); ++i) {
        if (auto* voice = dynamic_cast<KarplusStrongVoice*>(synthesiser_.getVoice(i))) {
            voice->pitchWheelMoved(value);
        }
    }
}

void KarplusStrongAudioProcessor::handleModWheel(int value) {
    // Mod wheel controls vibrato depth (0-1 semitone)
    float vibratoDepth = value / 127.0f;

    for (int i = 0; i < synthesiser_.getNumVoices(); ++i) {
        if (auto* voice = dynamic_cast<KarplusStrongVoice*>(synthesiser_.getVoice(i))) {
            voice->setVibratoDepth(vibratoDepth);
        }
    }
}

void KarplusStrongAudioProcessor::triggerBodyHit() {
    bodyHitTriggered_.store(true);
}

void KarplusStrongAudioProcessor::processBodyHit(juce::AudioBuffer<float>& buffer) {
    // Check if triggered
    if (bodyHitTriggered_.exchange(false)) {
        bodyHitEnvelope_ = 1.0f;
        bodyHitPhase_ = 0.0f;
    }

    if (bodyHitEnvelope_ < 0.0001f)
        return;

    float sampleRate = static_cast<float>(getSampleRate());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Body resonance frequencies (typical guitar body)
    float bodyFreq1 = 95.0f;   // Low body resonance
    float bodyFreq2 = 200.0f;  // Mid resonance
    float twoPi = 2.0f * juce::MathConstants<float>::pi;

    for (int i = 0; i < buffer.getNumSamples(); ++i) {
        // Generate noise burst with body resonance
        float noise = dist(bodyHitRng_);

        // Simple one-pole lowpass filter for thump character
        bodyHitFilter_ = bodyHitFilter_ * 0.85f + noise * 0.15f;

        // Add body resonances using the phase
        float body1 = std::sin(bodyHitPhase_ * twoPi * bodyFreq1);
        float body2 = std::sin(bodyHitPhase_ * twoPi * bodyFreq2) * 0.5f;

        float sample = (bodyHitFilter_ * 0.6f + body1 * 0.3f + body2 * 0.1f) * bodyHitEnvelope_;

        // Apply to all channels
        for (int channel = 0; channel < buffer.getNumChannels(); ++channel) {
            buffer.addSample(channel, i, sample * 0.4f);  // Mix level
        }

        // Fast exponential decay for percussive sound
        bodyHitEnvelope_ *= 0.9985f;
        bodyHitPhase_ += 1.0f / sampleRate;
        if (bodyHitPhase_ >= 1.0f)
            bodyHitPhase_ -= 1.0f;
    }
}

void KarplusStrongAudioProcessor::processReverb(juce::AudioBuffer<float>& buffer) {
    int numChannels = buffer.getNumChannels();
    int numSamples = buffer.getNumSamples();

    if (numChannels >= 2) {
        // Stereo processing
        auto* leftChannel = buffer.getWritePointer(0);
        auto* rightChannel = buffer.getWritePointer(1);

        for (int i = 0; i < numSamples; ++i) {
            float inL = leftChannel[i];
            float inR = rightChannel[i];
            float outL, outR;

            reverbL_.processStereo(inL, inR, outL, outR);
            // Use slightly different processing for R channel for width
            leftChannel[i] = outL;
            rightChannel[i] = reverbR_.process(inR) * 0.3f + outR * 0.7f;
        }
    } else if (numChannels == 1) {
        // Mono processing
        auto* channel = buffer.getWritePointer(0);
        for (int i = 0; i < numSamples; ++i) {
            channel[i] = reverbL_.process(channel[i]);
        }
    }
}

void KarplusStrongAudioProcessor::handleNoteOn(juce::MidiKeyboardState*, int midiChannel,
                                                int midiNoteNumber, float velocity) {
    synthesiser_.noteOn(midiChannel, midiNoteNumber, velocity);

    // Notify UI for visualization
    if (onNotePlayed_)
        onNotePlayed_(midiNoteNumber, velocity);
}

void KarplusStrongAudioProcessor::handleNoteOff(juce::MidiKeyboardState*, int midiChannel,
                                                 int midiNoteNumber, float) {
    synthesiser_.noteOff(midiChannel, midiNoteNumber, 0.0f, true);
}

bool KarplusStrongAudioProcessor::hasEditor() const { return true; }

juce::AudioProcessorEditor* KarplusStrongAudioProcessor::createEditor() {
    return new KarplusStrongAudioProcessorEditor(*this);
}

void KarplusStrongAudioProcessor::getStateInformation(juce::MemoryBlock& destData) {
    auto state = apvts_.copyState();
    std::unique_ptr<juce::XmlElement> xml(state.createXml());
    copyXmlToBinary(*xml, destData);
}

void KarplusStrongAudioProcessor::setStateInformation(const void* data, int sizeInBytes) {
    std::unique_ptr<juce::XmlElement> xmlState(getXmlFromBinary(data, sizeInBytes));
    if (xmlState != nullptr && xmlState->hasTagName(apvts_.state.getType())) {
        apvts_.replaceState(juce::ValueTree::fromXml(*xmlState));
    }
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter() {
    return new KarplusStrongAudioProcessor();
}
