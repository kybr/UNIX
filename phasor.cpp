#include "everything.h"


// (0, 1)
float sin7(float x) {
    // 7 multiplies + 7 addition/subtraction
    // 14 operations
    return x * (x * (x * (x * (x * (x * (66.5723768716453 * x - 233.003319050759) + 275.754490892928) - 106.877929605423) + 0.156842000875713) - 9.85899292126983) + 7.25653181200263) - 8.88178419700125e-16;
}

// functor class
class Phasor {
  float frequency_; // normalized frequency
  float offset_;
  float phase_;

 public:
  Phasor(float hertz, float sampleRate, float offset = 0)
      : frequency_(hertz / sampleRate), offset_(offset), phase_(0) {}

      // overload the "call" operator
  float operator()() {
    return process();
  }

  void frequency(float hertz, float sampleRate) {
    frequency_ = hertz / sampleRate;
  }

  float process() {
    // wrap
    if (phase_ >= 1.0f) {
      phase_ -= 1.0f;
    }
    float output = phase_ + offset_;
    if (output >= 1.0f) {
      output -= 1.0f;
    }

    phase_ += frequency_; // "side effect" // changes internal state
    return output;
  }
};

class Cycle : Phasor {
 public:
  Cycle(float hertz, float sampleRate)
      : Phasor(hertz, sampleRate, 0.25f) {} // offset by 0.25
  float operator()() {
    return sin7(Phasor::process());
  }
};

int main(int argc, char* argv[]) {
  Phasor carrier(220, SAMPLE_RATE, 0);
  Phasor modulator(5, SAMPLE_RATE, 0);
  Phasor envelope(250, SAMPLE_RATE, 0);
  for (int i = 0; i < SAMPLE_RATE; ++i) {
    carrier.frequency(220 + sin7(modulator()) * 10, SAMPLE_RATE);

    mono(sin7(carrier()) * sin7(envelope()));
  }
}
