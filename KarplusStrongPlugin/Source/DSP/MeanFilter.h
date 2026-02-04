#pragma once

class MeanFilter {
    float z1_ = 0.0f;

public:
    void reset() {
        z1_ = 0.0f;
    }

    float operator()(float input) {
        float output = 0.5f * (input + z1_);
        z1_ = input;
        return output;
    }
};
