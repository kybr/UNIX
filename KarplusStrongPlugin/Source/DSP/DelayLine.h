#pragma once

#include <vector>
#include <algorithm>

class DelayLine {
    std::vector<float> buffer_;
    int writePos_;
    int size_;

public:
    DelayLine(int maxSize = 4096)
        : buffer_(maxSize, 0.0f), writePos_(0), size_(maxSize) {}

    void resize(int size) {
        if (size > static_cast<int>(buffer_.size())) {
            buffer_.resize(size, 0.0f);
        }
        size_ = size;
        if (writePos_ >= size_) {
            writePos_ = 0;
        }
    }

    void clear() {
        std::fill(buffer_.begin(), buffer_.end(), 0.0f);
        writePos_ = 0;
    }

    void write(float sample) {
        buffer_[writePos_] = sample;
        ++writePos_;
        if (writePos_ >= size_) {
            writePos_ = 0;
        }
    }

    float read(int delaySamples) const {
        int readPos = writePos_ - delaySamples - 1;
        while (readPos < 0) {
            readPos += size_;
        }
        return buffer_[readPos];
    }

    float readLinear(float delaySamples) const {
        int intDelay = static_cast<int>(delaySamples);
        float frac = delaySamples - intDelay;

        float s0 = read(intDelay);
        float s1 = read(intDelay + 1);

        return s0 + frac * (s1 - s0);
    }

    float operator()(float input) {
        float oldest = buffer_[writePos_];
        write(input);
        return oldest;
    }

    int size() const { return size_; }
};
