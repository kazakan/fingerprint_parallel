#pragma once

#include "MatrixBuffer.hpp"

namespace fingerprint_parallel {
namespace core {

template <typename T>
class ScalarBuffer : public MatrixBuffer<T> {
   private:
    void set_value(const T& value) { *(this->data()) = value; }

   public:
    ScalarBuffer(T value) : MatrixBuffer<T>(1, 1) { set_value(value); }

    ScalarBuffer() : MatrixBuffer<T>(1, 1) {}

    T& value() { return *(this->data()); }

    void operator=(const T& value) { set_value(value); }
};

}  // namespace core
}  // namespace fingerprint_parallel