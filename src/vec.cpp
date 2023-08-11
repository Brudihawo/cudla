
// struct Vec {
//   float *data;
//   size_t cols;

//   Vec(size_t N);
//   Vec(size_t N, std::vector<float> &vals);
//   Vec(size_t N, std::function<float(size_t)> pred);

//   Vec operator-(float s) const;
//   Vec operator-=(float s);

//   Vec operator-(const Vec &o) const;
//   Vec operator-=(const Vec &o);

//   Vec operator+(float s) const;
//   Vec operator+=(float s);

//   Vec operator+(const Vec &o) const;
//   Vec operator+=(const Vec &o);

//   Vec operator*(const cudla::dense::Mat &o);
//   Vec operator*(const cudla::sparse::Mat &o);

//   float dot(const Vec& o) const;
// };

#include "vec.h"
#include "errors.h"
#include <cstddef>
#include <cstring>

namespace cudla::dense {
Vec::Vec(size_t N) : rows(N), data(new float[N]) {}

Vec::Vec(size_t N, float val) : rows(N), data(new float[N]) {
  std::fill_n(this->data, N, val);
}

Vec::Vec(size_t N, const std::vector<float> &vals)
    : rows(N), data(new float[N]) {
  std::copy(vals.begin(), vals.end(), data);
}

Vec::Vec(size_t N, std::function<float(size_t)> pred)
    : rows(N), data(new float[N]) {
  for (size_t i = 0; i < N; ++i) {
    data[i] = pred(i);
  }
}

Vec Vec::clone() const {
  Vec ret(this->rows);
  std::copy(&this->data[0], &this->data[ret.rows], ret.data);
  return ret;
}

float Vec::operator[](size_t i) const { return data[i]; }
float &Vec::operator()(size_t i) { return data[i]; }

Vec Vec::operator-(float s) const {
  Vec ret = this->clone();
  for (size_t i = 0; i < ret.rows; ++i) {
    ret.data[i] -= s;
  }

  return ret;
}

void Vec::operator-=(float s) {
  for (size_t i = 0; i < this->rows; ++i) {
    this->data[i] -= s;
  }
}

Vec Vec::operator-(const Vec &o) const {
  cudla_assert_msg(o.rows == this->rows,
                   "Vector addition can only be performed between vectors of "
                   "the same dimensions");

  Vec ret = this->clone();
  for (size_t i = 0; i < ret.rows; ++i) {
    ret.data[i] -= o.data[i];
  }
  return ret;
}

void Vec::operator-=(const Vec &o) {
  cudla_assert_msg(o.rows == this->rows,
                   "Vector addition can only be performed between vectors of "
                   "the same dimensions");

  for (size_t i = 0; i < this->rows; ++i) {
    this->data[i] -= o.data[i];
  }
}

Vec Vec::operator+(float s) const {
  Vec ret = this->clone();
  for (size_t i = 0; i < ret.rows; ++i) {
    ret.data[i] += s;
  }
  return ret;
}

void Vec::operator+=(float s) {
  for (size_t i = 0; i < this->rows; ++i) {
    this->data[i] -= s;
  }
}

Vec Vec::operator+(const Vec &o) const {
  cudla_assert_msg(o.rows == this->rows,
                   "Vector addition can only be performed between vectors of "
                   "the same dimensions");

  Vec ret = this->clone();
  for (size_t i = 0; i < ret.rows; ++i) {
    ret.data[i] += o.data[i];
  }
  return ret;
}

void Vec::operator+=(const Vec &o) {
  cudla_assert_msg(o.rows == this->rows,
                   "Vector addition can only be performed between vectors of "
                   "the same dimensions");

  for (size_t i = 0; i < this->rows; ++i) {
    this->data[i] += o.data[i];
  }
}

Vec Vec::operator*(float s) const {
  Vec ret = this->clone();
  for (size_t i = 0; i < ret.rows; ++i) {
    ret.data[i] *= s;
  }
  return ret;
}

void Vec::operator*=(float s) {
  for (size_t i = 0; i < this->rows; ++i) {
    this->data[i] *= s;
  }
}

Vec::~Vec() { delete[] this->data; }

} // namespace cudla::dense
