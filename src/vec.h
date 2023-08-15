#pragma once
#include <cstddef>
#include <functional>
#include <vector>

namespace cudla::dense {
struct Vec {
public:
  size_t rows;

  Vec(size_t N);
  Vec(size_t N, const std::vector<float> &vals);
  Vec(size_t N, float val);
  Vec(size_t N, std::function<float(size_t)> pred);

  Vec clone() const;

  ~Vec();

  // Accessors
  float operator[](size_t i) const;
  float &operator()(size_t i);

  Vec operator-(float s) const;
  void operator-=(float s);

  Vec operator-(const Vec &o) const;
  void operator-=(const Vec &o);

  Vec operator+(float s) const;
  void operator+=(float s);

  Vec operator+(const Vec &o) const;
  void operator+=(const Vec &o);

  Vec operator*(float s) const;
  void operator*=(float s);

  void normalize();
  void normalized();

  float dot(const Vec &o) const;

private:
  float *data;
};

} // namespace cudla::dense
