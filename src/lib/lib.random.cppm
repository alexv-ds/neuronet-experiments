module;
#include <random>
export module lib.random;

namespace lib {

export template<class Rng>
bool prob(Rng& rng, const float prob) {
  std::uniform_real_distribution dist{0.f, 1.f};
  return dist(rng) <=  prob;
}

}