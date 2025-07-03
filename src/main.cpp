#include <xtensor/containers/xarray.hpp>
#include <xtensor/containers/xtensor.hpp>
#include <xtensor/io/xio.hpp>
#include <xtensor/generators/xrandom.hpp>

#include <chrono>
#include <expected>

import mind;
using namespace std::chrono_literals;



int main() {
  constexpr std::size_t neurons = 10*10;
  std::default_random_engine rng{std::random_device{}()};


  MindData mind{
    .tick = 0,
    .activation_thresholds = xt::random::rand<float>({neurons}, 0,1,rng),
    .outputs_weights = [&rng] {
      xt::xtensor<float, 2> arr = xt::random::rand<float>({neurons, neurons}, 0, 1, rng);
      for (std::size_t neuron_idx = 0; neuron_idx < neurons; ++neuron_idx) {
        arr(neuron_idx, neuron_idx) = 0.0f;
      }
      return arr;
    }(),
    .input_weights = [&rng] {
      xt::xtensor<float, 2> arr = xt::random::rand<float>({neurons, neurons}, 0, 1, rng);
      for (std::size_t neuron_idx = 0; neuron_idx < neurons; ++neuron_idx) {
        arr(neuron_idx, neuron_idx) = 0.0f;
      }
      return arr;
    }(),
    .reactivation_delays = xt::random::rand<float>({neurons}, 0,10,rng),
    .next_activations = xt::zeros<float>({neurons}),
    .signal_map = xt::random::rand<float>({neurons}, 0,1,rng),
    .neural_activity = xt::zeros<float>({neurons})
  };

  mind_validate(mind).or_else(
      [](const auto &err) -> std::expected<void, std::string> {
        throw std::invalid_argument{err};
      });

  std::println("NEURONS: {}. LINKS: {}", neurons, mind.outputs_weights.size());


  std::int32_t tick = 0;
  int last_printed_tick = 0;
  std::chrono::steady_clock::time_point next_print_time{};
  while (true) {
    ++tick;
    mind_step(mind);
    if (const auto now = std::chrono::steady_clock::now();
        now >= next_print_time) {
      std::println("TICK: {}, DELTA: {}", tick, tick - last_printed_tick);
      last_printed_tick = tick;
      next_print_time = now + 1s;
    }
  }
}
