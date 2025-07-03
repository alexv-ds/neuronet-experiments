module;
#include <format>
#include <stdexcept>
#include <xtensor/containers/xtensor.hpp>
#include <xtensor/views/xview.hpp>
#include <expected>
export module mind;

export struct MindData {
  std::int32_t tick{};
  xt::xtensor<float, 1> activation_thresholds;
  xt::xtensor<float, 2> outputs_weights;
  xt::xtensor<float, 2> input_weights;
  xt::xtensor<float, 1> reactivation_delays;
  xt::xtensor<float, 1> next_activations;

  xt::xtensor<float, 1> signal_map;
  xt::xtensor<float, 1> neural_activity;
};

export std::expected<void, std::string> mind_validate(const MindData &mind) {
  const std::size_t size_t_count = mind.neural_activity.size();
  if (mind.activation_thresholds.size() != size_t_count) {
    return std::unexpected{
        std::format("activation_thresholds.size() != {}", size_t_count)};
  }
  if (mind.reactivation_delays.size() != size_t_count) {
    return std::unexpected{
        std::format("reactivation_delays.size() != {}", size_t_count)};
  }
  if (mind.next_activations.size() != size_t_count) {
    return std::unexpected{
        std::format("next_activations.size() != {}", size_t_count)};
  }
  if (mind.signal_map.size() != size_t_count) {
    return std::unexpected{
        std::format("signal_map.size() != {}", size_t_count)};
  }
  if (mind.outputs_weights.shape()[0] != size_t_count) {
    return std::unexpected{
        std::format("outputs_weights.shape()[0] != {}", size_t_count)};
  }
  if (mind.outputs_weights.shape()[1] != size_t_count) {
    return std::unexpected{
        std::format("outputs_weights.shape()[1] != {}", size_t_count)};
  }
  if (mind.input_weights.shape()[0] != size_t_count) {
    return std::unexpected{
        std::format("input_weights.shape()[0] != {}", size_t_count)};
  }
  if (mind.input_weights.shape()[1] != size_t_count) {
    return std::unexpected{
        std::format("input_weights.shape()[1] != {}", size_t_count)};
  }
  return {};
}

export void mind_step(MindData &mind) {
  const auto tick = static_cast<float>(mind.tick);
  const std::ptrdiff_t neurons_count = mind.neural_activity.size();

  // расчет сигналов пришедших в нейрон
  for (std::ptrdiff_t i = 0; i < neurons_count; ++i) {
    if (mind.next_activations(i) <= tick) {
      mind.neural_activity(i) =
          xt::sum(xt::row(mind.input_weights, i) * mind.signal_map)();
    } else {
      mind.neural_activity(i) = 0.0f;
    }
  }

  // применение функции активации
  for (std::ptrdiff_t i = 0; i < neurons_count; ++i) {
    if (mind.neural_activity(i) > mind.activation_thresholds(i)) {
      mind.next_activations(i) = tick + mind.reactivation_delays(i);
    } else {
      mind.neural_activity(i) = 0.0f;
    }
  }

  // передаем сигналы дальше, рассчитываем состояния нейронов через
  // outputs_weights и записываем в signal_map
  for (std::ptrdiff_t i = 0; i < neurons_count; ++i) {
    mind.signal_map(i) =
        xt::sum(xt::row(mind.outputs_weights, i) * mind.neural_activity)();
  }
}