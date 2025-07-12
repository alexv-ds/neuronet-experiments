module;
#include <expected>
#include <format>
#include <nlohmann/json.hpp>
#include <xtensor/containers/xtensor.hpp>
#include <xtensor/io/xnpy.hpp>
#include <xtensor/views/xview.hpp>
export module mind;

namespace mind {

/**
 * @brief Represents the complete internal state of a recurrent neural simulation step.
 *
 * This structure contains all the necessary components to simulate a fully connected,
 * time-dependent neural network where each neuron can:
 * - receive input signals,
 * - apply an activation function based on thresholds,
 * - propagate signals to other neurons,
 * - respect a refractory delay before the next activation.
 *
 * All tensors are expected to have consistent dimensions (`N` neurons), and are
 * updated in-place during each `mind_step()` call.
 */
export struct MindData {
  /**
   * @brief The current simulation tick (time step).
   *
   * Used as a global reference to control activation timing and refractory delays.
   */
  std::int32_t tick{};

  /**
   * @brief Per-neuron activation thresholds.
   *
   * If a neuron's computed activity exceeds this threshold, it becomes active
   * and enters a refractory period.
   *
   * Dimensions: (N)
   */
  xt::xtensor<float, 1> activation_thresholds;

  /**
   * @brief Output weight matrix.
   *
   * Defines how signals from active neurons are propagated to others.
   * Each row `i` specifies how neuron `i` influences the rest.
   *
   * Dimensions: (N, N)
   */
  xt::xtensor<float, 2> outputs_weights;

  /**
   * @brief Input weight matrix.
   *
   * Defines how incoming signals are aggregated for each neuron from the current signal map.
   * Each row `i` determines how much influence the global signal map has on neuron `i`.
   *
   * Dimensions: (N, N)
   */
  xt::xtensor<float, 2> input_weights;

  /**
   * @brief Refractory delay per neuron (in ticks).
   *
   * After activation, a neuron must wait this many ticks before it can activate again.
   *
   * Dimensions: (N)
   */
  xt::xtensor<float, 1> reactivation_delays;

  /**
   * @brief The earliest tick at which each neuron can activate again.
   *
   * Updated after activation based on the current tick and the neuron's delay.
   *
   * Dimensions: (N)
   */
  xt::xtensor<float, 1> next_activations;

  /**
   * @brief Current input signals to each neuron.
   *
   * Calculated during the signal propagation phase from other neurons.
   * Used as input for computing neural activity.
   *
   * Dimensions: (N)
   */
  xt::xtensor<float, 1> signal_map;

  /**
   * @brief Computed activity level of each neuron during the current tick.
   *
   * Updated based on inputs and input weights, thresholded, and then used for output signal computation.
   * Set to zero if the neuron is inactive or still in its refractory period.
   *
   * Dimensions: (N)
   */
  xt::xtensor<float, 1> neural_activity;
};

/**
 * @brief Validates the structural consistency of a MindData instance.
 *
 * This function performs a series of size checks to ensure that all internal fields
 * of the given `MindData` object are dimensionally consistent.
 *
 * Specifically:
 * - All 1D tensors (vectors) must have the same length `N`.
 * - All 2D tensors (matrices) must have shape `N × N`.
 *
 * If any mismatch is found, an error message is returned describing the inconsistency.
 *
 * @param mind A reference to the `MindData` object to validate.
 * @return
 * - `std::expected<void, std::string>{}` if validation succeeds (all sizes match),
 * - `std::unexpected<std::string>` with an error message if any check fails.
 *
 * @see MindData
 */
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

/**
 * @brief Executes a single update step of the neural network represented by MindData.
 *
 * This function simulates one time tick (`mind.tick`) of the neural activity by:
 * - Incrementing the global tick counter and resetting it when reaching a maximum (`max_tick = 1024`).
 *   When the tick resets, all `next_activations` values are decreased by the tick amount to maintain correct timing.
 * - Computing incoming signals to each neuron based on input weights and the current `signal_map`.
 * - Applying an activation function that compares neuron activity against thresholds.
 *   When a neuron activates, its next activation time is delayed by the maximum of its reactivation delay and `max_tick`,
 *   ensuring a minimal delay threshold.
 * - Propagating outputs through output weights to update the `signal_map`.
 *
 * The process includes three main phases:
 * 1. **Tick management**: Increment the tick, reset to zero upon reaching `max_tick`, adjusting `next_activations` accordingly.
 * 2. **Signal accumulation**: Neurons ready to activate (`next_activations <= tick`) sum weighted inputs from the signal map.
 * 3. **Activation decision**: Neurons exceeding their activation threshold update `next_activations` with a delay ensuring at least `max_tick`.
 * 4. **Signal propagation**: Active neurons propagate their signals using output weights to update the signal map.
 *
 * This function modifies `neural_activity`, `next_activations`, and `signal_map` in-place.
 *
 * @param mind A reference to the `MindData` structure containing the full network state.
 *
 * @see mind_validate
 * @see MindData
 */
export void mind_step(MindData &mind) {
  ++mind.tick;
  constexpr std::int32_t max_tick = 1024;

  const auto neurons_count =
      static_cast<std::ptrdiff_t>(mind.neural_activity.size());

  // calc ticks
  if (mind.tick >= max_tick) {
    const auto tick = static_cast<float>(mind.tick);
    for (std::ptrdiff_t i = 0; i < neurons_count; ++i) {
      mind.next_activations(i) -= tick;
    }
    mind.tick = 0;
  }

  const auto tick = static_cast<float>(mind.tick);

  // calculate signals received by each neuron
  for (std::ptrdiff_t i = 0; i < neurons_count; ++i) {
    if (mind.next_activations(i) <= tick) {
      mind.neural_activity(i) =
          xt::sum(xt::row(mind.input_weights, i) * mind.signal_map)();
    } else {
      mind.neural_activity(i) = 0.0f;
    }
  }

  // apply activation function
  for (std::ptrdiff_t i = 0; i < neurons_count; ++i) {
    if (mind.neural_activity(i) > mind.activation_thresholds(i)) {
      mind.next_activations(i) = tick + std::max(mind.reactivation_delays(i),
                                                 static_cast<float>(max_tick));
    } else {
      mind.neural_activity(i) = 0.0f;
    }
  }


  // propagate signals using outputs_weights and store results in signal_map
  for (std::ptrdiff_t i = 0; i < neurons_count; ++i) {
    mind.signal_map(i) =
        xt::sum(xt::row(mind.outputs_weights, i) * mind.neural_activity)();
  }
}

export std::vector<std::uint8_t>
mind_serialize(const MindData &mind) {
  auto _ = mind_validate(mind).or_else(
      [](const auto &err) -> std::expected<void, std::string> {
        throw std::invalid_argument{err};
      });

  const std::vector<std::uint8_t> data = [&mind] {
    using json = nlohmann::json;
    json j;
    j["version"] = 1;
    j["tick"] = mind.tick;
    j["activation_thresholds"] = json::binary(
        {mind.activation_thresholds.begin(), mind.activation_thresholds.end()});
    j["outputs_weights"] = json::binary(
        {mind.outputs_weights.begin(), mind.outputs_weights.end()});
    j["input_weights"] =
        json::binary({mind.input_weights.begin(), mind.input_weights.end()});
    j["reactivation_delays"] = json::binary(
        {mind.reactivation_delays.begin(), mind.reactivation_delays.end()});
    j["next_activations"] = json::binary(
        {mind.next_activations.begin(), mind.next_activations.end()});
    j["signal_map"] =
        json::binary({mind.signal_map.begin(), mind.signal_map.end()});
    return json::to_bson(j);
  }();

  return data;
}

export MindData mind_deserialize(const std::vector<std::uint8_t> &data) {
  using json = nlohmann::json;

  const json j = json::from_bson(data);

  if (const int version = j.at("version").get<int>(); version != 1) {
    throw std::runtime_error{
        std::format("invalid version: {} (expected 1)", version)};
  }

  const auto xt_load = [&j]<std::size_t Rank>(xt::xtensor<float, Rank> &tensor,
                                             const char *field_name) {
    const auto var = j.at(field_name).get_binary();
    const std::span tensor_data{reinterpret_cast<const float *>(var.data()),
                                var.size() / sizeof(float)};
    tensor = xt::adapt(tensor_data.data(), tensor_data.size(),
                       xt::no_ownership(), // span управляет жизнью данных
                       tensor.shape());
  };

  MindData mind = {};
  mind.tick = j.at("tick").get<decltype(MindData::tick)>();
  xt_load(mind.activation_thresholds, "activation_thresholds");
  xt_load(mind.outputs_weights, "outputs_weights"); 
  xt_load(mind.input_weights, "input_weights");
  xt_load(mind.reactivation_delays, "reactivation_delays");
  xt_load(mind.next_activations, "next_activations");
  xt_load(mind.signal_map, "signal_map");
  mind.neural_activity = xt::zeros<float>({mind.signal_map.size()});

  auto _ = mind_validate(mind).or_else(
      [](const auto &err) -> std::expected<void, std::string> {
        throw std::invalid_argument{err};
      });

  return mind;
}

} // namespace mind
