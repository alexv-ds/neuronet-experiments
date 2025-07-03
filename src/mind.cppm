module;
#include <expected>
#include <format>
#include <nlohmann/json.hpp>
#include <xtensor/containers/xtensor.hpp>
#include <xtensor/io/xnpy.hpp>
#include <xtensor/views/xview.hpp>
#include <zstd.h>
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
 * - Computing incoming signals to each neuron based on input weights and signal map.
 * - Applying an activation function using thresholds and reactivation delays.
 * - Propagating outputs to update the signal map using output weights.
 *
 * The process includes three main phases:
 * 1. **Signal accumulation**: Neurons that are ready (`next_activations <= tick`) receive weighted input signals.
 * 2. **Activation decision**: Neurons whose activity exceeds the threshold become active and get delayed for the next activation.
 * 3. **Signal propagation**: Active neurons propagate their signal through the output weights.
 *
 * This function modifies `neural_activity`, `next_activations`, and `signal_map` in-place.
 *
 * @param mind A reference to the `MindData` structure containing the full network state.
 *
 * @see mind_validate
 * @see MindData
 */
export void mind_step(MindData &mind) {
  const auto tick = static_cast<float>(mind.tick);
  const std::ptrdiff_t neurons_count = mind.neural_activity.size();

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
      mind.next_activations(i) = tick + mind.reactivation_delays(i);
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

/**
 * @brief Serializes a MindData instance into a compressed binary format.
 *
 * This function performs the following steps:
 * 1. Validates the integrity of the given `MindData` object using `mind_validate`.
 *    - If validation fails, it throws `std::invalid_argument` with the error message.
 * 2. Converts the internal fields of `MindData` into a BSON-formatted binary JSON structure:
 *    - Scalar and tensor fields are serialized using `nlohmann::json` and `xt::dump_npy`.
 *    - Tensor data is stored in `.npy` format and wrapped as binary JSON fields.
 * 3. Compresses the resulting BSON document using Zstandard (`ZSTD_compress`) with the specified compression level.
 *
 * The output is a compact binary blob suitable for storage or transmission,
 * and can later be deserialized using a matching `mind_deserialize` function.
 *
 * @param mind The `MindData` instance to serialize.
 * @param compression_level Compression level for Zstandard (default: `ZSTD_defaultCLevel()`).
 *                          Valid range is typically 1 (fastest) to 22 (maximum compression).
 *
 * @return A `std::vector<std::uint8_t>` containing the compressed serialized data.
 *
 * @throws std::invalid_argument If `mind` fails validation.
 * @throws std::runtime_error If Zstandard compression fails.
 *
 * @see mind_validate
 * @see xt::dump_npy
 * @see ZSTD_compress
 */
export std::vector<std::uint8_t>
mind_serialize(const MindData &mind,
               const int compression_level = ZSTD_defaultCLevel()) {
  mind_validate(mind).or_else(
      [](const auto &err) -> std::expected<void, std::string> {
        throw std::invalid_argument{err};
      });

  const std::vector<std::uint8_t> uncompressed = [&mind] {
    using json = nlohmann::json;
    json j;
    j["version"] = 1;
    j["tick"] = mind.tick;
    j["activation_thresholds"] = [&mind] {
      const auto npy_dump = xt::dump_npy(mind.activation_thresholds);
      return json::binary({npy_dump.begin(), npy_dump.end()});
    }();
    j["outputs_weights"] =
        [&mind] {
          const auto npy_dump = xt::dump_npy(mind.outputs_weights);
          return json::binary({npy_dump.begin(), npy_dump.end()});
        }(),
    j["input_weights"] =
        [&mind] {
          const auto npy_dump = xt::dump_npy(mind.input_weights);
          return json::binary({npy_dump.begin(), npy_dump.end()});
        }(),
    j["reactivation_delays"] =
        [&mind] {
          const auto npy_dump = xt::dump_npy(mind.reactivation_delays);
          return json::binary({npy_dump.begin(), npy_dump.end()});
        }(),
    j["next_activations"] =
        [&mind] {
          const auto npy_dump = xt::dump_npy(mind.next_activations);
          return json::binary({npy_dump.begin(), npy_dump.end()});
        }(),
    j["signal_map"] = [&mind] {
      const auto npy_dump = xt::dump_npy(mind.signal_map);
      return json::binary({npy_dump.begin(), npy_dump.end()});
    }();
    return json::to_bson(j);
  }();

  std::vector<std::uint8_t> compressed(ZSTD_compressBound(uncompressed.size()));
  const std::size_t compressed_size =
      ZSTD_compress(compressed.data(), compressed.size(), uncompressed.data(),
                    uncompressed.size(), compression_level);
  if (ZSTD_isError(compressed_size)) {
    throw std::runtime_error{std::format("compression error: {}",
                                         ZSTD_getErrorName(compressed_size))};
  }
  compressed.resize(compressed_size);

  return compressed;
}

/**
 * @brief Deserializes a compressed binary buffer into a MindData instance.
 *
 * This function performs the inverse operation of `mind_serialize`. It:
 * 1. Decompresses the given buffer using Zstandard (`ZSTD_decompress`).
 *    - Validates frame content size before decompression to avoid zip-bombs.
 *    - Limits decompressed size to 100 GiB for safety.
 * 2. Parses the decompressed data as BSON and reconstructs a JSON object.
 * 3. Deserializes individual fields from the JSON:
 *    - Scalar values via `nlohmann::json`.
 *    - Tensor values (stored in `.npy` format) via `xt::load_npy`.
 * 4. Constructs a `MindData` object from the deserialized fields.
 *    - Initializes `neural_activity` as a zero tensor with appropriate size.
 * 5. Validates the resulting object using `mind_validate`.
 *    - Throws `std::invalid_argument` if validation fails.
 *
 * @param data A `std::vector<std::uint8_t>` containing the compressed binary representation
 *             produced by `mind_serialize`.
 * @return A fully reconstructed and validated `MindData` object.
 *
 * @throws std::invalid_argument If:
 * - The decompressed size is unknown, invalid, or exceeds the 100 GiB safety threshold.
 * - The data cannot be parsed as BSON/JSON.
 * - The deserialized version is incompatible.
 * - Structural validation fails.
 *
 * @throws std::runtime_error If decompression fails (e.g., invalid or corrupted Zstd stream).
 *
 * @see mind_serialize
 * @see mind_validate
 * @see xt::load_npy
 * @see ZSTD_decompress
 */
export MindData mind_deserialize(const std::vector<std::uint8_t> &data) {
  using json = nlohmann::json;

  const json j = [&data] {
    const auto estimated_decompressed_size =
        ZSTD_getFrameContentSize(data.data(), data.size());
    if (estimated_decompressed_size == ZSTD_CONTENTSIZE_UNKNOWN) {
      throw std::invalid_argument{"cannot decompress: content size is unknown"};
    }
    if (estimated_decompressed_size == ZSTD_CONTENTSIZE_ERROR) {
      throw std::invalid_argument{
          "cannot decompress: content size reading error"};
    }

    // zip-bombs not allowed, 100 GiB max
    if (estimated_decompressed_size > 100ull * 1024ull * 1024ull * 1024ull) {
      throw std::invalid_argument{std::format(
          "cannot decompress: decompressed size is too large (max 100 GiB)")};
    }
    std::vector<std::uint8_t> decompressed(estimated_decompressed_size);

    const auto decompressed_size = ZSTD_decompress(
        decompressed.data(), decompressed.size(), data.data(), data.size());
    if (ZSTD_isError(decompressed_size)) {
      throw std::runtime_error{std::format(
          "cannot decompress: {}", ZSTD_getErrorName(decompressed_size))};
    }
    decompressed.resize(decompressed_size);
    return json::from_bson(decompressed);
  }();

  if (const int version = j.at("version").get<int>(); version != 1) {
    throw std::runtime_error{
        std::format("invalid version: {} (expected 1)", version)};
  }

  const auto xt_load = [&j](const char *field_name) {
    const auto var = j.at(field_name).get_binary();
    std::stringbuf sbuf{std::string_view(
        reinterpret_cast<const char *>(var.data()), var.size())};
    std::istream is{&sbuf};
    return xt::load_npy<float>(is);
  };

  MindData mind = {
      .tick = j.at("tick").get<decltype(MindData::tick)>(),
      .activation_thresholds = xt_load("activation_thresholds"),
      .outputs_weights = xt_load("outputs_weights"),
      .input_weights = xt_load("input_weights"),
      .reactivation_delays = xt_load("reactivation_delays"),
      .next_activations = xt_load("next_activations"),
      .signal_map = xt_load("signal_map"),
      .neural_activity = {}
  };
  mind.neural_activity = xt::zeros<float>({mind.signal_map.size()});

  mind_validate(mind).or_else(
      [](const auto &err) -> std::expected<void, std::string> {
        throw std::invalid_argument{err};
      });

  return mind;
}

} // namespace mind
