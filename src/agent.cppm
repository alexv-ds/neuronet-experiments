module;
#include <cstddef>
#include <expected>
#include <random>
#include <stdexcept>
#include <xtensor/generators/xbuilder.hpp>

export module agent;
import mind;

export class Agent {
public:
  explicit Agent(std::size_t neurons);

private:
  mind::MindData mind_;
  std::string command_;

  void set_command(const std::string& command);

  void energize_neuron(std::size_t index, float value = 1.0f);

  void validate() const;
};

// //////////////////////// //
// ///////// IMPL ///////// //
// //////////////////////// //
Agent::Agent(const std::size_t neurons) {
  mind_.tick = 0;
  mind_.activation_thresholds = xt::zeros<float>({neurons});
  mind_.outputs_weights = xt::zeros<float>({neurons, neurons});
  mind_.input_weights = xt::zeros<float>({neurons, neurons});
  mind_.reactivation_delays = xt::zeros<float>({neurons});
  mind_.next_activations = xt::zeros<float>({neurons});
  mind_.signal_map = xt::zeros<float>({neurons});
  mind_.neural_activity = xt::zeros<float>({neurons});
  validate();
}

// void Agent::set_command(const std::string& command) {
//   for (const char& c=[]
//   }
//   command_ = command;
// }

void Agent::energize_neuron(const std::size_t index, const float value) {
  if (index >= mind_.signal_map.size()) {
    throw std::invalid_argument("index out of range");
  }
  mind_.signal_map(index) = value;
  mind_.next_activations(index) = 0.0f;
}

void Agent::validate() const {
  mind_validate(mind_).or_else( // NOLINT(*-unused-return-value)
      [](const auto &err) -> std::expected<void, std::string> {
        throw std::runtime_error{err};
      });
}

