module;
#include <cstddef>
#include <stdexcept>
#include <expected>

export module agent;
import mind;

export class Agent {
public:
  Agent(std::size_t neurons);

private:
  mind::MindData mind_;
};


// //////////////////////// //
// ///////// IMPL ///////// //
// //////////////////////// //
Agent::Agent(const std::size_t neurons) {
  



  mind_validate(mind_).or_else(
      [](const auto &err) -> std::expected<void, std::string> {
        throw std::runtime_error{err};
      });
}

