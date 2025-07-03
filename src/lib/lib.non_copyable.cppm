export module lib.non_copyable;

namespace lib {
export class NonCopyable {
public:
  NonCopyable() = default;
  ~NonCopyable() = default;

  NonCopyable(const NonCopyable &) = delete;
  NonCopyable &operator=(const NonCopyable &) = delete;
};
} // namespace lib