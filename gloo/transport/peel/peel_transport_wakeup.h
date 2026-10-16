// Internal local stop signal for the PEEL Reno ACK aggregator.
#pragma once

#include <cerrno>
#include <cstdint>
#include <sys/eventfd.h>
#include <sys/select.h>
#include <unistd.h>

namespace gloo {
namespace transport {
namespace peel {
namespace detail {

class PeelLocalWakeup {
 public:
  PeelLocalWakeup() : fd_(::eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK)) {
    // select() cannot represent descriptors outside its fixed fd_set.
    if (fd_ >= FD_SETSIZE) {
      ::close(fd_);
      fd_ = -1;
      errno = EMFILE;
    }
  }

  ~PeelLocalWakeup() {
    if (fd_ >= 0) {
      ::close(fd_);
    }
  }

  PeelLocalWakeup(const PeelLocalWakeup&) = delete;
  PeelLocalWakeup& operator=(const PeelLocalWakeup&) = delete;

  bool valid() const noexcept { return fd_ >= 0; }
  int fd() const noexcept { return fd_; }

  bool signal() const noexcept {
    const uint64_t value = 1;
    ssize_t result;
    do {
      result = ::write(fd_, &value, sizeof(value));
    } while (result < 0 && errno == EINTR);
    // A saturated eventfd is already readable, so no further write is needed.
    return result == sizeof(value) ||
        (result < 0 && (errno == EAGAIN || errno == EWOULDBLOCK));
  }

 private:
  int fd_;
};

} // namespace detail
} // namespace peel
} // namespace transport
} // namespace gloo

