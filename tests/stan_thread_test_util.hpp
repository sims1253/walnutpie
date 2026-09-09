#pragma once
#include <walnutpie/load_stan.hpp>
struct ThreadStub {
  walnutpie::internal::dynamic_library library;
  explicit ThreadStub(const char* path)
      : library(walnutpie::internal::dlopen_safe(path)) {}
  void reset() {
    walnutpie::internal::dlsym_cast_impl<void (*)()>(library, "stub_reset")();
  }
  int calls(const char* symbol) {
    return walnutpie::internal::dlsym_cast_impl<int (*)()>(library, symbol)();
  }
};
