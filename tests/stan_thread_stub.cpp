// Thread-safe test double, not a Stan model or a race detector.
#include <bridgestan.h>
#include <atomic>
#include <cstdlib>
#include <string>
struct bs_model {
  std::string info;
};
struct bs_rng {};
namespace {
std::atomic<int> grad_calls{0}, init_calls{0}, constrain_calls{0};
}
extern "C" {
BS_PUBLIC void stub_reset() {
  grad_calls = 0;
  init_calls = 0;
  constrain_calls = 0;
}
BS_PUBLIC int stub_gradient_calls() { return grad_calls.load(); }
BS_PUBLIC int stub_init_calls() { return init_calls.load(); }
BS_PUBLIC int stub_constrain_calls() { return constrain_calls.load(); }
bs_model* bs_model_construct(const char* data, unsigned int, char**) {
  return new bs_model{data ? data : ""};
}
void bs_model_destruct(bs_model* m) { delete m; }
void bs_free_error_msg(char* p) { std::free(p); }
#ifndef OMIT_MODEL_INFO
const char* bs_model_info(const bs_model* m) {
  return m->info == "<null>" ? nullptr : m->info.c_str();
}
#endif
int bs_param_unc_num(const bs_model*) { return 1; }
int bs_param_num(const bs_model*, bool, bool) { return 1; }
const char* bs_param_names(const bs_model*, bool, bool) { return "x"; }
int bs_set_print_callback(STREAM_CALLBACK, char**) { return 0; }
bs_rng* bs_rng_construct(unsigned int, char**) { return new bs_rng; }
void bs_rng_destruct(bs_rng* p) { delete p; }
int bs_param_initialize(const bs_model*, const char*, bs_rng*, double, int,
                        bool, double* x, char**) {
  ++init_calls;
  *x = 0.2;
  return 0;
}
int bs_param_constrain(const bs_model*, bool, bool, const double* x, double* y,
                       bs_rng*, char**) {
  ++constrain_calls;
  *y = *x;
  return 0;
}
int bs_log_density_gradient(const bs_model*, bool, bool, const double* x,
                            double* lp, double* g, char**) {
  ++grad_calls;
  *lp = -0.5 * (*x) * (*x);
  *g = -*x;
  return 0;
}
}
