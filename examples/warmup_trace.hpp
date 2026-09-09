#pragma once

#include <Eigen/Dense>
#include <bit>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

namespace warmup_trace {
using Flag = std::variant<std::string, double, std::uint64_t, bool>;
struct Settings {
  std::filesystem::path directory;
  std::string model;
  std::map<std::string, Flag> flags;
};

inline bool valid_utf8(const std::string& text) {
  for (std::size_t i = 0; i < text.size();) {
    auto c = static_cast<unsigned char>(text[i++]);
    if (c < 0x80) {
      continue;
    }
    unsigned count;
    std::uint32_t code;
    if (c >= 0xc2 && c <= 0xdf) {
      count = 1;
      code = c & 0x1f;
    } else if (c >= 0xe0 && c <= 0xef) {
      count = 2;
      code = c & 0x0f;
    } else if (c >= 0xf0 && c <= 0xf4) {
      count = 3;
      code = c & 0x07;
    } else {
      return false;
    }
    if (i + count > text.size()) {
      return false;
    }
    for (unsigned j = 0; j < count; ++j) {
      auto next = static_cast<unsigned char>(text[i++]);
      if ((next & 0xc0) != 0x80) {
        return false;
      }
      code = (code << 6) | (next & 0x3f);
    }
    if ((count == 1 && code < 0x80) || (count == 2 && code < 0x800) ||
        (count == 3 && code < 0x10000) || code > 0x10ffff ||
        (code >= 0xd800 && code <= 0xdfff)) {
      return false;
    }
  }
  return true;
}
inline std::string quote(const std::string& value) {
  if (!valid_utf8(value)) {
    throw std::invalid_argument("trace metadata requires UTF-8 strings");
  }
  static constexpr char hex[] = "0123456789abcdef";
  std::string result = "\"";
  for (unsigned char c : value) {
    if (c == '"' || c == '\\') {
      result += '\\';
      result += static_cast<char>(c);
    } else if (c < 0x20) {
      result += "\\u00";
      result += hex[c >> 4];
      result += hex[c & 15];
    } else {
      result += static_cast<char>(c);
    }
  }
  return result + '"';
}
inline void number(std::ostream& out, double value) {
  if (std::isfinite(value)) {
    out << value;
  } else {
    out << "null";
  }
}
inline void write_u64(std::ostream& out, std::uint64_t value) {
  char bytes[8];
  for (unsigned i = 0; i < 8; ++i) {
    bytes[i] = static_cast<char>((value >> (8 * i)) & 255);
  }
  out.write(bytes, 8);
}
inline void write_f64(std::ostream& out, double value) {
  static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559,
                "warmup trace requires IEEE754 binary64");
  write_u64(out, std::bit_cast<std::uint64_t>(value));
}

template <typename Write>
void file(const std::filesystem::path& path, Write&& write) {
  try {
    std::ofstream out;
    out.exceptions(std::ios::failbit | std::ios::badbit);
    out.open(path, std::ios::binary);
    out.imbue(std::locale::classic());
    out << std::setprecision(17);
    write(out);
    out.close();  // A close failure must prevent publication of the manifest.
  } catch (const std::exception& error) {
    throw std::runtime_error("warmup trace file " + path.string() + ": " +
                             error.what());
  }
}

class Writer {
 public:
  Writer(const Writer&) = delete;
  Writer& operator=(const Writer&) = delete;
  Writer(Writer&&) = delete;
  Writer& operator=(Writer&&) = delete;
  Writer(const Settings& settings, const Eigen::VectorXd& initial_position,
         const Eigen::VectorXd& initial_mass, std::size_t requested_warmup,
         std::uint64_t seed)
      : settings_(settings),
        initial_position_(initial_position),
        initial_mass_(initial_mass),
        dim_(initial_position.size()),
        requested_(requested_warmup),
        seed_(seed) {
    if (initial_mass.size() != dim_) {
      throw std::invalid_argument("trace initial mass dimension mismatch");
    }
    // Validate all JSON strings before claiming the output directory.
    quote(settings_.model);
    for (const auto& [key, value] : settings_.flags) {
      quote(key);
      if (auto string = std::get_if<std::string>(&value)) {
        quote(*string);
      }
    }
    if (settings_.directory.empty() ||
        !std::filesystem::create_directory(settings_.directory)) {
      throw std::runtime_error("warmup trace requires a new directory: " +
                               settings_.directory.string());
    }
  }
  void record(const Eigen::VectorXd& theta, const Eigen::VectorXd& grad,
              double lp, double step, const Eigen::VectorXd& invmass,
              std::size_t depth) {
    if (finished_) {
      throw std::logic_error("trace already finished");
    }
    if (theta.size() != dim_ || grad.size() != dim_ || invmass.size() != dim_) {
      throw std::invalid_argument("trace row dimension mismatch");
    }
    if (depths_.size() >= requested_) {
      throw std::invalid_argument("too many warmup trace rows");
    }
    append(theta_, theta);
    append(grad_, grad);
    append(invmass_, invmass);
    lp_.push_back(lp);
    step_.push_back(step);
    depths_.push_back(depth);
  }
  std::size_t rows() const noexcept { return depths_.size(); }
  void finish() {
    if (finished_) {
      throw std::logic_error("trace already finished");
    }
    if (rows() != requested_ || lp_.size() != rows() ||
        step_.size() != rows() || theta_.size() != rows() * dim_ ||
        grad_.size() != theta_.size() || invmass_.size() != theta_.size()) {
      throw std::logic_error("incomplete warmup trace");
    }
    doubles("theta.f64", theta_);
    doubles("grad.f64", grad_);
    doubles("invmass.f64", invmass_);
    doubles("step.f64", step_);
    doubles("lp.f64", lp_);
    file(settings_.directory / "depth.u64", [&](auto& out) {
      for (auto d : depths_) {
        write_u64(out, d);
      }
    });
    auto temporary = settings_.directory / "meta.json.tmp";
    file(temporary, [&](auto& out) { metadata(out); });
    // Atomic publication on the same filesystem, only after every stream closed
    // successfully.
    std::filesystem::rename(temporary, settings_.directory / "meta.json");
    finished_ = true;
  }

 private:
  static void append(std::vector<double>& values, const Eigen::VectorXd& row) {
    for (Eigen::Index i = 0; i < row.size(); ++i) {
      values.push_back(row[i]);
    }
  }
  void doubles(const char* name, const std::vector<double>& values) {
    file(settings_.directory / name, [&](auto& out) {
      for (double value : values) {
        write_f64(out, value);
      }
    });
  }
  static void vector_json(std::ostream& out, const Eigen::VectorXd& values) {
    out << '[';
    for (Eigen::Index i = 0; i < values.size(); ++i) {
      if (i) {
        out << ',';
      }
      number(out, values[i]);
    }
    out << ']';
  }
  void metadata(std::ostream& out) const {
    out << "{\n\"format_version\":1,\"complete\":true,\"byte_order\":"
           "\"little\","
        << "\"nonfinite_metadata\":\"null\",\"invmass_phase\":\"pre-"
           "transition\","
        << "\"step_phase\":\"post-adaptation\",\n\"model\":"
        << quote(settings_.model) << ",\"dim\":" << dim_
        << ",\"num_warmup\":" << requested_ << ",\"seed\":" << seed_
        << ",\"warmup_iters_recorded\":" << rows() << ",\"flags\":{";
    bool first = true;
    for (const auto& [key, value] : settings_.flags) {
      if (!first) {
        out << ',';
      }
      first = false;
      out << quote(key) << ':';
      std::visit(
          [&](const auto& v) {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, std::string>) {
              out << quote(v);
            } else if constexpr (std::is_same_v<T, double>) {
              number(out, v);
            } else if constexpr (std::is_same_v<T, bool>) {
              out << (v ? "true" : "false");
            } else {
              out << v;
            }
          },
          value);
    }
    out << "},\"initial_position\":";
    vector_json(out, initial_position_);
    out << ",\"initial_mass\":";
    vector_json(out, initial_mass_);
    out << "}\n";
  }
  Settings settings_;
  Eigen::VectorXd initial_position_, initial_mass_;
  Eigen::Index dim_;
  std::size_t requested_;
  std::uint64_t seed_;
  std::vector<double> theta_, grad_, invmass_, lp_, step_;
  std::vector<std::uint64_t> depths_;
  bool finished_ = false;
};
// Non-owning dispatch used by the CLI handler. Disabled tracing retains no
// vectors.
struct Sink {
  Writer* writer = nullptr;
  void record(const Eigen::VectorXd& theta, const Eigen::VectorXd& grad,
              double lp, double step, const Eigen::VectorXd& invmass,
              std::size_t depth) const {
    if (writer) {
      writer->record(theta, grad, lp, step, invmass, depth);
    }
  }
};
}  // namespace warmup_trace
