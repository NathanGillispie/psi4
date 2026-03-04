#include <pybind11/embed.h>
#include <string>

namespace {

#define BREAKPOINT(...) process_variables(#__VA_ARGS__, __VA_ARGS__)

template <typename... Args>
void process_variables(const char* names_str, Args... args) {
    pybind11::gil_scoped_acquire gil;
    pybind11::dict snapshot;

    std::string s = names_str;
    size_t pos = 0;

    ([&](auto& val) {
        size_t next_comma = s.find(',', pos);
        std::string name = s.substr(pos, next_comma - pos);

        // Clean up whitespace/commas
        name.erase(0, name.find_first_not_of(" ,"));
        name.erase(name.find_last_not_of(" ,") + 1);

		snapshot[name.c_str()] = val;
        pos = next_comma + 1;
    }(args), ...);

    pybind11::module_ code = pybind11::module_::import("code");
    code.attr("interact")(
        "C++ debug REPL (locals: "+s+"):",
        pybind11::none(),
        snapshot
    );
}

}

