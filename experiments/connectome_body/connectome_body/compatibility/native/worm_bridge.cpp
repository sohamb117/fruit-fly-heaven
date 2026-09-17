// C ABI for the separately downloaded, pinned Boyle/Izquierdo/Beer body model.
// No neural circuit or feedback controller is linked into this library.
#include "WormBody.h"
#include <array>
#include <cstring>
#include <mutex>
#include <new>
#include <type_traits>

#define API extern "C" __attribute__((visibility("default")))
struct Simulation {
    WormBody body;
    std::array<double, 48> muscle;
};
static_assert(std::is_trivially_copyable<Simulation>::value,
              "Snapshots require a pointer-free, trivially copyable body");
static std::once_flag constants_initialized;

static bool finite_body(Simulation* sim) {
    for (int i = 1; i <= N_rods; ++i)
        if (!std::isfinite(sim->body.X(i)) || !std::isfinite(sim->body.Y(i)) ||
            !std::isfinite(sim->body.Phi(i))) return false;
    for (double value : sim->muscle) if (!std::isfinite(value)) return false;
    return true;
}

API std::size_t ceworm_state_bytes() { return sizeof(Simulation); }
API void* ceworm_create() {
    std::call_once(constants_initialized, InitializeBodyConstants);
    auto* sim = new (std::nothrow) Simulation;
    if (!sim) return nullptr;
    std::memset(sim, 0, sizeof(Simulation));
    sim->body.InitializeBodyState();
    return sim;
}
API void ceworm_destroy(void* handle) { delete static_cast<Simulation*>(handle); }
API void ceworm_reset(void* handle) {
    auto* sim = static_cast<Simulation*>(handle);
    std::memset(sim, 0, sizeof(Simulation));
    sim->body.InitializeBodyState();
}
API int ceworm_step(void* handle, const double* input, const double* capacity,
                    double dt, int steps) {
    if (!handle || !input || !capacity || dt <= 0 || dt > 0.01 || steps < 1) return -1;
    auto* sim = static_cast<Simulation*>(handle);
    for (int step = 0; step < steps; ++step) {
        // Preserve the published body -> muscle update order and 100 ms
        // first-order muscle filter. Inputs replace only its neural controller.
        sim->body.StepBody(dt);
        for (int i = 0; i < 48; ++i) {
            if (!std::isfinite(input[i]) || !std::isfinite(capacity[i])) return -1;
            const double drive = std::fmax(0.0, std::fmin(1.0, input[i]));
            sim->muscle[i] += (dt / 0.1) * (drive - sim->muscle[i]);
        }
        auto activation = [&](int side, int muscle) {
            const int i = side * 24 + muscle;
            return sim->muscle[i] * std::fmax(0.0, std::fmin(1.0, capacity[i]));
        };
        for (int segment = 0; segment < 50; ++segment) {
            double values[2];
            for (int side = 0; side < 2; ++side) {
                if (segment < 2) values[side] = activation(side, 0) / 2.0;
                else if (segment >= 48) values[side] = activation(side, 23) / 2.0;
                else {
                    const int muscle = segment / 2 - 1;
                    values[side] = (activation(side, muscle) + activation(side, muscle + 1)) / 2.0;
                }
            }
            sim->body.SetDorsalSegmentActivation(segment + 1, values[0]);
            sim->body.SetVentralSegmentActivation(segment + 1, values[1]);
        }
        if (!finite_body(sim)) return -1;
    }
    return 0;
}
API void ceworm_observe(void* handle, double* rods, double* strains, double* muscles) {
    auto* sim = static_cast<Simulation*>(handle);
    for (int i = 0; i < 51; ++i) {
        rods[3*i] = sim->body.X(i+1);
        rods[3*i+1] = sim->body.Y(i+1);
        rods[3*i+2] = sim->body.Phi(i+1);
    }
    for (int i = 0; i < 50; ++i) {
        strains[i] = sim->body.DorsalSegmentLength(i+1) / sim->body.RestingLength(i+1) - 1;
        strains[50+i] = sim->body.VentralSegmentLength(i+1) / sim->body.RestingLength(i+1) - 1;
    }
    std::memcpy(muscles, sim->muscle.data(), 48 * sizeof(double));
}
API void ceworm_save(void* handle, unsigned char* output) {
    std::memcpy(output, handle, sizeof(Simulation));
}
API int ceworm_restore(void* handle, const unsigned char* state, std::size_t length) {
    if (length != sizeof(Simulation)) return -1;
    std::memcpy(handle, state, sizeof(Simulation));
    return finite_body(static_cast<Simulation*>(handle)) ? 0 : -1;
}
