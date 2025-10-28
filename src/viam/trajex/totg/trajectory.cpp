#include <viam/trajex/totg/trajectory.hpp>

#include <cassert>
#include <functional>
#include <numeric>
#include <ranges>
#include <stdexcept>

#include <viam/trajex/totg/path.hpp>
#include <viam/trajex/types/arc_length.hpp>

namespace viam::trajex::totg {

namespace {

// Computes maximum path velocity (s_dot) from joint velocity and acceleration limits.
// Two constraints apply: centripetal acceleration from path curvature (eq 31) and
// direct velocity limits (eq 36). Returns both so caller can take the minimum.
// See Kunz & Stilman equations 31 and 36.
[[gnu::pure]] [[maybe_unused]] auto compute_velocity_limits(const xt::xarray<double>& q_prime,
                                                            const xt::xarray<double>& q_double_prime,
                                                            const xt::xarray<double>& q_dot_max,
                                                            const xt::xarray<double>& q_ddot_max,
                                                            double epsilon) {
    struct result {
        double s_dot_max_acc;
        double s_dot_max_vel;
    };

    // Compute the path velocity limit imposed by joint acceleration constraints (equation 31).
    // This is the acceleration limit curve in the phase plane. The derivation in the paper
    // converts joint acceleration bounds into constraints on path velocity by considering
    // the centripetal acceleration term q''(s)*s_dot^2 that appears when following a curved
    // path. The result is a set of downward-facing parabolas centered at the origin, and we
    // take the minimum of their positive bounds to find the feasible path velocity.
    double s_dot_max_accel = std::numeric_limits<double>::infinity();

    // For each pair of joints that are moving along the path, compare their curvature ratios
    // q''(s)/q'(s). When these ratios differ, the joints are curving at different rates, which
    // limits how fast we can traverse the path. This is the pairwise constraint from equation 31.
    for (size_t i = 0; i < q_prime.size(); ++i) {
        if (std::abs(q_prime(i)) < epsilon) {
            continue;
        }

        for (size_t j = i + 1; j < q_prime.size(); ++j) {
            if (std::abs(q_prime(j)) < epsilon) {
                continue;
            }

            const double curvature_ratio_i = q_double_prime(i) / q_prime(i);
            const double curvature_ratio_j = q_double_prime(j) / q_prime(j);
            const double curvature_difference = std::abs(curvature_ratio_i - curvature_ratio_j);

            if (curvature_difference < epsilon) {
                continue;
            }

            const double accel_sum = (q_ddot_max(i) / std::abs(q_prime(i))) + (q_ddot_max(j) / std::abs(q_prime(j)));
            const double limit = std::sqrt(accel_sum / curvature_difference);
            s_dot_max_accel = std::min(s_dot_max_accel, limit);
        }
    }

    // A joint that is stationary in path space (q'(s) = 0) but has non-zero curvature (q''(s) != 0)
    // also constrains the path velocity. From equations 19-20, when q'(s) = 0, the constraint
    // q''(s)*s_dot^2 <= q_ddot_max directly limits s_dot. This case appears at points where a joint
    // reaches a local extremum along the path while the path itself is curved.
    for (size_t i = 0; i < q_prime.size(); ++i) {
        if (std::abs(q_prime(i)) >= epsilon) {
            continue;
        }

        if (std::abs(q_double_prime(i)) < epsilon) {
            continue;
        }

        const double limit = std::sqrt(q_ddot_max(i) / std::abs(q_double_prime(i)));
        s_dot_max_accel = std::min(s_dot_max_accel, limit);
    }

    // Compute the path velocity limit imposed by joint velocity constraints (equation 36).
    // This is the velocity limit curve in the phase plane. For each joint moving along
    // the path, the joint velocity q_dot = q'(s)*s_dot must respect the joint velocity
    // limit, giving us s_dot <= q_dot_max / |q'(s)|. We take the minimum across all joints.
    double s_dot_max_vel = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < q_prime.size(); ++i) {
        if (std::abs(q_prime(i)) < epsilon) {
            continue;
        }

        const double limit = q_dot_max(i) / std::abs(q_prime(i));
        s_dot_max_vel = std::min(s_dot_max_vel, limit);
    }

    return result{s_dot_max_accel, s_dot_max_vel};
}

// Computes the feasible range of path acceleration (s_ddot) given current path velocity (s_dot)
// and joint acceleration limits. The path acceleration must satisfy joint constraints in all DOF.
// Using chain rule q''(t) = q'(s)*s_ddot + q''(s)*s_dot^2, we solve for s_ddot bounds.
// See Kunz & Stilman equations 22-23.
[[gnu::pure]] [[maybe_unused]] auto compute_acceleration_bounds(const xt::xarray<double>& q_prime,
                                                                const xt::xarray<double>& q_double_prime,
                                                                double s_dot,
                                                                const xt::xarray<double>& q_ddot_max,
                                                                double epsilon) {
    struct result {
        double s_ddot_min;
        double s_ddot_max;
    };

    double s_ddot_min = -std::numeric_limits<double>::infinity();
    double s_ddot_max = std::numeric_limits<double>::infinity();

    // Each joint independently constrains the feasible range of path acceleration (equations 22-23).
    // From the chain rule q''(t) = q'(s)*s_ddot + q''(s)*s_dot^2, we can solve for s_ddot given
    // the constraint -q_ddot_max <= q''(t) <= q_ddot_max. The centripetal term q''(s)*s_dot^2 is
    // already "using up" part of the acceleration budget at the current path velocity, which tightens
    // the bounds on how much path acceleration we can apply. Each joint shrinks the feasible region,
    // so we take the intersection by using max for lower bounds and min for upper bounds.
    for (size_t i = 0; i < q_prime.size(); ++i) {
        if (std::abs(q_prime(i)) < epsilon) {
            continue;
        }

        const double centripetal_term = q_double_prime(i) * s_dot * s_dot;

        const double min_from_joint = (-q_ddot_max(i) - centripetal_term) / q_prime(i);
        s_ddot_min = std::max(s_ddot_min, min_from_joint);

        const double max_from_joint = (q_ddot_max(i) - centripetal_term) / q_prime(i);
        s_ddot_max = std::min(s_ddot_max, max_from_joint);
    }

    return result{s_ddot_min, s_ddot_max};
}

}  // namespace

trajectory::trajectory(class path p, options opt, integration_points points)
    : path_{std::move(p)}, options_{std::move(opt)}, integration_points_{std::move(points)} {
    if (path_.empty() || path_.length() <= arc_length{0.0}) {
        throw std::invalid_argument{"Path must not be empty"};
    }

    // Must have at least 2 integration points (start and end)
    if (integration_points_.size() < 2) {
        throw std::invalid_argument{"Trajectory must have at least 2 integration points"};
    }

    // First point must be at t=0, s=0 (trajectory start)
    if (integration_points_.front().time != seconds{0.0}) {
        throw std::invalid_argument{"First integration point must have time == 0"};
    }

    if (integration_points_.front().s != arc_length{0.0}) {
        throw std::invalid_argument{"First integration point must have arc length == 0"};
    }

    // Last point must reach end of path
    if (integration_points_.back().s != path_.length()) {
        throw std::invalid_argument{"Last integration point must have arc length == path.length()"};
    }

    // Set duration from last integration point
    duration_ = integration_points_.back().time;
}

trajectory trajectory::create(class path p, options opt, integration_points test_points) {
    if (opt.max_velocity.shape(0) != p.dof()) {
        throw std::invalid_argument{"max_velocity DOF doesn't match path DOF"};
    }

    if (opt.max_acceleration.shape(0) != p.dof()) {
        throw std::invalid_argument{"max_acceleration DOF doesn't match path DOF"};
    }

    if (!xt::all(xt::isfinite(opt.max_velocity) && opt.max_velocity >= 0.0)) {
        throw std::invalid_argument{"max_velocity must be finite and non-negative"};
    }

    if (!xt::all(xt::isfinite(opt.max_acceleration) && opt.max_acceleration >= 0.0)) {
        throw std::invalid_argument{"max_acceleration must be finite and non-negative"};
    }

    if (opt.delta <= seconds{0.0}) {
        throw std::invalid_argument{"delta must be positive"};
    }

    if (opt.epsilon <= 0.0) {
        throw std::invalid_argument{"epsilon must be positive"};
    }

    if (test_points.empty()) {
        // Production path: run the TOTG algorithm to compute time-optimal parameterization.
        // TODO(acm): Replace this stub with real TOTG implementation (see plan.md Item 8).
        //
        // The TOTG algorithm uses phase plane (s, s_dot) integration per Kunz & Stilman:
        // 1. Forward pass: integrate with maximum acceleration to build Maximum Velocity Curve (MVC)
        // 2. Backward pass: integrate with minimum acceleration from path end to find optimal trajectory
        // 3. Result: time parameterization respecting all velocity and acceleration constraints
        //
        // For now, create a fake constant-velocity trajectory to enable testing of the
        // sampling infrastructure without requiring the full TOTG implementation.

        const double path_length = static_cast<double>(p.length());

        const double sum = std::accumulate(opt.max_velocity.begin(), opt.max_velocity.end(), 0.0);
        const double avg_max_vel = opt.max_velocity.size() > 0 ? sum / static_cast<double>(opt.max_velocity.size()) : 1.0;
        const double fake_duration = (avg_max_vel > 0.0) ? path_length / avg_max_vel : 1.0;

        const int num_steps = std::max(1, static_cast<int>(std::ceil(fake_duration / opt.delta.count())));
        const double actual_delta = fake_duration / static_cast<double>(num_steps);

        test_points.reserve(static_cast<size_t>(num_steps) + 1);

        auto step_indices = std::views::iota(0, num_steps + 1);
        for (int i : step_indices) {
            const double t = static_cast<double>(i) * actual_delta;
            // Force last point to exactly path.length() to avoid floating point error accumulation
            // during linear interpolation, which could cause queries to exceed path bounds.
            const arc_length s = (i == num_steps) ? p.length() : arc_length{path_length * (t / fake_duration)};
            test_points.push_back({
                .time = seconds{t},
                .s = s,
                .s_dot = 0.0,  // Zero velocity (fake)
                .s_ddot = 0.0  // Zero acceleration (fake)
            });
        }
    } else {
        // Test path: validate provided integration points
        // First point must be at t=0
        if (test_points.front().time != seconds{0.0}) {
            throw std::invalid_argument{"First integration point must have time == 0"};
        }

        // Pairwise validation using C++20 ranges
        auto pairs = std::views::iota(size_t{0}, test_points.size() - 1) |
                     std::views::transform([&](size_t i) { return std::pair{std::cref(test_points[i]), std::cref(test_points[i + 1])}; });

        for (const auto& [curr, next] : pairs) {
            // Times must be strictly increasing
            if (next.get().time <= curr.get().time) {
                throw std::invalid_argument{"Integration points must be sorted by strictly increasing time"};
            }

            // Arc lengths must be non-negative and within path bounds
            if (curr.get().s < arc_length{0.0} || curr.get().s > p.length()) {
                throw std::invalid_argument{"Integration point arc lengths must be in [0, path.length()]"};
            }

            // Arc lengths must be monotonically non-decreasing
            if (next.get().s < curr.get().s) {
                throw std::invalid_argument{"Integration point arc lengths must be monotonically non-decreasing"};
            }
        }

        if (test_points.back().s < arc_length{0.0} || test_points.back().s > p.length()) {
            throw std::invalid_argument{"Integration point arc lengths must be in [0, path.length()]"};
        }
    }

    // Construct trajectory with integration points
    // Duration is set automatically in constructor from last integration point
    return trajectory{std::move(p), std::move(opt), std::move(test_points)};
}

struct trajectory::sample trajectory::sample(trajectory::seconds t) const {
    if (t < trajectory::seconds{0.0} || t > duration_) {
        throw std::out_of_range{"Time out of trajectory bounds"};
    }

    return create_cursor().seek(t).sample();
}

trajectory::seconds trajectory::duration() const noexcept {
    return duration_;
}

const class path& trajectory::path() const noexcept {
    return path_;
}

size_t trajectory::dof() const noexcept {
    return path_.dof();
}

const trajectory::integration_points& trajectory::get_integration_points() const noexcept {
    return integration_points_;
}

const trajectory::options& trajectory::get_options() const noexcept {
    return options_;
}

trajectory::cursor trajectory::create_cursor() const {
    return cursor{this};
}

trajectory::cursor::cursor(const class trajectory* traj)
    : traj_{traj}, time_hint_{traj->integration_points_.begin()}, path_cursor_{traj->path_.create_cursor()} {
    // Constructor initializes cursor at trajectory start (t=0, s=0)
    // time_hint_ points to first integration point (if any)
    // path_cursor_ is at arc length 0 by default
}

const trajectory& trajectory::cursor::trajectory() const noexcept {
    return *traj_;
}

trajectory::seconds trajectory::cursor::time() const noexcept {
    return time_;
}

struct trajectory::sample trajectory::cursor::sample() const {
    if (*this == end()) [[unlikely]] {
        throw std::out_of_range{"Cannot sample cursor at sentinel position"};
    }

    assert(time_hint_ != traj_->get_integration_points().end());
    assert(time_hint_->time <= time_);

    // Use the time hint for O(1) lookup of the integration point at or before current time.
    // The hint is maintained by seek() to always point to the correct interval.
    const integration_point& p0 = *time_hint_;

    // Interpolate path space (s, s_dot, s_ddot) using piecewise constant acceleration between
    // integration points. This is the standard kinematic model: constant acceleration
    // produces linear velocity and quadratic position.
    const double dt = (time_ - p0.time).count();
    const double s_dot = p0.s_dot + (p0.s_ddot * dt);
    const double s_ddot = p0.s_ddot;

    // Query the path geometry at the current arc length position. The path_cursor_ has
    // already been positioned by update_path_cursor_position_ in seek().
    const auto q = path_cursor_.configuration();
    const auto q_prime = path_cursor_.tangent();
    const auto q_double_prime = path_cursor_.curvature();

    // Convert from path space (s, s_dot, s_ddot) to joint space (q, q_dot, q_ddot) using the chain rule.
    // This gives us the actual joint velocities and accelerations that result from
    // moving along the path at the computed path velocity and acceleration.
    //
    // q_dot(t) = q'(s) * s_dot(t)
    const auto q_dot = q_prime * s_dot;

    // q_ddot(t) = q'(s) * s_ddot(t) + q''(s) * s_dot(t)^2
    //
    // The second term captures the centripetal acceleration from following a curved path.
    const auto q_ddot = q_prime * s_ddot + q_double_prime * (s_dot * s_dot);

    return {.time = time_, .configuration = q, .velocity = q_dot, .acceleration = q_ddot};
}

void trajectory::cursor::update_path_cursor_position_(seconds t) {
    // Postcondition: seek() must position time_hint_ correctly
    assert(time_hint_ != traj_->get_integration_points().end());
    assert(time_hint_->time <= t);

    // Interpolate arc length at time t using constant acceleration motion from time_hint_
    // s(t) = s0 + s_dot0 * dt + 0.5 * s_ddot0 * dt^2
    // This assumes piecewise constant acceleration between TOTG integration points

    const auto& p0 = *time_hint_;
    const double dt = (t - p0.time).count();
    const double s_interp = static_cast<double>(p0.s) + (p0.s_dot * dt) + (0.5 * p0.s_ddot * dt * dt);

    // By construction, s_interp cannot exceed path.length() when t <= duration
    assert(s_interp <= static_cast<double>(traj_->path_.length()));

    // Position path cursor at interpolated arc length
    path_cursor_.seek(arc_length{s_interp});
}

trajectory::cursor& trajectory::cursor::seek(seconds t) {
    // Use +/-infinity sentinels for out-of-bounds positions, matching the pattern used by
    // path::cursor. This allows algorithms to detect when they've stepped beyond trajectory
    // bounds without throwing exceptions on every overstep.
    if (t < seconds{0.0}) {
        time_ = seconds{-std::numeric_limits<double>::infinity()};
        return *this;
    }
    if (t > traj_->duration()) {
        time_ = seconds{std::numeric_limits<double>::infinity()};
        return *this;
    }

    time_ = t;

    assert(time_hint_ != traj_->get_integration_points().end());

    // Maintain a hint iterator pointing to the integration point at or before current time.
    // This provides O(1) amortized performance for sequential time queries (the common case
    // during sampling) by checking nearby integration points before falling back to binary
    // search for large time jumps.
    //
    // Strategy:
    // 1. Check if current hint is still valid (small forward step within same interval)
    // 2. Check adjacent intervals (common during uniform sampling)
    // 3. Binary search for large jumps (rare, but handles arbitrary seeks)

    const auto& points = traj_->get_integration_points();

    // Fast path: Check if current hint is still valid
    if (time_hint_ != points.end() && time_hint_->time <= t) {
        auto next_hint = time_hint_;
        ++next_hint;
        if (next_hint == points.end() || t < next_hint->time) {
            // Hint is still valid, O(1)
            update_path_cursor_position_(t);
            return *this;
        }
    }

    // Check forward by one (common in sequential forward sampling)
    if (time_hint_ != points.end()) {
        auto next_hint = time_hint_;
        ++next_hint;
        if (next_hint != points.end() && next_hint->time <= t) {
            auto next_next = next_hint;
            ++next_next;
            if (next_next == points.end() || t < next_next->time) {
                time_hint_ = next_hint;
                update_path_cursor_position_(t);
                return *this;
            }
        }
    }

    // Check backward by one (common in backward integration)
    if (time_hint_ != points.begin()) {
        auto prev_hint = time_hint_;
        --prev_hint;
        if (prev_hint->time <= t && t < time_hint_->time) {
            time_hint_ = prev_hint;
            update_path_cursor_position_(t);
            return *this;
        }
    }

    // Large jump - use binary search (O(log n))
    // Find first point with time > t (upper_bound)
    auto it =
        std::upper_bound(points.begin(), points.end(), t, [](seconds value, const integration_point& point) { return value < point.time; });

    // The point before upper_bound contains our time
    if (it == points.begin()) {
        // At or before first point
        time_hint_ = points.begin();
    } else {
        --it;
        time_hint_ = it;
    }

    update_path_cursor_position_(t);
    return *this;
}

trajectory::cursor& trajectory::cursor::seek_by(seconds dt) {
    return seek(time_ + dt);
}

// NOLINTNEXTLINE(readability-convert-member-functions-to-static): Must be non-static for consistent cursor API
std::default_sentinel_t trajectory::cursor::end() const noexcept {
    return std::default_sentinel;
}

bool operator==(const trajectory::cursor& c, std::default_sentinel_t) noexcept {
    return !std::isfinite(c.time_.count());
}

bool operator==(std::default_sentinel_t, const trajectory::cursor& c) noexcept {
    return !std::isfinite(c.time_.count());
}

}  // namespace viam::trajex::totg
