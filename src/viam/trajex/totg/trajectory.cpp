#include <viam/trajex/totg/trajectory.hpp>

#include <cassert>
#include <functional>
#include <ranges>
#include <stdexcept>

#include <boost/range/adaptors.hpp>

#include <viam/trajex/totg/path.hpp>
#include <viam/trajex/types/arc_length.hpp>

namespace viam::trajex::totg {

namespace {

// Computes maximum path velocity (s_dot) from joint velocity and acceleration limits.
// Two constraints apply: centripetal acceleration from path curvature (eq 31) and
// direct velocity limits (eq 36). Returns both so caller can take the minimum.
// See Kunz & Stilman equations 31 and 36.
[[gnu::pure]] auto compute_velocity_limits(const xt::xarray<double>& q_prime,
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
[[gnu::pure]] auto compute_acceleration_bounds(const xt::xarray<double>& q_prime,
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

// Computes the derivative of the velocity limit curve in the phase plane.
// This is d/ds s_dot_max_vel(s), which tells us the slope of the velocity limit curve.
// Used in Algorithm Step 3 to determine if we can leave the curve or must search for switching points.
// See Kunz & Stilman equation 37.
[[gnu::pure]] double compute_velocity_limit_derivative(const xt::xarray<double>& q_prime,
                                                       const xt::xarray<double>& q_double_prime,
                                                       const xt::xarray<double>& q_dot_max,
                                                       double epsilon) {
    // Find which joint is the limiting constraint (has minimum q_dot_max / |q'|)
    double min_limit = std::numeric_limits<double>::infinity();
    size_t limiting_joint = 0;

    for (size_t i = 0; i < q_prime.size(); ++i) {
        if (std::abs(q_prime(i)) < epsilon) {
            continue;
        }

        const double limit = q_dot_max(i) / std::abs(q_prime(i));
        if (limit < min_limit) {
            min_limit = limit;
            limiting_joint = i;
        }
    }

    // Compute derivative for the limiting joint (equation 37)
    // d/ds s_dot_max_vel = -(q_dot_max_i * q''_i) / (q'_i * |q'_i|)
    const double numerator = -q_dot_max(limiting_joint) * q_double_prime(limiting_joint);
    const double denominator = q_prime(limiting_joint) * std::abs(q_prime(limiting_joint));

    // TODO: This check can trigger even when limiting_joint was validly selected, because a joint
    // with |q'| slightly above epsilon (e.g., 1.1*epsilon) produces a denominator of |q'|^2 which
    // can be below epsilon (e.g., 1.21*epsilon^2 << epsilon for small epsilon). This represents a
    // numerically singular case where the joint is barely moving along the path (q' ≈ 0) but happens
    // to be the limiting constraint. The derivative d/ds(1/|q'|) becomes huge and ill-defined.
    //
    // Physical interpretation: When |q'| ≈ 0, the velocity limit s_dot_max = q_dot_max / |q'| is
    // enormous, meaning this joint isn't really constraining velocity in any meaningful sense. The
    // fact that it's the "limiting joint" may be an artifact rather than a real constraint.
    //
    // This needs investigation:
    // 1. Check reference implementation (src/third_party/trajectories/) for how it handles this case
    // 2. Consider whether limiting joint selection should filter out near-singular joints
    // 3. Determine correct behavior: throw, return infinity with sign, or use different epsilon threshold
    // 4. May need tighter check: |denominator| < epsilon^2 to match the q' filter tolerance
    //
    // For now, throw to make this case visible during testing rather than silently returning an
    // incorrect value that could cause incorrect curve-following behavior.
    if (std::abs(denominator) < epsilon) {
        throw std::runtime_error{
            "compute_velocity_limit_derivative: denominator near zero for limiting joint - "
            "velocity limit curve derivative is numerically undefined (joint barely moving along path)"};
    }

    return numerator / denominator;
}

// Performs a single Euler integration step in phase plane (s, s_dot).
// Given current position, velocity, and applied acceleration, computes the next state.
// Uses constant acceleration kinematic equations: v_new = v + a*dt, s_new = s + v*dt + 0.5*a*dt^2.
// This is direction-agnostic - caller determines whether dt is positive (forward) or negative (backward).
[[gnu::const]] auto euler_step(arc_length s, double s_dot, double s_ddot, double dt) {
    struct result {
        arc_length s;
        double s_dot;
    };

    const double s_dot_new = s_dot + (s_ddot * dt);
    const arc_length s_new = s + arc_length{(s_dot * dt) + (0.5 * s_ddot * dt * dt)};

    return result{s_new, s_dot_new};
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

trajectory trajectory::create(class path p, options opt, integration_points points) {
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

    if (points.empty()) {
        // Production path: run the TOTG algorithm to compute time-optimal parameterization.
        // Algorithm from Kunz & Stilman Section VI: phase plane (s, ṡ) integration.

        // Create path cursor for querying geometry during integration. This cursor will be
        // advanced by the integration functions as they move along the path.
        path::cursor path_cursor = p.create_cursor();

        // Start trajectory at rest at the beginning of the path
        points.push_back({.time = seconds{0.0}, .s = path_cursor.position(), .s_dot = 0.0, .s_ddot = 0.0});

        // Continue until the trajectory reaches the end of the path. The loop maintains an invariant:
        // points.back() is a valid integration point, and we need to extend the trajectory from there.
        while (points.back().s < p.length()) {
            // Starting from our current position, integrate forward applying maximum acceleration
            // to build the fastest possible trajectory from here. We're being optimistic - assuming
            // we can continue accelerating indefinitely. This will either get us to the end of the
            // path, or we'll hit a velocity or acceleration limit curve that forces us to stop.
            while (true) {
                auto next_point = points.back();

                // Check if we've hit either the velocity or acceleration limit curve at this position.
                // These curves represent the boundary of what's kinematically feasible given the joint
                // constraints. If we're at or above this boundary, we can't accelerate further without
                // violating joint limits.
                const auto q_prime = path_cursor.tangent();
                const auto q_double_prime = path_cursor.curvature();

                const auto [s_dot_max_acc, s_dot_max_vel] =
                    compute_velocity_limits(q_prime, q_double_prime, opt.max_velocity, opt.max_acceleration, opt.epsilon);
                const auto s_dot_limit = std::min(s_dot_max_acc, s_dot_max_vel);

                if (s_dot_limit <= 0.0) [[unlikely]] {
                    throw std::runtime_error{"TOTG algorithm error: velocity limit curve is non-positive"};
                }

                if ((s_dot_limit - next_point.s_dot) < opt.epsilon) {
                    // NOTE: This implementation of Algorithm Step 3 follows the paper's description
                    // directly, but we have not fully verified it matches the reference implementation's
                    // approach. The reference code is complex and uses velocity clamping with bisection
                    // that may be functionally equivalent but structurally different. This warrants
                    // additional testing and potentially comparison with the reference on edge cases.

                    // We've hit either the velocity limit curve (eq 36) or acceleration limit curve (eq 31).
                    // Determine which curve we hit and whether we can handle it locally or need switching
                    // point search (Algorithm Step 4, Section VII).
                    const bool hit_velocity_curve = (s_dot_max_vel < s_dot_max_acc - opt.epsilon);

                    // Decide how to proceed based on which limit curve we hit and the feasible accelerations
                    enum class limit_curve_action : std::uint8_t { k_escape, k_follow_curve, k_search_for_switching_point };
                    limit_curve_action action;

                    if (hit_velocity_curve) {
                        // Algorithm Step 3: Analyze velocity limit curve s_dot_max_vel(s).
                        // Compute the curve's slope to determine exit conditions.
                        const double curve_slope =
                            compute_velocity_limit_derivative(q_prime, q_double_prime, opt.max_velocity, opt.epsilon);

                        // Get acceleration bounds at current position on the velocity limit curve
                        const auto [s_ddot_min, s_ddot_max] =
                            compute_acceleration_bounds(q_prime, q_double_prime, s_dot_max_vel, opt.max_acceleration, opt.epsilon);

                        // Determine which of three possible outcomes applies:
                        if ((s_ddot_max / next_point.s_dot) < curve_slope - opt.epsilon) {
                            // Exit condition 1: Maximum acceleration at the velocity limit would take us below
                            // the curve. We can escape back to normal maximum acceleration integration.
                            action = limit_curve_action::k_escape;
                        } else if ((s_ddot_min / next_point.s_dot) > curve_slope + opt.epsilon) {
                            // Exit condition 2: Minimum acceleration would still keep us above the curve.
                            // We're stuck on the limit curve and must search for the next switching point.
                            action = limit_curve_action::k_search_for_switching_point;
                        } else {
                            // Neither exit condition met - we can stay on the velocity limit curve by using
                            // constrained acceleration that matches the curve's slope.
                            action = limit_curve_action::k_follow_curve;
                        }
                    } else {
                        // Hit acceleration limit curve (from centripetal constraints).
                        // Unlike velocity curves, we cannot "follow" an acceleration curve by adjusting s_ddot,
                        // since the curve itself represents the limit on s_ddot given current s_dot. We must
                        // search for the next switching point where the constraint changes.
                        action = limit_curve_action::k_search_for_switching_point;
                    }

                    // Execute the determined action
                    switch (action) {
                        case limit_curve_action::k_search_for_switching_point: {
                            // TODO: Implement Algorithm Step 4: Switching point detection (Section VII)
                            // For velocity limit: Search along curve for point where we can escape (Section VII-B, eq 40-42)
                            // For acceleration limit: Find curvature discontinuity or q'_i = 0 point (Section VII-A, eq 38-39)
                            throw std::runtime_error{"Must search for switching point - not yet implemented"};
                        }

                        case limit_curve_action::k_follow_curve: {
                            // Follow velocity limit curve by using curve slope as acceleration.
                            // s_ddot = (d/ds s_dot_max_vel) * s_dot keeps s_dot equal to s_dot_max_vel(s).
                            const double curve_slope =
                                compute_velocity_limit_derivative(q_prime, q_double_prime, opt.max_velocity, opt.epsilon);
                            const auto [s_ddot_min, s_ddot_max] =
                                compute_acceleration_bounds(q_prime, q_double_prime, s_dot_max_vel, opt.max_acceleration, opt.epsilon);

                            const double s_ddot_curve = curve_slope * next_point.s_dot;

                            // Verify this acceleration is within feasible bounds
                            if (s_ddot_curve < s_ddot_min || s_ddot_curve > s_ddot_max) [[unlikely]] {
                                throw std::runtime_error{"TOTG algorithm error: curve-following acceleration outside feasible bounds"};
                            }

                            // Integrate along the velocity limit curve
                            const auto [next_s, next_s_dot] = euler_step(next_point.s, next_point.s_dot, s_ddot_curve, opt.delta.count());

                            // Check if we've reached the end while on the curve
                            if (next_s >= p.length()) {
                                points.push_back({.time = next_point.time + opt.delta, .s = p.length(), .s_dot = 0.0, .s_ddot = 0.0});
                                path_cursor.seek(p.length());
                                break;
                            }

                            // Record this point and continue forward integration (will re-evaluate on next iteration)
                            next_point.time += opt.delta;
                            next_point.s = next_s;
                            next_point.s_dot = next_s_dot;
                            next_point.s_ddot = s_ddot_curve;
                            points.push_back(next_point);
                            path_cursor.seek(next_s);
                            continue;  // Next iteration will check if still on curve or can exit
                        }

                        case limit_curve_action::k_escape:
                            // Fall through to normal maximum acceleration integration below.
                            // We'll naturally drop below the velocity limit on the next iteration.
                            break;
                    }
                }

                // We're below the limit curve, so compute maximum acceleration and take an integration step.
                // The acceleration bounds tell us how much path acceleration we can apply while respecting
                // all joint acceleration limits given our current path velocity.
                const auto [s_ddot_min, s_ddot_max] =
                    compute_acceleration_bounds(q_prime, q_double_prime, next_point.s_dot, opt.max_acceleration, opt.epsilon);

                if (s_ddot_min > s_ddot_max) [[unlikely]] {
                    throw std::runtime_error{"TOTG algorithm error: acceleration bounds are infeasible"};
                }

                const auto [next_s, next_s_dot] = euler_step(next_point.s, next_point.s_dot, s_ddot_max, opt.delta.count());

                // Forward integration should move "up and to the right" in phase plane: s increasing, s_dot increasing
                if ((next_s <= next_point.s) || (next_s_dot < next_point.s_dot)) [[unlikely]] {
                    throw std::runtime_error{"TOTG algorithm error: forward integration must increase both s and s_dot"};
                }

                next_point.time += opt.delta;
                next_point.s_ddot = s_ddot_max;

                // Check if this step would take us past the end of the path. The end at rest is the
                // terminal switching point - push it and exit forward integration.
                if (next_s >= p.length()) {
                    points.push_back({.time = next_point.time, .s = p.length(), .s_dot = 0.0, .s_ddot = 0.0});
                    path_cursor.seek(p.length());
                    break;
                }

                next_point.s = next_s;
                next_point.s_dot = next_s_dot;
                points.push_back(next_point);
                path_cursor.seek(next_s);
            }

            // Verify loop invariant: cursor positioned at the switching point we just pushed
            assert(path_cursor.position() == points.back().s);

            // We've found and pushed a switching point at points.back(). Integrate backward from it
            // to correct the over-optimistic forward trajectory. Starting from rest (or low velocity),
            // minimum acceleration is negative, causing s_dot to become negative and s to decrease.
            // We build a backward trajectory moving from the switching point back toward the start,
            // stopping when we hit the start or intersect the forward trajectory.
            std::vector<integration_point> backward_trajectory;
            backward_trajectory.push_back(points.back());

            while (backward_trajectory.back().s > arc_length{0.0}) {
                auto next_point = backward_trajectory.back();

                // Query path geometry at current position for limit curve and acceleration bounds
                const auto q_prime = path_cursor.tangent();
                const auto q_double_prime = path_cursor.curvature();

                // Check if we've hit a velocity or acceleration limit curve during backward integration.
                // This indicates the trajectory is infeasible - we cannot decelerate from the switching
                // point without violating joint constraints. This is an algorithm failure condition.
                const auto [s_dot_max_acc, s_dot_max_vel] =
                    compute_velocity_limits(q_prime, q_double_prime, opt.max_velocity, opt.max_acceleration, opt.epsilon);
                const auto s_dot_limit = std::min(s_dot_max_acc, s_dot_max_vel);

                if (s_dot_limit <= 0.0) [[unlikely]] {
                    throw std::runtime_error{"TOTG algorithm error: velocity limit curve is non-positive during backward integration"};
                }

                if ((s_dot_limit - next_point.s_dot) < opt.epsilon) {
                    throw std::runtime_error{"TOTG algorithm error: backward integration hit limit curve - trajectory is infeasible"};
                }

                // Compute minimum acceleration (should be negative to produce backward motion)
                const auto [s_ddot_min, s_ddot_max] =
                    compute_acceleration_bounds(q_prime, q_double_prime, next_point.s_dot, opt.max_acceleration, opt.epsilon);

                // Validate that minimum acceleration is negative (required for backward motion)
                if (s_ddot_min >= 0.0) [[unlikely]] {
                    throw std::runtime_error{"TOTG algorithm error: backward integration requires negative minimum acceleration"};
                }

                // Integrate one step backward in time (negative dt) with minimum acceleration.
                // Negative dt reverses time direction, so we reconstruct what velocities led to this point.
                // With s_ddot_min < 0 and dt < 0, s_dot increases (up) while s decreases (left).
                const auto [next_s, next_s_dot] = euler_step(next_point.s, next_point.s_dot, s_ddot_min, -opt.delta.count());

                // Backward integration should move "up and to the left" in phase plane: s decreasing, s_dot increasing
                if ((next_s >= next_point.s) || (next_s_dot <= next_point.s_dot)) [[unlikely]] {
                    throw std::runtime_error{"TOTG algorithm error: backward integration must decrease s and increase s_dot"};
                }

                next_point.time += opt.delta;  // Placeholder time, will recalculate when combining trajectories
                next_point.s_ddot = s_ddot_min;

                // Check if we've reached the start of the path
                if (next_s <= arc_length{0.0}) {
                    backward_trajectory.push_back({.time = next_point.time, .s = arc_length{0.0}, .s_dot = 0.0, .s_ddot = 0.0});
                    path_cursor.seek(arc_length{0.0});
                    break;
                }

                // TODO: Check for intersection with forward trajectory in (s, s_dot) space
                // Find where backward trajectory crosses below forward trajectory - that's the new switching point

                // Record this backward point and move cursor
                next_point.s = next_s;
                next_point.s_dot = next_s_dot;
                backward_trajectory.push_back(next_point);
                path_cursor.seek(next_s);
            }

            // Combine forward and backward trajectories at their intersection point.
            // The backward trajectory has lower velocities, representing the constraint from
            // needing to decelerate to rest at the switching point.

            // TODO: Find intersection in forward trajectory (where backward crosses below in s, s_dot space)
            const auto intersection = points.end();

            // Truncate forward trajectory at intersection
            points.erase(std::next(intersection), points.end());
            seconds intersection_time = points.back().time;

            // Reserve space to avoid reallocations during bulk append
            points.reserve(points.size() + backward_trajectory.size());

            // Bulk append reversed backward trajectory with corrected times
            for (const auto& indexed : backward_trajectory | boost::adaptors::reversed | boost::adaptors::indexed(1)) {
                auto corrected = indexed.value();
                corrected.time = intersection_time + seconds{static_cast<double>(indexed.index()) * opt.delta.count()};
                points.push_back(corrected);
            }
        }
    } else {
        // Test path: validate provided integration points
        // First point must be at t=0
        if (points.front().time != seconds{0.0}) {
            throw std::invalid_argument{"First integration point must have time == 0"};
        }

        // Pairwise validation using C++20 ranges
        auto pairs = std::views::iota(size_t{0}, points.size() - 1) |
                     std::views::transform([&](size_t i) { return std::pair{std::cref(points[i]), std::cref(points[i + 1])}; });

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

        if (points.back().s < arc_length{0.0} || points.back().s > p.length()) {
            throw std::invalid_argument{"Integration point arc lengths must be in [0, path.length()]"};
        }
    }

    // Construct trajectory with integration points
    // Duration is set automatically in constructor from last integration point
    return trajectory{std::move(p), std::move(opt), std::move(points)};
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
