from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

from model_class import ship_model


VARIABLE_NAMES = ("E_D", "W_C", "V")
VARIABLE_BOUNDS = {
    "E_D": (9000.0, 24000.0),
    "W_C": (151000.0, 175000.0),
    "V": (10.0, 16.0),
}

SWBM_KNM = 4_000_000.0
DISPLACEMENT_TONS = 200_000.0
ON_DECK_VOLUME_LIMIT_M3 = 6000.0
IN_HOLD_VOLUME_LIMIT_M3 = 22.0 * 28.0 * 38.0


@dataclass(frozen=True)
class ConstraintSet:
    volume_limit_m3: float
    bending_moment_limit_knm: float = 0.30 * SWBM_KNM
    weight_limit_tons: float = 0.025 * DISPLACEMENT_TONS
    gm_min_m: float = 2.0
    gm_max_m: float = 8.0


@dataclass(frozen=True)
class PsoConfig:
    n_particles: int = 24
    iters: int = 220
    c1: float = 1.1
    c2: float = 1.1
    w: float = 0.45
    velocity_clamp_fraction: float = 0.04
    reflection_damping: float = 0.5
    repair_line_search_steps: int = 12
    initialization_tries: int = 20000
    seed: int = 42


@dataclass(frozen=True)
class VisualizationConfig:
    output_dir: str = "."
    fps: int = 12
    interpolation_steps: int = 4


@dataclass
class PsoRun:
    pos_history: list[np.ndarray]
    cost_history: list[float]


def baseline_constraints(tank_type: str) -> ConstraintSet:
    if tank_type == "on-deck":
        return ConstraintSet(volume_limit_m3=ON_DECK_VOLUME_LIMIT_M3)
    if tank_type == "in-hold":
        return ConstraintSet(volume_limit_m3=IN_HOLD_VOLUME_LIMIT_M3)
    raise ValueError(f"Unsupported tank type: {tank_type}")


def lower_upper_bounds():
    lower = np.array([VARIABLE_BOUNDS[name][0] for name in VARIABLE_NAMES], dtype=float)
    upper = np.array([VARIABLE_BOUNDS[name][1] for name in VARIABLE_NAMES], dtype=float)
    return lower, upper


def velocity_clamp(config: PsoConfig) -> tuple[np.ndarray, np.ndarray]:
    lower, upper = lower_upper_bounds()
    max_step = config.velocity_clamp_fraction * (upper - lower)
    return -max_step, max_step


def evaluate_candidate(position: np.ndarray, tank_type: str, constraints: ConstraintSet) -> dict[str, float]:
    model = ship_model(
        E_D=float(position[0]),
        W_C=float(position[1]),
        V=float(position[2]),
        tank_type=tank_type,
    )

    violations = model.constraint_violations(
        volume_limit_m3=constraints.volume_limit_m3,
        bending_moment_limit_knm=constraints.bending_moment_limit_knm,
        weight_limit_tons=constraints.weight_limit_tons,
        gm_min_m=constraints.gm_min_m,
        gm_max_m=constraints.gm_max_m,
    )
    total_penalty = sum(value * value for value in violations.values())

    return {
        "C_TCO": model.C_TCO,
        "vol_NH3": model.vol_NH3,
        "M_add_kNm": model.M_add_kNm,
        "W_add": model.W_add,
        "GM": model.GM,
        "feasible": model.is_feasible(
            volume_limit_m3=constraints.volume_limit_m3,
            bending_moment_limit_knm=constraints.bending_moment_limit_knm,
            weight_limit_tons=constraints.weight_limit_tons,
            gm_min_m=constraints.gm_min_m,
            gm_max_m=constraints.gm_max_m,
        ),
        "penalty": total_penalty,
        **{f"{name}_violation": value for name, value in violations.items()},
    }


def reflect_to_bounds(position: np.ndarray, velocity: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reflected_position = position.copy()
    reflected_velocity = velocity.copy()

    for dimension in range(len(reflected_position)):
        while reflected_position[dimension] < lower[dimension] or reflected_position[dimension] > upper[dimension]:
            if reflected_position[dimension] < lower[dimension]:
                reflected_position[dimension] = 2.0 * lower[dimension] - reflected_position[dimension]
                reflected_velocity[dimension] *= -1.0
            elif reflected_position[dimension] > upper[dimension]:
                reflected_position[dimension] = 2.0 * upper[dimension] - reflected_position[dimension]
                reflected_velocity[dimension] *= -1.0

    reflected_position = np.clip(reflected_position, lower, upper)
    return reflected_position, reflected_velocity


def repair_particle(
    previous_position: np.ndarray,
    proposed_position: np.ndarray,
    proposed_velocity: np.ndarray,
    tank_type: str,
    constraints: ConstraintSet,
    config: PsoConfig,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    candidate_position, candidate_velocity = reflect_to_bounds(proposed_position, proposed_velocity, lower, upper)
    candidate_result = evaluate_candidate(candidate_position, tank_type=tank_type, constraints=constraints)
    if candidate_result["feasible"]:
        return candidate_position, candidate_velocity, candidate_result

    reflected_velocity = -config.reflection_damping * candidate_velocity
    reflected_position = previous_position + reflected_velocity
    reflected_position, reflected_velocity = reflect_to_bounds(reflected_position, reflected_velocity, lower, upper)
    reflected_result = evaluate_candidate(reflected_position, tank_type=tank_type, constraints=constraints)
    if reflected_result["feasible"]:
        return reflected_position, reflected_velocity, reflected_result

    direction = reflected_position - previous_position
    best_position = previous_position.copy()
    best_velocity = np.zeros_like(proposed_velocity)
    best_result = evaluate_candidate(best_position, tank_type=tank_type, constraints=constraints)

    for alpha in np.linspace(0.5, 0.0, config.repair_line_search_steps, endpoint=False):
        trial_position = previous_position + alpha * direction
        trial_position, _ = reflect_to_bounds(trial_position, np.zeros_like(proposed_velocity), lower, upper)
        trial_result = evaluate_candidate(trial_position, tank_type=tank_type, constraints=constraints)
        if trial_result["feasible"]:
            return trial_position, alpha * reflected_velocity, trial_result

    return best_position, best_velocity, best_result


def sample_feasible_initial_positions(
    rng: np.random.Generator,
    n_particles: int,
    tank_type: str,
    constraints: ConstraintSet,
    config: PsoConfig,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    positions = []

    for _ in range(config.initialization_tries):
        if len(positions) == n_particles:
            break
        candidate = rng.uniform(lower, upper)
        result = evaluate_candidate(candidate, tank_type=tank_type, constraints=constraints)
        if result["feasible"]:
            positions.append(candidate)

    if len(positions) != n_particles:
        raise RuntimeError(
            f"Unable to initialize {n_particles} feasible particles for {tank_type} after "
            f"{config.initialization_tries} random samples."
        )

    return np.asarray(positions, dtype=float)


def solve_case(
    tank_type: str,
    constraints: ConstraintSet | None = None,
    config: PsoConfig | None = None,
) -> tuple[dict[str, float], PsoRun]:
    if constraints is None:
        constraints = baseline_constraints(tank_type)
    if config is None:
        config = PsoConfig()

    rng = np.random.default_rng(config.seed)
    lower, upper = lower_upper_bounds()
    vmin, vmax = velocity_clamp(config)

    positions = sample_feasible_initial_positions(
        rng=rng,
        n_particles=config.n_particles,
        tank_type=tank_type,
        constraints=constraints,
        config=config,
        lower=lower,
        upper=upper,
    )
    velocities = rng.uniform(vmin, vmax, size=positions.shape)

    personal_best_positions = positions.copy()
    personal_best_costs = np.empty(config.n_particles, dtype=float)
    personal_best_results: list[dict[str, float]] = []

    for index, position in enumerate(positions):
        result = evaluate_candidate(position, tank_type=tank_type, constraints=constraints)
        personal_best_costs[index] = result["C_TCO"]
        personal_best_results.append(result)

    best_index = int(np.argmin(personal_best_costs))
    global_best_position = personal_best_positions[best_index].copy()
    global_best_cost = float(personal_best_costs[best_index])
    global_best_result = personal_best_results[best_index]

    pos_history = [positions.copy()]
    cost_history = [global_best_cost]

    for _ in range(config.iters):
        random_cognitive = rng.random(size=positions.shape)
        random_social = rng.random(size=positions.shape)
        velocities = (
            config.w * velocities
            + config.c1 * random_cognitive * (personal_best_positions - positions)
            + config.c2 * random_social * (global_best_position - positions)
        )
        velocities = np.clip(velocities, vmin, vmax)

        proposed_positions = positions + velocities
        updated_positions = positions.copy()
        updated_velocities = velocities.copy()

        for index in range(config.n_particles):
            repaired_position, repaired_velocity, repaired_result = repair_particle(
                previous_position=positions[index],
                proposed_position=proposed_positions[index],
                proposed_velocity=velocities[index],
                tank_type=tank_type,
                constraints=constraints,
                config=config,
                lower=lower,
                upper=upper,
            )
            updated_positions[index] = repaired_position
            updated_velocities[index] = repaired_velocity

            cost = float(repaired_result["C_TCO"])
            if cost < personal_best_costs[index]:
                personal_best_costs[index] = cost
                personal_best_positions[index] = repaired_position
                personal_best_results[index] = repaired_result
                if cost < global_best_cost:
                    global_best_cost = cost
                    global_best_position = repaired_position.copy()
                    global_best_result = repaired_result

        positions = updated_positions
        velocities = updated_velocities
        pos_history.append(positions.copy())
        cost_history.append(global_best_cost)

    result = {
        "tank_type": tank_type,
        "best_cost": global_best_cost,
        "E_D": float(global_best_position[0]),
        "W_C": float(global_best_position[1]),
        "V": float(global_best_position[2]),
        **global_best_result,
    }
    optimizer = PsoRun(pos_history=pos_history, cost_history=cost_history)
    return result, optimizer


def save_cost_history_plot(optimizer: PsoRun, tank_type: str, output_dir: str) -> str:
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(optimizer.cost_history, linewidth=2.0, color="tab:blue")
    axis.set_title(f"{tank_type} PSO convergence")
    axis.set_xlabel("Iteration")
    axis.set_ylabel("Best feasible C_TCO")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()

    output_path = f"{output_dir}/{tank_type.replace('-', '_')}_pso_convergence.png"
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    return output_path


def save_swarm_animation(
    optimizer: PsoRun,
    result: dict[str, float],
    tank_type: str,
    output_dir: str,
    fps: int,
    interpolation_steps: int,
) -> str:
    pos_history = np.asarray(optimizer.pos_history, dtype=float)
    lower, upper = lower_upper_bounds()
    dense_history = interpolate_history(pos_history, interpolation_steps)
    dense_cost_history = interpolate_cost_history(optimizer.cost_history, interpolation_steps)

    figure, axis = plt.subplots(figsize=(9, 7))
    scatter = axis.scatter([], [], c=[], cmap="viridis", vmin=lower[2], vmax=upper[2], s=45, alpha=0.85)
    best_marker = axis.scatter([], [], marker="*", s=220, c="crimson", edgecolors="black", linewidths=0.8)
    colorbar = figure.colorbar(scatter, ax=axis)
    colorbar.set_label("Speed, V (kt)")

    axis.set_xlim(lower[0], upper[0])
    axis.set_ylim(lower[1], upper[1])
    axis.set_xlabel("Endurance, E_D (nm)")
    axis.set_ylabel("Cargo deadweight, W_C (tons)")
    axis.set_title(f"{tank_type} PSO swarm trajectory")
    cost_text = axis.text(0.02, 0.98, "", transform=axis.transAxes, va="top")

    def update(frame_index: int):
        positions = dense_history[frame_index]
        offsets = positions[:, :2]
        speeds = positions[:, 2]
        scatter.set_offsets(offsets)
        scatter.set_array(speeds)

        best_position = np.array([result["E_D"], result["W_C"]], dtype=float)
        best_marker.set_offsets(best_position[:2].reshape(1, 2))
        cost_text.set_text(
            f"Frame: {frame_index + 1}/{len(dense_history)}\n"
            f"Best feasible C_TCO: ${dense_cost_history[frame_index]:,.0f}"
        )
        return scatter, best_marker, cost_text

    anim = animation.FuncAnimation(
        figure,
        update,
        frames=len(dense_history),
        interval=max(1, int(1000 / fps)),
        blit=False,
        repeat=False,
    )

    output_path = f"{output_dir}/{tank_type.replace('-', '_')}_pso_swarm.gif"
    anim.save(output_path, writer=animation.PillowWriter(fps=fps))
    plt.close(figure)
    return output_path


def interpolate_history(pos_history: np.ndarray, interpolation_steps: int) -> np.ndarray:
    if interpolation_steps <= 1 or len(pos_history) < 2:
        return pos_history

    frames = []
    for index in range(len(pos_history) - 1):
        start = pos_history[index]
        end = pos_history[index + 1]
        for alpha in np.linspace(0.0, 1.0, interpolation_steps, endpoint=False):
            frames.append((1.0 - alpha) * start + alpha * end)
    frames.append(pos_history[-1])
    return np.asarray(frames, dtype=float)


def interpolate_cost_history(cost_history: list[float], interpolation_steps: int) -> np.ndarray:
    cost_array = np.asarray(cost_history, dtype=float)
    if interpolation_steps <= 1 or len(cost_array) < 2:
        return cost_array

    frames = []
    for index in range(len(cost_array) - 1):
        start = cost_array[index]
        end = cost_array[index + 1]
        for alpha in np.linspace(0.0, 1.0, interpolation_steps, endpoint=False):
            frames.append((1.0 - alpha) * start + alpha * end)
    frames.append(cost_array[-1])
    return np.asarray(frames, dtype=float)


def print_solution(result: dict[str, float]) -> None:
    print(f"{result['tank_type']} optimum")
    print(f"  E_D = {result['E_D']:.2f} nm")
    print(f"  W_C = {result['W_C']:.2f} tons")
    print(f"  V = {result['V']:.2f} kt")
    print(f"  Best feasible C_TCO = ${result['best_cost']:.2f}")
    print(f"  C_TCO = ${result['C_TCO']:.2f}")
    print(f"  vol_NH3 = {result['vol_NH3']:.2f} m^3")
    print(f"  M_add_kNm = {result['M_add_kNm']:.2f} kN-m")
    print(f"  W_add = {result['W_add']:.2f} tons")
    print(f"  GM = {result['GM']:.2f} m")
    print(f"  penalty = {result['penalty']:.8f}")
    print(f"  feasible = {result['feasible']}")
    print("  normalized violations")
    print(f"    volume = {result['volume_violation']:.8f}")
    print(f"    bending_moment = {result['bending_moment_violation']:.8f}")
    print(f"    weight = {result['weight_violation']:.8f}")
    print(f"    gm_min = {result['gm_min_violation']:.8f}")
    print(f"    gm_max = {result['gm_max_violation']:.8f}")
    print()


def main() -> None:
    for tank_type in ("on-deck", "in-hold"):
        result, optimizer = solve_case(tank_type)
        print_solution(result)
        outputs = create_visualizations(optimizer, result)
        print(f"  convergence plot saved to {outputs['convergence_plot']}")
        print(f"  swarm animation saved to {outputs['swarm_animation']}")
        print()


def create_visualizations(
    optimizer: PsoRun,
    result: dict[str, float],
    visualization: VisualizationConfig | None = None,
) -> dict[str, str]:
    if visualization is None:
        visualization = VisualizationConfig()

    tank_type = str(result["tank_type"])
    convergence_path = save_cost_history_plot(optimizer, tank_type, visualization.output_dir)
    animation_path = save_swarm_animation(
        optimizer,
        result,
        tank_type,
        visualization.output_dir,
        visualization.fps,
        visualization.interpolation_steps,
    )
    return {
        "convergence_plot": convergence_path,
        "swarm_animation": animation_path,
    }


if __name__ == "__main__":
    main()
