from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import pyswarms as ps

from model_class import ship_model


VARIABLE_NAMES = ("E_D", "W_C", "V")
VARIABLE_BOUNDS = {
    "E_D": (9000.0, 48000.0),
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
    n_particles: int = 10000
    iters: int = 220
    c1: float = 1
    c2: float = 1.5
    w: float = 0.5
    velocity_clamp_fraction: float = 0.05
    penalty_multiplier: float = 1.0e8
    seed: int = 67


@dataclass(frozen=True)
class VisualizationConfig:
    output_dir: str = "."
    fps: int = 12
    interpolation_steps: int = 4


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


def objective_function(
    positions: np.ndarray,
    tank_type: str,
    constraints: ConstraintSet,
    penalty_multiplier: float,
) -> np.ndarray:
    costs = np.empty(positions.shape[0], dtype=float)

    for index, position in enumerate(positions):
        result = evaluate_candidate(position, tank_type=tank_type, constraints=constraints)
        costs[index] = result["C_TCO"] + penalty_multiplier * result["penalty"]

    return costs


def solve_case(
    tank_type: str,
    constraints: ConstraintSet | None = None,
    config: PsoConfig | None = None,
) -> tuple[dict[str, float], ps.single.GlobalBestPSO]:
    if constraints is None:
        constraints = baseline_constraints(tank_type)
    if config is None:
        config = PsoConfig()

    np.random.seed(config.seed)
    lower, upper = lower_upper_bounds()
    optimizer = ps.single.GlobalBestPSO(
        n_particles=config.n_particles,
        dimensions=len(VARIABLE_NAMES),
        options={"c1": config.c1, "c2": config.c2, "w": config.w},
        bounds=(lower, upper),
        bh_strategy="reflective",
        velocity_clamp=velocity_clamp(config),
    )
    best_cost, best_position = optimizer.optimize(
        objective_function,
        iters=config.iters,
        tank_type=tank_type,
        constraints=constraints,
        penalty_multiplier=config.penalty_multiplier,
        verbose=False,
    )

    best_result = evaluate_candidate(best_position, tank_type=tank_type, constraints=constraints)
    result = {
        "tank_type": tank_type,
        "best_cost": float(best_cost),
        "E_D": float(best_position[0]),
        "W_C": float(best_position[1]),
        "V": float(best_position[2]),
        **best_result,
    }
    return result, optimizer


def save_cost_history_plot(optimizer: ps.single.GlobalBestPSO, tank_type: str, output_dir: str) -> str:
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(optimizer.cost_history, linewidth=2.0, color="tab:blue")
    axis.set_title(f"{tank_type} PSO convergence")
    axis.set_xlabel("Iteration")
    axis.set_ylabel("Best penalized objective")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()

    output_path = f"{output_dir}/{tank_type.replace('-', '_')}_pso_convergence.png"
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    return output_path


def save_swarm_animation(
    optimizer: ps.single.GlobalBestPSO,
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
            f"Best penalized objective: ${dense_cost_history[frame_index]:,.0f}"
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


def create_visualizations(
    optimizer: ps.single.GlobalBestPSO,
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


def print_solution(result: dict[str, float]) -> None:
    print(f"{result['tank_type']} optimum")
    print(f"  E_D = {result['E_D']:.2f} nm")
    print(f"  W_C = {result['W_C']:.2f} tons")
    print(f"  V = {result['V']:.2f} kt")
    print(f"  Penalized objective = ${result['best_cost']:.2f}")
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
    # for tank_type in ("on-deck", "in-hold"):
    for tank_type in ("in-hold", "on-deck"):
        result, optimizer = solve_case(tank_type)
        print_solution(result)
        outputs = create_visualizations(optimizer, result)
        print(f"  convergence plot saved to {outputs['convergence_plot']}")
        print(f"  swarm animation saved to {outputs['swarm_animation']}")
        print()


if __name__ == "__main__":
    main()
