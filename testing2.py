import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from model_class import ship_model


E_D_VALUES = np.linspace(9e3, 24e3, 50)
W_C_VALUES = np.linspace(80000.0, 175000.0, 45)
SPEEDS = [10.0, 12.0, 14.0, 16.0]
# TANK_TYPE = "in-hold"
TANK_TYPE = "on-deck"


def evaluate_cost_grid(speed):
    e_grid, w_grid = np.meshgrid(E_D_VALUES, W_C_VALUES)
    cost_grid = np.zeros_like(e_grid, dtype=float)

    for i in range(w_grid.shape[0]):
        for j in range(e_grid.shape[1]):
            model = ship_model(
                E_D=float(e_grid[i, j]),
                W_C=float(w_grid[i, j]),
                V=speed,
                tank_type=TANK_TYPE,
            )
            cost_grid[i, j] = model.C_TCO

    return e_grid, w_grid, cost_grid


def main():
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    cmap = plt.get_cmap("viridis")
    colors = [cmap(value) for value in np.linspace(0.15, 0.85, len(SPEEDS))]
    legend_items = []

    for speed, color in zip(SPEEDS, colors):
        e_grid, w_grid, cost_grid = evaluate_cost_grid(speed)
        ax.plot_surface(
            e_grid,
            w_grid,
            cost_grid,
            color=color,
            alpha=0.65,
            linewidth=0,
            antialiased=True,
            shade=False,
        )
        legend_items.append(Line2D([0], [0], color=color, lw=4, label=f"V = {speed:.0f} kt"))

    ax.set_title("In-hold C_TCO surfaces vs endurance and cargo capacity")
    ax.set_xlabel("Endurance, E_D (nm)")
    ax.set_ylabel("Cargo deadweight, W_C (tons)")
    ax.set_zlabel("Additional TCO ($)")
    ax.view_init(elev=24, azim=-130)
    ax.legend(handles=legend_items, loc="upper left")

    fig.tight_layout()
    fig.savefig("C_TCO_surfaces_in_hold.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()


# want to do regression that is quadratic in E_D
# also need someone to do base regression

