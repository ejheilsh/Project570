import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from model_class import ship_model

# AI was used to help generalize and improve the functionality of this plotting script


E_D_VALUES = np.linspace(9e3, 48e3, 50)
W_C_VALUES = np.linspace(80000.0, 175000.0, 45)
SPEEDS = [10.0, 12.0, 14.0, 16.0]
# TANK_TYPE = "in-hold"
TANK_TYPE = "on-deck"
OUTPUT_NAME = "W_add"
OUTPUT_NAME = "C_TCO"
# OUTPUT_NAME = "vol_NH3"

OUTPUT_CONFIG = {
    "vol_NH3": {
        "zlabel": "Ammonia volume (m^3)",
        "title": "Ammonia volume surfaces vs endurance and cargo capacity",
        "filename": "vol_NH3_surfaces",
    },
    "M_add_kNm": {
        "zlabel": "Additional bending moment (kN-m)",
        "title": "Additional bending moment surfaces vs endurance and cargo capacity",
        "filename": "M_add_kNm_surfaces",
    },
    "W_add": {
        "zlabel": "Additional weight (tons)",
        "title": "Additional weight surfaces vs endurance and cargo capacity",
        "filename": "W_add_surfaces",
    },
    "GM": {
        "zlabel": "GM (m)",
        "title": "GM surfaces vs endurance and cargo capacity",
        "filename": "GM_surfaces",
    },
    "C_TCO": {
        "zlabel": "Additional TCO ($)",
        "title": "C_TCO surfaces vs endurance and cargo capacity",
        "filename": "C_TCO_surfaces",
    },
}


def evaluate_response_grid(speed, output_name):
    e_grid, w_grid = np.meshgrid(E_D_VALUES, W_C_VALUES)
    response_grid = np.zeros_like(e_grid, dtype=float)

    for i in range(w_grid.shape[0]):
        for j in range(e_grid.shape[1]):
            model = ship_model(
                E_D=float(e_grid[i, j]),
                W_C=float(w_grid[i, j]),
                V=speed,
                tank_type=TANK_TYPE,
            )
            response_grid[i, j] = getattr(model, output_name)

    return e_grid, w_grid, response_grid


def main():
    if OUTPUT_NAME not in OUTPUT_CONFIG:
        valid_outputs = ", ".join(OUTPUT_CONFIG)
        raise ValueError(f"Unsupported output '{OUTPUT_NAME}'. Choose from: {valid_outputs}")

    output_config = OUTPUT_CONFIG[OUTPUT_NAME]
    tank_slug = TANK_TYPE.replace("-", "_")

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    cmap = plt.get_cmap("viridis")
    colors = [cmap(value) for value in np.linspace(0.15, 0.85, len(SPEEDS))]
    legend_items = []

    for speed, color in zip(SPEEDS, colors):
        e_grid, w_grid, response_grid = evaluate_response_grid(speed, OUTPUT_NAME)
        ax.plot_surface(
            e_grid,
            w_grid,
            response_grid,
            color=color,
            alpha=0.65,
            linewidth=0,
            antialiased=True,
            shade=False,
        )
        legend_items.append(Line2D([0], [0], color=color, lw=4, label=f"V = {speed:.0f} kt"))

    ax.set_title(f"{TANK_TYPE.title()} {output_config['title']}")
    ax.set_xlabel("Endurance, E_D (nm)")
    ax.set_ylabel("Cargo deadweight, W_C (tons)")
    ax.set_zlabel(output_config["zlabel"])
    ax.view_init(elev=24, azim=-130)
    ax.legend(handles=legend_items, loc="upper left")

    fig.tight_layout()
    fig.savefig(f"{output_config['filename']}_{tank_slug}.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()


# want to do regression that is quadratic in E_D
# also need someone to do base regression
