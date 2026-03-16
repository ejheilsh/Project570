from itertools import product

import numpy as np
import pandas as pd

from model_class import ship_model


BOUNDS = {
    "E_D": (9000.0, 24000.0),
    "W_C": (151000.0, 175000.0),
    "V": (10.0, 16.0),
}

VARIABLES = ("E_D", "W_C", "V")
RESPONSE = "vol_NH3"

FULL_FACTORIAL_LEVELS = {
    "E_D": np.linspace(BOUNDS["E_D"][0], BOUNDS["E_D"][1], 4),
    "W_C": np.linspace(BOUNDS["W_C"][0], BOUNDS["W_C"][1], 4),
    "V": np.linspace(BOUNDS["V"][0], BOUNDS["V"][1], 4),
}

BASE_TERMS = ["E_D", "W_C", "V"]
CANDIDATE_TERMS = ["E_D*W_C", "E_D*V", "W_C*V", "E_D^2", "W_C^2", "V^2"]


def midpoint_and_half_range(name):
    lower, upper = BOUNDS[name]
    midpoint = 0.5 * (lower + upper)
    half_range = 0.5 * (upper - lower)
    return midpoint, half_range


def physical_to_coded(name, values):
    midpoint, half_range = midpoint_and_half_range(name)
    return (np.asarray(values, dtype=float) - midpoint) / half_range


def generate_faced_ccd_points():
    corners = np.array(list(product([-1.0, 1.0], repeat=len(VARIABLES))), dtype=float)
    center = np.zeros((1, len(VARIABLES)), dtype=float)

    faces = []
    for index in range(len(VARIABLES)):
        point = np.zeros(len(VARIABLES), dtype=float)
        point[index] = 1.0
        faces.append(point.copy())
        point[index] = -1.0
        faces.append(point.copy())

    return np.vstack([corners, center, np.array(faces, dtype=float)])


def generate_four_level_full_factorial_points():
    grids = [FULL_FACTORIAL_LEVELS[name] for name in VARIABLES]
    physical_points = np.array(list(product(*grids)), dtype=float)
    coded_points = np.column_stack(
        [physical_to_coded(name, physical_points[:, column]) for column, name in enumerate(VARIABLES)]
    )
    return coded_points, physical_points


def coded_to_physical(coded_points):
    physical_points = np.zeros_like(coded_points, dtype=float)

    for column, name in enumerate(VARIABLES):
        midpoint, half_range = midpoint_and_half_range(name)
        physical_points[:, column] = midpoint + half_range * coded_points[:, column]

    return physical_points


def classify_point(coded_row, design_name):
    if design_name == "faced_ccd":
        nonzero = np.count_nonzero(coded_row)
        if nonzero == 0:
            return "center"
        if nonzero == 1:
            return "axial"
        return "corner"

    return "factorial"


def build_sampling_table(coded_points, physical_points, design_name):
    records = []

    for run_id, (coded_row, physical_row) in enumerate(zip(coded_points, physical_points), start=1):
        record = {"run": run_id, "design": design_name, "point_type": classify_point(coded_row, design_name)}

        for column, name in enumerate(VARIABLES):
            record[f"{name}_coded"] = coded_row[column]
            record[name] = physical_row[column]

        records.append(record)

    return pd.DataFrame.from_records(records)


def evaluate_response(table, tank_type):
    response_values = []

    for row in table.itertuples(index=False):
        model = ship_model(
            E_D=float(row.E_D),
            W_C=float(row.W_C),
            V=float(row.V),
            tank_type=tank_type,
        )
        response_values.append(getattr(model, RESPONSE))

    result = table.copy()
    result["tank_type"] = tank_type
    result[RESPONSE] = response_values
    return result


def compute_term(table, term_name):
    if term_name == "E_D":
        return table["E_D"].to_numpy()
    if term_name == "W_C":
        return table["W_C"].to_numpy()
    if term_name == "V":
        return table["V"].to_numpy()
    if term_name == "E_D*W_C":
        return table["E_D"].to_numpy() * table["W_C"].to_numpy()
    if term_name == "E_D*V":
        return table["E_D"].to_numpy() * table["V"].to_numpy()
    if term_name == "W_C*V":
        return table["W_C"].to_numpy() * table["V"].to_numpy()
    if term_name == "E_D^2":
        return table["E_D"].to_numpy() ** 2
    if term_name == "W_C^2":
        return table["W_C"].to_numpy() ** 2
    if term_name == "V^2":
        return table["V"].to_numpy() ** 2

    raise ValueError(f"Unknown term: {term_name}")


def build_design_matrix(table, terms):
    columns = [np.ones(len(table), dtype=float)]
    columns.extend(compute_term(table, term_name) for term_name in terms)
    return np.column_stack(columns)


def fit_model(table, terms):
    design_matrix = build_design_matrix(table, terms)
    response = table[RESPONSE].to_numpy()
    coefficients, _, _, _ = np.linalg.lstsq(design_matrix, response, rcond=None)
    predicted = design_matrix @ coefficients
    residual = response - predicted

    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((response - np.mean(response)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot else 1.0

    n_samples = len(response)
    n_parameters = len(coefficients)
    if n_samples > n_parameters:
        adjusted_r_squared = 1.0 - (1.0 - r_squared) * (n_samples - 1.0) / (n_samples - n_parameters)
    else:
        adjusted_r_squared = np.nan

    rmse = float(np.sqrt(np.mean(residual**2)))

    return {
        "terms": list(terms),
        "coefficients": coefficients,
        "predicted": predicted,
        "residual": residual,
        "r_squared": r_squared,
        "adjusted_r_squared": adjusted_r_squared,
        "rmse": rmse,
    }


def select_compact_model(table, improvement_tolerance=1e-10):
    current = fit_model(table, BASE_TERMS)
    selection_log = []
    remaining_terms = list(CANDIDATE_TERMS)

    while remaining_terms:
        trials = []

        for candidate in remaining_terms:
            trial = fit_model(table, current["terms"] + [candidate])
            trials.append((candidate, trial))

        trials.sort(
            key=lambda item: (
                -item[1]["adjusted_r_squared"],
                item[1]["rmse"],
                len(item[1]["terms"]),
                item[0],
            )
        )

        candidate_name, best_trial = trials[0]
        improvement = best_trial["adjusted_r_squared"] - current["adjusted_r_squared"]
        selection_log.append(
            {
                "candidate": candidate_name,
                "adj_R2_current": current["adjusted_r_squared"],
                "adj_R2_trial": best_trial["adjusted_r_squared"],
                "delta_adj_R2": improvement,
                "rmse_trial": best_trial["rmse"],
                "selected": improvement > improvement_tolerance,
            }
        )

        if improvement <= improvement_tolerance:
            break

        current = best_trial
        remaining_terms.remove(candidate_name)

    return fit_model(table, BASE_TERMS), current, pd.DataFrame(selection_log)


def format_equation(coefficients, term_names, precision=12):
    terms = [f"{coefficients[0]:.{precision}f}"]

    for coefficient, term_name in zip(coefficients[1:], term_names):
        sign = "+" if coefficient >= 0 else "-"
        terms.append(f"{sign} {abs(coefficient):.{precision}f}*{term_name}")

    return " ".join(terms)


def print_fit_summary(model_name, fit_result):
    print(model_name)
    print(f"Terms: {', '.join(fit_result['terms'])}")
    print(f"R^2 = {fit_result['r_squared']:.10f}")
    print(f"Adjusted R^2 = {fit_result['adjusted_r_squared']:.10f}")
    print(f"RMSE = {fit_result['rmse']:.10f}")
    print(f"{RESPONSE} = {format_equation(fit_result['coefficients'], fit_result['terms'])}")
    print()


def print_selection_log(selection_log):
    print("Adjusted R^2 term-selection log:")
    if selection_log.empty:
        print("No candidate terms were tested.")
    else:
        print(selection_log.to_string(index=False, float_format=lambda value: f"{value:12.10f}"))
    print()


def export_results(table, tank_type, design_label):
    base_name = f"{RESPONSE}_{tank_type}_{design_label}".replace("-", "_")
    output_path = f"{base_name}_points.csv"
    table.to_csv(output_path, index=False)
    return output_path


def run_case(tank_type, design_label, coded_points, physical_points):
    table = build_sampling_table(coded_points, physical_points, design_label)
    table = evaluate_response(table, tank_type=tank_type)
    output_path = export_results(table, tank_type, design_label)

    simple_model, final_model, selection_log = select_compact_model(table)

    print(f"{tank_type} {design_label}:")
    print(f"Sampling points and {RESPONSE} values written to {output_path}")
    print(f"Rows: {len(table)}")
    print()
    print_fit_summary(f"{tank_type} simple linear model:", simple_model)
    print_selection_log(selection_log)
    print_fit_summary(f"{tank_type} final compact model:", final_model)


def main():
    on_deck_coded = generate_faced_ccd_points()
    on_deck_physical = coded_to_physical(on_deck_coded)

    in_hold_coded, in_hold_physical = generate_four_level_full_factorial_points()

    run_case("on-deck", "faced_ccd", on_deck_coded, on_deck_physical)
    run_case("in-hold", "full_factorial_4_level", in_hold_coded, in_hold_physical)


if __name__ == "__main__":
    main()
