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
RESPONSES = ("vol_NH3", "M_add_kNm", "W_add", "GM", "C_TCO")

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


def evaluate_responses(table, tank_type):
    records = []

    for row in table.itertuples(index=False):
        model = ship_model(
            E_D=float(row.E_D),
            W_C=float(row.W_C),
            V=float(row.V),
            tank_type=tank_type,
        )

        record = {response_name: getattr(model, response_name) for response_name in RESPONSES}
        records.append(record)

    result = table.copy()
    result["tank_type"] = tank_type

    for response_name in RESPONSES:
        result[response_name] = [record[response_name] for record in records]

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


def fit_model(table, response_name, terms):
    design_matrix = build_design_matrix(table, terms)
    response = table[response_name].to_numpy()
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
        "r_squared": r_squared,
        "adjusted_r_squared": adjusted_r_squared,
        "rmse": rmse,
    }


def select_compact_model(table, response_name, improvement_tolerance=1e-10):
    current = fit_model(table, response_name, BASE_TERMS)
    remaining_terms = list(CANDIDATE_TERMS)

    while remaining_terms:
        trials = []

        for candidate in remaining_terms:
            trial = fit_model(table, response_name, current["terms"] + [candidate])
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

        if improvement <= improvement_tolerance:
            break

        current = best_trial
        remaining_terms.remove(candidate_name)

    return current


def format_equation(coefficients, term_names, precision=12, zero_tolerance=1e-9):
    intercept = 0.0 if abs(coefficients[0]) < zero_tolerance else coefficients[0]
    terms = [f"{intercept:.{precision}f}"]

    for coefficient, term_name in zip(coefficients[1:], term_names):
        if abs(coefficient) < zero_tolerance:
            continue
        sign = "+" if coefficient >= 0 else "-"
        terms.append(f"{sign} {abs(coefficient):.{precision}f}*{term_name}")

    return " ".join(terms)


def export_results(table, tank_type, design_label):
    base_name = f"{tank_type}_{design_label}".replace("-", "_")
    output_path = f"sample_points_{base_name}.csv"
    table.to_csv(output_path, index=False)
    return output_path


def print_final_models(tank_type, design_label, output_path, model_results):
    print(f"{tank_type} {design_label}:")
    print(f"Sampling points and response values written to {output_path}")

    for response_name in RESPONSES:
        fit_result = model_results[response_name]
        # print(
        #     f"{response_name}: "
        #     f"Adj R^2 = {fit_result['adjusted_r_squared']:.10f}, "
        #     f"RMSE = {fit_result['rmse']:.10f}"
        # )
        print(f"{response_name} = {format_equation(fit_result['coefficients'], fit_result['terms'])}")

    print()


def run_case(tank_type, design_label, coded_points, physical_points):
    table = build_sampling_table(coded_points, physical_points, design_label)
    table = evaluate_responses(table, tank_type=tank_type)
    output_path = export_results(table, tank_type, design_label)

    model_results = {
        response_name: select_compact_model(table, response_name)
        for response_name in RESPONSES
    }

    print_final_models(tank_type, design_label, output_path, model_results)


def main():
    on_deck_coded = generate_faced_ccd_points()
    on_deck_physical = coded_to_physical(on_deck_coded)

    in_hold_coded, in_hold_physical = generate_four_level_full_factorial_points()

    run_case("on-deck", "faced_ccd", on_deck_coded, on_deck_physical)
    run_case("in-hold", "full_factorial_4_level", in_hold_coded, in_hold_physical)


if __name__ == "__main__":
    main()
