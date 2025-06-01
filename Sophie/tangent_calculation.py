#!/usr/bin/env python3
"""
Beregning af tangenter til g(x) = (x+5)(x-1)(x-10)(x-15) der går gennem (-3, 10)
"""

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from sympy import solve, symbols


def g(x):
    """Function g(x) = (x+5)(x-1)(x-10)(x-15)"""
    return (x + 5) * (x - 1) * (x - 10) * (x - 15)


def dg(x):
    """Derivative of g(x)"""
    term1 = (x - 1) * (x - 10) * (x - 15)  # derivative of (x+5)
    term2 = (x + 5) * (x - 10) * (x - 15)  # derivative of (x-1)
    term3 = (x + 5) * (x - 1) * (x - 15)  # derivative of (x-10)
    term4 = (x + 5) * (x - 1) * (x - 10)  # derivative of (x-15)
    return term1 + term2 + term3 + term4


def main():
    print("Find tangenter til g(x) = (x+5)(x-1)(x-10)(x-15) der går gennem (-3, 10)")
    print("=" * 80)

    # Define symbolic variable
    a = symbols("a")

    # Define g(a) and g'(a) symbolically
    g_a = (a + 5) * (a - 1) * (a - 10) * (a - 15)
    dg_a = sp.diff(g_a, a)

    # Set up equation: 10 = g(a) + g'(a)(-3 - a)
    # This means: 10 = g(a) - g'(a)(3 + a)
    equation = sp.Eq(10, g_a - dg_a * (3 + a))

    print("Ligning der skal løses:")
    print("10 = g(a) - g'(a)(3 + a)")

    # Solve the equation
    solutions = solve(equation, a)
    print(f"\nAntal løsninger: {len(solutions)}")

    # Filter real solutions
    real_solutions = [sol for sol in solutions if sol.is_real is True]
    print(f"Antal reelle løsninger: {len(real_solutions)}")

    # Convert to numerical values
    a_values = [float(sol.evalf()) for sol in real_solutions]
    print("\nNumeriske værdier:")
    for i, a_val in enumerate(a_values):
        print(f"a_{i + 1} = {a_val:.6f}")

    # Calculate tangent line equations
    print("\nTangentlinjer:")
    print("=" * 60)

    tangent_equations = []
    for i, a_val in enumerate(a_values):
        # Calculate g(a) and g'(a)
        g_val = g(a_val)
        dg_val = dg(a_val)

        print(f"\nTangent {i + 1}: Røringspunkt a = {a_val:.6f}")
        print(f"g({a_val:.6f}) = {g_val:.6f}")
        print(f"g'({a_val:.6f}) = {dg_val:.6f}")

        # Tangent line: y = g(a) + g'(a)(x - a)
        # Rewrite as: y = g'(a)*x + (g(a) - g'(a)*a)
        slope = dg_val
        intercept = g_val - dg_val * a_val

        print(f"Tangentlinje: y = {slope:.6f}(x - {a_val:.6f}) + {g_val:.6f}")
        print(f"Forenklet:    y = {slope:.6f}x + {intercept:.6f}")

        # Verify that the line passes through (-3, 10)
        y_at_minus3 = slope * (-3) + intercept
        print(f"Verificering: Ved x = -3: y = {y_at_minus3:.6f} (skal være 10)")

        tangent_equations.append((slope, intercept, a_val))

    print("\n" + "=" * 80)
    print("SVAR: Ligninger for tangenter til g(x) der går gennem (-3, 10):")
    print("=" * 80)
    for i, (slope, intercept, a_val) in enumerate(tangent_equations):
        print(
            f"Tangent {i + 1} (røringspunkt a = {a_val:.6f}): y = {slope:.6f}x + {intercept:.6f}"
        )

    # Create visualization
    create_plot(tangent_equations)

    return tangent_equations


def create_plot(tangent_equations):
    """Create a plot showing g(x), tangent lines, and the point (-3, 10)"""
    plt.figure(figsize=(15, 10))

    # Create x values for plotting
    x_plot = np.linspace(-8, 18, 1000)
    g_values = g(x_plot)

    # Plot g(x)
    plt.plot(
        x_plot, g_values, "b-", linewidth=2, label=r"$g(x) = (x+5)(x-1)(x-10)(x-15)$"
    )

    # Plot the point (-3, 10)
    plt.plot(-3, 10, "ro", markersize=10, label="Punkt (-3, 10)")

    # Plot tangent lines
    colors = ["green", "orange", "purple"]
    for i, (slope, intercept, a_val) in enumerate(tangent_equations):
        # Plot tangent point
        plt.plot(
            a_val,
            g(a_val),
            "o",
            color=colors[i],
            markersize=8,
            label=f"Røringspunkt {i + 1}: ({a_val:.2f}, {g(a_val):.1f})",
        )

        # Plot tangent line
        x_tangent = np.linspace(-8, 18, 100)
        y_tangent = slope * x_tangent + intercept
        plt.plot(
            x_tangent,
            y_tangent,
            "--",
            color=colors[i],
            linewidth=2,
            label=f"Tangent {i + 1}: y = {slope:.2f}x + {intercept:.1f}",
        )

    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color="k", linewidth=0.5)
    plt.axvline(x=0, color="k", linewidth=0.5)

    plt.xlabel("x", fontsize=12)
    plt.ylabel("y", fontsize=12)
    plt.title("Tangenter til g(x) der går gennem (-3, 10)", fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # Set reasonable axis limits
    plt.xlim(-8, 18)
    plt.ylim(-5000, 5000)

    plt.tight_layout()
    plt.savefig("tangent_lines_through_point.pdf", dpi=600, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    tangent_equations = main()
