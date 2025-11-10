# Peaceable Armies of Knights — CP-SAT Only
# Finds the maximum equal per-army k via feasibility (no formulas used).
# Requirements: ortools, matplotlib, (optional) pandas for CSV export

# new_file.py
# Peaceable Knights — Streamlit UI driven by CP-SAT (no formulas)
# Run with: streamlit run new_file.py

import itertools
import io
import time
from typing import Optional, Tuple, List

import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from ortools.sat.python import cp_model # type: ignore
# from ortools.sat.python import cp_model


# -------------------- CP-SAT core --------------------

def solve_knights_equal_k(N: int, C: int, K: int,
                          time_limit: float = 10.0,
                          workers: int = 4,
                          seed: int = 0) -> Tuple[Optional[List[List[int]]], str, float]:
    """Return (board, status_name, runtime) for equal-army size K, else (None, status, runtime)."""
    model = cp_model.CpModel()
    moves = [(2,1),(1,2),(-1,2),(-2,1),(-2,-1),(-1,-2),(1,-2),(2,-1)]

    # x[i,j,c] ∈ {0,1} if color c occupies square (i,j)
    x = {(i, j, c): model.NewBoolVar(f"x_{i}_{j}_{c}")
         for i in range(N) for j in range(N) for c in range(C)}

    # (1) ≤ 1 knight per square
    for i in range(N):
        for j in range(N):
            model.Add(sum(x[(i, j, c)] for c in range(C)) <= 1)

    # (2) Peaceable constraint: forbid cross-color attacks on knight edges
    for i in range(N):
        for j in range(N):
            for di, dj in moves:
                ni, nj = i + di, j + dj
                if 0 <= ni < N and 0 <= nj < N:
                    for a in range(C):
                        # if (i,j) has color a, then (ni,nj) cannot have any b!=a
                        model.Add(
                            x[(i, j, a)] +
                            sum(x[(ni, nj, b)] for b in range(C) if b != a)
                            <= 1
                        )

    # (3) Equal armies: exactly K per color
    for c in range(C):
        model.Add(sum(x[(i, j, c)] for i in range(N) for j in range(N)) == K)

    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = workers
    solver.parameters.random_seed = seed
    solver.parameters.cp_model_presolve = True
    solver.parameters.log_search_progress = False

    start = time.time()
    status = solver.Solve(model)
    runtime = time.time() - start

    status_map = {
        cp_model.OPTIMAL: "OPTIMAL",
        cp_model.FEASIBLE: "FEASIBLE",
        cp_model.INFEASIBLE: "INFEASIBLE",
        cp_model.MODEL_INVALID: "MODEL_INVALID",
        cp_model.UNKNOWN: "UNKNOWN",
    }
    status_name = status_map.get(status, "UNKNOWN")

    if status not in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        return None, status_name, runtime

    # Extract assignment to a board
    board = [[-1] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            for c in range(C):
                if solver.Value(x[(i, j, c)]) == 1:
                    board[i][j] = c
    return board, status_name, runtime


def max_equal_k_by_cpsat(N: int, C: int,
                         time_limit: float = 10.0,
                         workers: int = 4,
                         seed: int = 0) -> Tuple[int, Optional[List[List[int]]], float]:
    """Binary search for the largest equal per-army k using feasibility only."""
    # Safe solver-agnostic upper cap by area
    ub = (N * N) // C
    lo, hi = 0, ub
    best_k, best_board = 0, None
    total_t = 0.0

    while lo <= hi:
        mid = (lo + hi) // 2
        board, status, rt = solve_knights_equal_k(N, C, mid, time_limit=time_limit,
                                                  workers=workers, seed=seed)
        total_t += rt
        if status in ("FEASIBLE", "OPTIMAL"):
            best_k, best_board = mid, board
            lo = mid + 1
        else:
            hi = mid - 1

    return best_k, best_board, total_t


# -------------------- Rendering --------------------
def render_board(board: List[List[int]], title: str = "") -> plt.Figure:
    """Return a Matplotlib Figure rendering the board (no plt.show())."""
    N = len(board)
    fig, ax = plt.subplots(figsize=(6, 6))

    # background checkerboard with custom colors
    light_sq = "#f0d9b5"
    dark_sq  = "#b58863"

    bg = [[(i + j) % 2 for j in range(N)] for i in range(N)]
    cmap = mcolors.ListedColormap([light_sq, dark_sq])
    ax.imshow(bg, cmap=cmap, vmin=0, vmax=1)

    for k in range(N + 1):
        ax.axhline(k - 0.5, color='k', linewidth=0.6)
        ax.axvline(k - 0.5, color='k', linewidth=0.6)

# def render_board(board: List[List[int]], title: str = "") -> plt.Figure:
#     """Return a Matplotlib Figure rendering the board (no plt.show())."""
#     N = len(board)
#     fig, ax = plt.subplots(figsize=(6, 6))
#     # background checkerboard
#     bg = [[(i + j) % 2 for j in range(N)] for i in range(N)]
#     ax.imshow(bg, cmap="binary", vmin=0, vmax=1)
#     for k in range(N + 1):
#         ax.axhline(k - 0.5, color='k', linewidth=0.6)
#         ax.axvline(k - 0.5, color='k', linewidth=0.6)

    # palette
    armies_present = sorted(set(v for row in board for v in row if v >= 0))
    C = len(armies_present)
    palette = list(mcolors.TABLEAU_COLORS.values())
    if C > len(palette):
        palette = list(itertools.islice(itertools.cycle(palette), C))
    color_map = {a: palette[i] for i, a in enumerate(armies_present)}

    # draw knights
    for i in range(N):
        for j in range(N):
            a = board[i][j]
            if a >= 0:
                ax.scatter(j, i, s=400, c=color_map[a], edgecolors='black', linewidths=1.2)
                ax.text(j, i, "♞", ha="center", va="center", fontsize=16, color="white")

    ax.invert_yaxis()
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title)
    fig.tight_layout()
    return fig


# -------------------- Streamlit UI --------------------

st.set_page_config(page_title="Peaceable Knights — CP-SAT", layout="wide")
st.title("Peaceable Armies of Knights — CP-SAT Model")

left, right = st.columns([1, 2], gap="large")

with left:
    st.subheader("Inputs")
    N = st.number_input("Board size (nxn)", min_value=1, max_value=40, value=8, step=1)
    C = st.number_input("Number of Armies (c)", min_value=1, max_value=20, value=3, step=1)

    st.caption("Solver settings")
    time_limit = st.slider("Time limit per feasibility check (seconds)", 1, 60, 10, step=1)
    workers = st.slider("Parallel workers", 1, 8, 4, step=1)
    seed = st.number_input("Random seed", min_value=0, max_value=10_000, value=1, step=1)

    run = st.button("Compute maximum equal k using CP-SAT", type="primary", use_container_width=True)

    st.markdown("""
    **What you’ll get:**  
    • Largest equal per-army k (solver-certified)  
    • A concrete board layout achieving that k (if feasible)  
    • Timing information
    """)

with right:
    st.subheader("Result")
    if run:
        with st.spinner("Solving…"):
            k_star, board, total_t = max_equal_k_by_cpsat(
                N=int(N), C=int(C),
                time_limit=float(time_limit),
                workers=int(workers),
                seed=int(seed),
            )
        # st.success(f"CP-SAT Solver: Best known lower bound: **k* = {k_star}** (N={N}, C={C})")
        st.success(f"CP-SAT Solver: Best known lower bound Kn({c}, {n}) = {k_star}")

        st.caption(f"Total solver time (binary search over k): {total_t:.2f} s")

        if board is not None:
            fig = render_board(board, title=f"N={N}, C={C}, k*={k_star}")
            st.pyplot(fig, clear_figure=True)

            # download PNG
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
            st.download_button(
                "Download board as PNG",
                data=buf.getvalue(),
                file_name=f"knights_N{N}_C{C}_k{k_star}.png",
                mime="image/png"
            )
        else:
            st.warning("Solver could not produce a board (unexpected).")

st.divider()
st.subheader("Batch sweep (optional)")

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    N_min = st.number_input("N min", 1, 40, 4, step=1, key="Nmin")
with c2:
    N_max = st.number_input("N max", 1, 40, 10, step=1, key="Nmax")
with c3:
    C_min = st.number_input("C min", 1, 20, 2, step=1, key="Cmin")
with c4:
    C_max = st.number_input("C max", 1, 20, 6, step=1, key="Cmax")
with c5:
    go_sweep = st.button("Run sweep and show table", key="sweep")

if go_sweep:
    rows = []
    prog = st.progress(0.0)
    total = (int(N_max) - int(N_min) + 1) * (int(C_max) - int(C_min) + 1)
    done = 0
    for NN in range(int(N_min), int(N_max) + 1):
        for CC in range(int(C_min), int(C_max) + 1):
            k_star, board, t = max_equal_k_by_cpsat(
                N=NN, C=CC, time_limit=float(time_limit), workers=int(workers), seed=int(seed)
            )
            rows.append({"N": NN, "C": CC, "k_star": k_star, "time_s": round(t, 2)})
            done += 1
            prog.progress(min(1.0, done / total))
    st.success("Sweep complete.")
    # show table (no pandas required)
    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv, file_name="peaceable_knights_results.csv", mime="text/csv")
    except Exception:
        # fallback simple text
        st.text(rows)





