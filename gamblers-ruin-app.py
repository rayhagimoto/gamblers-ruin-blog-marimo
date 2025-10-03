import marimo

__generated_with = "0.16.4"
app = marimo.App(width="full", auto_download=["html"])


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    return mo, np


@app.cell
def _():
    # Constants 
    # ---------

    p = 0.6           # Probability of winning a given round
    x_start = 100     # Starting position
    x_win = 1000      # Position needed to win
    dx = 10           # Step size
    return


@app.cell
def _(mo, np):
    @mo.persistent_cache()
    def simulate_many_full_vectorized(p, x_start, x_win, dx, max_iter, n_sims, seed=None, keep_paths=False):
        rng = np.random.default_rng(seed)

        # early exit if already absorbed at t=0
        if x_start >= x_win:
            outcome = np.ones(n_sims, dtype=np.int8)
            steps   = np.zeros(n_sims, dtype=np.int32)
            paths   = (np.full((1, n_sims), x_start, dtype=float) if keep_paths else None)
            return outcome, steps, paths
        if x_start <= 0:
            outcome = np.zeros(n_sims, dtype=np.int8)
            steps   = np.zeros(n_sims, dtype=np.int32)
            paths   = (np.full((1, n_sims), x_start, dtype=float) if keep_paths else None)
            return outcome, steps, paths

        T = max_iter
        # steps: +dx with prob p, else -dx
        steps_sign = (rng.random((n_sims, T)) < p).astype(np.int8) * 2 - 1
        steps_arr  = dx * steps_sign

        # positions after each step (t = 1..T)
        pos = x_start + np.cumsum(steps_arr, axis=1)   # shape (n_sims, T)

        # hit masks
        win_hit  = pos >= x_win
        lose_hit = pos <= 0

        # first-hit times via sentinel-min trick
        idx = np.arange(1, T + 1, dtype=np.int32)
        big = T + 1

        t_win  = np.minimum.reduce(np.where(win_hit,  idx, big), axis=1)
        t_lose = np.minimum.reduce(np.where(lose_hit, idx, big), axis=1)

        # overall first hit
        t_hit = np.minimum(t_win, t_lose)
        hit_mask = t_hit <= T

        # outcomes: 1 win, 0 lose, -1 timeout
        outcome = np.full(n_sims, -1, dtype=np.int8)
        win_mask  = (t_win  < t_lose) & hit_mask
        lose_mask = (t_lose < t_win)  & hit_mask
        outcome[win_mask]  = 1
        outcome[lose_mask] = 0

        # steps taken (T if timeout)
        steps_taken = np.where(hit_mask, t_hit, T).astype(np.int32)

        if keep_paths:
            # optional: mask positions after absorption with NaN
            # build a per-sim mask with True up to t_hit
            run_mask = np.arange(1, T + 1)[None, :] <= steps_taken[:, None]
            pos_masked = np.where(run_mask, pos, np.nan)
            return outcome, steps_taken, pos_masked

        return outcome, steps_taken, None
    return (simulate_many_full_vectorized,)


@app.cell
def _(mo):
    @mo.persistent_cache()
    def plot_paths(paths, dx=1, outcome=None, bins=None, x_win=None, density_threshold=100,
                   bins_x=20, use_zsmooth=False):
        """
        paths: (n_sims, T) with NaNs after absorption
        dx: step size multiplier for x-axis
        outcome: optional array (n_sims,), {1=win, 0=loss, -1=timeout} for coloring lines
        bins: required if n_sims >= density_threshold (used for density heatmap)
        x_win: REQUIRED to fix y-axis to [0, x_win] and draw reference lines
        density_threshold: cutoff between line vs density mode
        bins_x: number of x (step) bins in density mode
        use_zsmooth: pass True to set zsmooth="best"
        """
        if x_win is None:
            raise ValueError("x_win is required to fix y-axis limits to [0, x_win].")

        import numpy as np
        import plotly.graph_objects as go

        paths = np.asarray(paths)
        n_sims, T = paths.shape

        # Intelligent xmax: last step index with any finite value
        valid_any = np.isfinite(paths).any(axis=0)
        max_step_idx = int(np.nonzero(valid_any)[0][-1]) if valid_any.any() else -1

        # --- line plot mode ---
        if n_sims < density_threshold:
            fig = go.Figure()
            for i, y in enumerate(paths):
                valid = np.isfinite(y)
                if not np.any(valid):
                    continue
                x = dx * np.arange(valid.sum())
                if outcome is not None:
                    color = "#008080" if outcome[i] == 1 else "#b22222" if outcome[i] == 0 else "gray"
                else:
                    color = None
                fig.add_trace(go.Scatter(
                    x=x, y=y[valid], mode="lines",
                    line=dict(color=color) if color else None,
                    opacity=min(1.0, 2.0 / n_sims)  # updated opacity
                ))
            fig.update_layout(xaxis_title=f"dx × steps (dx={dx})", yaxis_title="position",
                              template="plotly_white", hovermode="x")
            fig.update_xaxes(range=[0, 7000])
            fig.update_yaxes(range=[0, x_win])
            return fig

        # --- density heatmap mode ---
        if bins is None:
            raise ValueError("bins must be provided when using density mode")

        dens, x_steps, y_centers, counts = density_from_paths(paths, bins, x_win=x_win)

        # trim to horizon
        if max_step_idx >= 0:
            keep_cols = max_step_idx + 1
            dens, counts, x_steps = dens[:, :keep_cols], counts[:, :keep_cols], x_steps[:keep_cols]

        # reduce to bins_x columns
        C = dens.shape[1]
        steps_per_bin = max(1, -(-C // bins_x))
        cut = (C // steps_per_bin) * steps_per_bin
        dens   = dens[:, :cut].reshape(dens.shape[0], cut // steps_per_bin, steps_per_bin).max(axis=2)
        counts = counts[:, :cut].reshape(counts.shape[0], cut // steps_per_bin, steps_per_bin).sum(axis=2)
        xb     = x_steps[:cut].reshape(cut // steps_per_bin, steps_per_bin)
        x_steps = xb[:, steps_per_bin // 2]

        # scale x axis
        x_vals = dx * x_steps

        fig = go.Figure(go.Heatmap(
            x=x_vals, y=y_centers, z=dens, zmin=0, zmax=1,
            zsmooth=("best" if use_zsmooth else False),
            colorscale="Teal",
            colorbar=dict(title="relative density")
        ))
        fig.update_layout(xaxis_title=f"dx × steps (dx={dx})", yaxis_title="position", template="plotly_white")
        fig.update_xaxes(range=[0, 7000])
        fig.update_yaxes(range=[0, x_win])
        return fig


    def density_from_paths(paths, bins, x_win=None):
        """
        Computes marginal densities over y-bins for each step.
        Forward-fills paths after absorption and clamps to bin range.
        Returns (dens, x_steps, y_centers, counts).
        """
        import numpy as np

        paths = np.asarray(paths)
        n_sims, T = paths.shape
        nb = len(bins) - 1

        valid = np.isfinite(paths)
        has_any = valid.any(axis=1)
        rev_first_true = np.argmax(valid[:, ::-1], axis=1)
        last_idx = (T - 1) - rev_first_true
        last_idx[~has_any] = -1
        rows = np.arange(n_sims)
        last_val = np.empty(n_sims, dtype=float)
        last_val[has_any] = paths[rows[has_any], last_idx[has_any]]
        last_val[~has_any] = 0.0
        if x_win is not None:
            last_val = np.clip(last_val, 0.0, float(x_win))

        steps = np.arange(T)[None, :]
        future_mask = steps > last_idx[:, None]
        paths_ff = np.where(future_mask, last_val[:, None], paths)

        right_inclusive = np.nextafter(bins[-1], -np.inf)
        paths_ff = np.clip(paths_ff, bins[0], right_inclusive)

        y = paths_ff.ravel()
        step_idx = np.tile(np.arange(T), n_sims)
        bin_idx = np.searchsorted(bins, y, side="right") - 1
        bin_idx = np.clip(bin_idx, 0, nb - 1)

        counts = np.zeros((nb, T), dtype=np.int32)
        np.add.at(counts, (bin_idx, step_idx), 1)

        col_max = counts.max(axis=0)
        dens = counts.astype(float)
        nz = col_max > 0
        if nz.any():
            dens[:, nz] /= col_max[nz]
        dens[:, ~nz] = 0.0

        y_centers = 0.5 * (bins[:-1] + bins[1:])
        return dens, np.arange(T), y_centers, counts
    return (plot_paths,)


@app.cell
def _(mo, np):
    p_slider = mo.ui.slider(steps=np.arange(0, 105, 5) / 100.0, value=0.6, label="`p = `")
    x_start_picker = mo.ui.slider(steps=np.arange(0, 10_010, 10), value=100, label='`start = `')
    x_win_picker = mo.ui.slider(steps=np.arange(0, 10_010, 10), value=1_000, label='`winning amount =`')
    dx_picker = mo.ui.slider(steps=np.geomspace(0.1, 100, 4), value=10, label='`step_size = `')
    n_sims = mo.ui.slider(steps=np.geomspace(1, 1000, 4).astype(int), value=10, label='`n_sims = `')
    return dx_picker, n_sims, p_slider, x_start_picker, x_win_picker


@app.cell
def _(dx_picker, mo, n_sims, p_slider, x_start_picker, x_win_picker):
    mo.vstack([
        p_slider,
        x_start_picker,
        x_win_picker,
        dx_picker,
        n_sims
    ])
    return


@app.cell
def _(
    dx_picker,
    n_sims,
    p_slider,
    simulate_many_full_vectorized,
    x_start_picker,
    x_win_picker,
):
    # Generate the simulations
    outcome, steps, paths = simulate_many_full_vectorized(
        p_slider.value, x_start_picker.value, x_win_picker.value, dx_picker.value, max_iter=100_000, n_sims=n_sims.value, seed=42, keep_paths=True
    )
    return outcome, paths


@app.cell
def _(dx_picker, mo, np, outcome, paths, plot_paths, x_win_picker):
    _delta = dx_picker.value
    _bins = np.arange(0, x_win_picker.value + _delta, _delta)

    fig = plot_paths(
        paths,
        dx=_delta,
        outcome=outcome,          # include if you have it
        bins=_bins,
        x_win=x_win_picker.value,
        bins_x=40,
        density_threshold=100,
    )
    mo.ui.plotly(fig)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
