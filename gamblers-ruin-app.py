import marimo

__generated_with = "0.16.4"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import altair as alt
    import plotly.graph_objects as go
    return alt, go, mo, np


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

        # positions after each step (t=1..T)
        pos = x_start + np.cumsum(steps_arr, axis=1)   # shape (n_sims, T)

        # prepend initial position so paths start at x_start
        pos = np.concatenate([np.full((n_sims, 1), x_start, dtype=float), pos], axis=1)  # shape (n_sims, T+1)

        # hit masks
        win_hit  = pos >= x_win
        lose_hit = pos <= 0

        idx = np.arange(pos.shape[1], dtype=np.int32)  # 0..T
        big = T + 1

        t_win  = np.minimum.reduce(np.where(win_hit,  idx, big), axis=1)
        t_lose = np.minimum.reduce(np.where(lose_hit, idx, big), axis=1)

        # overall first hit
        t_hit = np.minimum(t_win, t_lose)
        hit_mask = t_hit <= T

        # outcomes
        outcome = np.full(n_sims, -1, dtype=np.int8)
        win_mask  = (t_win  < t_lose) & hit_mask
        lose_mask = (t_lose < t_win)  & hit_mask
        outcome[win_mask]  = 1
        outcome[lose_mask] = 0

        # steps taken (T if timeout)
        steps_taken = np.where(hit_mask, t_hit, T).astype(np.int32)

        if keep_paths:
            # mask positions after absorption
            run_mask = np.arange(pos.shape[1])[None, :] <= steps_taken[:, None]
            pos_masked = np.where(run_mask, pos, np.nan)
            return outcome, steps_taken, pos_masked

        return outcome, steps_taken, None
    return (simulate_many_full_vectorized,)


@app.cell
def _(mo):
    mo.md(r"""# Monte Carlo Simulations""")
    return


@app.cell
def _(simulate_many_full_vectorized):
    def generate_data_for_plot():

        p = 0.6
        x_start = 100.0
        x_win = 1000.0
        _n_sims = 3

        # Pre-compute plots for multiple dx values
        dx_values = [0.1, 1, 10, 100]
        paths = {}
        all_plots = {}

        for dx in dx_values:
            outcome, steps, _paths = simulate_many_full_vectorized(
                p, x_start, x_win, dx, max_iter=100_000, n_sims=_n_sims, seed=16, keep_paths=True
            )
            paths[dx] = _paths

        return paths
    return (generate_data_for_plot,)


@app.cell
def _(generate_data_for_plot, go, mo, np):
    def _subsample_paths_for_dx(paths_for_dx, dx, max_samples=1000):
        out = []
        for p in paths_for_dx:
            f = ~np.isnan(p)
            y = p[f]
            n = y.size
            if n == 0:
                continue

            # x must correspond ONLY to the non-NaN steps
            x = (1 + np.arange(n, dtype=float)) * dx

            # optional subsample
            if n > max_samples:
                step = int(np.ceil(n / max_samples))
                x = x[::step]
                y = y[::step]

            out.append((x, y))
        return out


    def _make_path_traces_black(subsampled_paths):
        traces = []
        for i, (x, y) in enumerate(subsampled_paths, 1):
            traces.append(go.Scatter(
                x=x, y=y, mode="lines",
                line=dict(color="black"),
                opacity=0.5,
                name=f"Sim {i}",
                showlegend=False
            ))
        return traces

    def plot_paths_with_dx_slider(paths_dict, max_samples=500):
        dx_values = [0.1, 10.0, 100.0]

        fig = go.Figure()

        # Fixed EV line
        fig.add_trace(go.Scatter(
            x=[0, 5000], y=[100, 1000],
            mode="lines", name="EV",
            line_color="Red"
        ))

        # Initial dx (prefer 10 if available)
        init_dx = 10.0 if 10.0 in paths_dict else list(paths_dict.keys())[0]
        init_sub = _subsample_paths_for_dx(paths_dict[init_dx], init_dx, max_samples=max_samples)
        for tr in _make_path_traces_black(init_sub):
            fig.add_trace(tr)

        # Frames: swap sim traces and update a LaTeX annotation for dx
        frames = []
        for dx in dx_values:
            if dx not in paths_dict:
                continue
            sub = _subsample_paths_for_dx(paths_dict[dx], dx, max_samples=max_samples)
            frame_traces = []
            frame_traces.append(go.Scatter(  # EV line stays the same
                x=[0, 5000], y=[100, 100 + 1000],
                mode="lines", name="EV",
                line_color="Red"
            ))
            frame_traces.extend(_make_path_traces_black(sub))

            # LaTeX dx annotation (e.g., \mathrm{d}x = 0.1)
            ann = [dict(
                xref="paper", yref="paper",
                x=0.02, y=1.05,
                xanchor="left", yanchor="bottom",
                text=rf"$\mathrm{{d}}x = {dx:g}$",
                showarrow=False
            )]

            frames.append(go.Frame(name=str(dx), data=frame_traces, layout=go.Layout(annotations=ann)))

        fig.frames = frames

        # Slider only (no play/pause)
        steps = []
        for dx in dx_values:
            # when building slider steps
            steps.append({
                    "method": "animate",
                    "label": f"{dx:g}",
                    "args": [[str(dx)], {"mode": "immediate",
                                         "frame": {"duration": 0, "redraw": True},
                                         "transition": {"duration": 0}}]
                })


        fig.update_layout(
            sliders=[{
                "active": [str(v) for v in dx_values].index(str(init_dx)) if str(init_dx) in [str(v) for v in dx_values] else 0,
                "pad": {"t": 10},
                "len": 0.6,
                "x": 0.2,
                "y": -0.2,
                "currentvalue": {"visible": False},  # hide built-in current value
                "steps": steps
            }]
        )

        # Axes limits & aesthetics
        fig.update_layout(
            xaxis_range=[0, 5500],
            yaxis_range=[0, 1000],
            xaxis=dict(
                showline=True, linecolor="black",
                showgrid=False, zeroline=False,
                ticks="inside", ticklen=7,
                minor=dict(showgrid=False, ticks="inside", ticklen=4)
            ),
            yaxis=dict(
                showline=True, linecolor="black",
                showgrid=False, zeroline=False,
                ticks="inside", ticklen=7,
                minor=dict(showgrid=False, ticks="inside", ticklen=4)
            ),
            plot_bgcolor="white",
            # Axis titles in LaTeX; all other fonts default
            xaxis_title=r"$\text{Total Distance Traveled (m)}$",
            yaxis_title=r"$\text{Position } X \text{ (m)}$",
        )

        # Keep scale fixed on frame change
        fig.update_xaxes(fixedrange=True)
        fig.update_yaxes(fixedrange=True)

        # Add initial dx annotation for the starting frame
        fig.update_layout(annotations=[
            dict(
                xref="paper", yref="paper",
                x=0.02, y=1.05,
                xanchor="left", yanchor="bottom",
                text=rf"$\mathrm{{d}}x = {init_dx:g}$",
                showarrow=False
            )
        ])

        return fig

    def write_interactive_plot_to_html():

        paths = generate_data_for_plot()
        # Example usage:
        fig = plot_paths_with_dx_slider(paths)  # paths: {0.1: [...], 10.0: [...], 100.0: [...]}
        mo.ui.plotly(fig)
        fig.write_html("interactive_plot.html")
    return (write_interactive_plot_to_html,)


@app.cell
def _(write_interactive_plot_to_html):
    write_interactive_plot_to_html()
    return


@app.cell
def _(mo, np):
    @mo.persistent_cache()
    def compute_chase_probability_distribution(total_distance, dx, x_start=100, x_win=1000, p=0.6):
        """
        Compute the probability distribution of final position after traveling total_distance
        in the high-speed chase scenario using binomial distribution.

        Parameters:
        - total_distance: total distance traveled (meters)
        - dx: step size (meters per step)
        - x_start: starting position (100m from cops)
        - x_win: safehouse position (1000m from cops)
        - p: probability of taking a step away from cops

        Returns:
        - positions: array of possible final positions
        - probabilities: array of probabilities for each position
        """
        from scipy.stats import binom

        # Calculate number of steps from total distance and step size
        n = int(total_distance / dx)

        # Use a fixed x-axis range that doesn't depend on dx
        # This ensures constant bar widths regardless of step size
        x_min = 0
        x_max = 1200

        # Constant resolution in x-space (100 points across the range)
        n_points = 100
        x_positions = np.linspace(x_min, x_max, n_points)

        # Convert positions back to k_away values
        k_away = (x_positions - x_start + n * dx) / (dx)
        k_away = np.round(k_away).astype(int)
        print(k_away)

        # Filter to valid range
        valid_mask = (k_away >= 0) & (k_away <= n)
        k_away = k_away[valid_mask]
        x_positions = x_positions[valid_mask]

        # Ensure we don't exceed Altair's row limit
        print(x_positions)
        if len(x_positions) > 10000:
            # Downsample if we have too many points
            step = len(x_positions) // 1000
            x_positions = x_positions[::step]
            k_away = k_away[::step]

        # Calculate probabilities for the valid k_away values
        probabilities = binom.pmf(k_away, n, p)

        # Use the x_positions as final_positions
        final_positions = x_positions

        return final_positions, probabilities

    @mo.persistent_cache()
    def plot_chase_probability_distribution(positions, probabilities, x_start=100, x_win=1000, dx=None):
        """
        Plot the probability distribution for the chase scenario using Altair.

        Parameters:
        - positions: array of possible final positions
        - probabilities: array of probabilities for each position
        - x_start: starting position
        - x_win: safehouse position
        - dx: step size (for title)
        """
        import pandas as pd
        import altair as alt

        # Create DataFrame for Altair
        df = pd.DataFrame({
            'position': positions,
            'probability': probabilities
        })

        # Calculate statistics (currently unused but kept for potential future use)
        # prob_safe = np.sum(probabilities[positions >= x_win])
        # prob_caught = np.sum(probabilities[positions <= 0])
        # prob_surviving = np.sum(probabilities[(positions > 0) & (positions < x_win)])
        # expected_position = np.sum(positions * probabilities)

        # Always use full range [0, 1200] for x-axis
        x_min = 0
        x_max = 1200

        # Create base chart
        base = alt.Chart(df).mark_bar(
            color='steelblue',
            opacity=0.7
        ).encode(
            x=alt.X('position:Q', title='Position (meters from cops)', scale=alt.Scale(domain=[x_min, x_max])),
            y=alt.Y('probability:Q', title='Probability', axis=alt.Axis(tickCount=0)),
            tooltip=[
                alt.Tooltip('position:Q', title='Position', format='.1f'),
                alt.Tooltip('probability:Q', title='Probability', format='.4f')
            ]
        )

        # Add vertical lines for key positions
        start_line = alt.Chart(pd.DataFrame({'x': [x_start]})).mark_rule(
            color='orange',
            strokeDash=[5, 5]
        ).encode(x='x:Q')

        safehouse_line = alt.Chart(pd.DataFrame({'x': [x_win]})).mark_rule(
            color='green',
            strokeDash=[5, 5]
        ).encode(x='x:Q')

        cops_line = alt.Chart(pd.DataFrame({'x': [0]})).mark_rule(
            color='red',
            strokeDash=[5, 5]
        ).encode(x='x:Q')



        # Combine all layers
        chart = (base + start_line + safehouse_line + cops_line).resolve_scale(
            x='shared'
        ).properties(
            title=f"Probability Distribution of Final Position (dx = {dx}m)",
            width=600,
            height=400
        )

        return chart
    return


@app.cell
def _(mo):
    mo.md(r"""# Terminal distribution after $n$ steps $p(x_{n})$""")
    return


@app.cell
def _(np):
    from scipy.special import gammaln

    def compute_distribution_after_nsteps(
        p: float,
        dx: float,
        distance_traveled: float,
        x_start: float = 100.0,
        x_end: float = 1000.0,   # unused, kept for API symmetry
        xmin: float = 0.0,       # unused, for plotting bounds if needed
        xmax: float = 1100.0,    # unused
        ymin: float = 0.0,       # unused
        ymax: float = 1.0,       # unused
        res_x_max: float = 0.5   # max samples per unit x (≥ 0)
    ):
        """
        Binomial walk endpoint distribution after n_steps = floor(distance_traveled/dx).
        - Uses ±3σ window around mean for speed.
        - Subsamples states so that plotted x-resolution ≤ 1/res_x_max per unit x.
        """
        # steps and offsets
        n_steps = int(distance_traveled // dx)
        if n_steps < 0:
            raise ValueError("distance_traveled must be ≥ 0")
        if n_steps == 0:
            return np.array([x_start], dtype=float), np.array([1.0], dtype=float)

        k_offset = int(x_start // dx)

        # all possible k_right (0..n_steps), implied x spacing is 2*dx
        k_right_all = np.arange(n_steps + 1, dtype=np.int64)
        k_away_all  = 2 * k_right_all - n_steps
        k_all       = k_away_all + k_offset
        x_all       = k_all * dx

        # mean and ±3σ window
        x_mean = x_start + (2.0 * p - 1.0) * n_steps * dx
        sdev_3 = 3.0 * (np.sqrt(n_steps * p * (1.0 - p)) * 2.0 * dx)
        in_win = (x_all > (x_mean - sdev_3)) & (x_all < (x_mean + sdev_3))

        # subsample to enforce ≤ res_x_max samples per unit x
        # each increment of k_right changes x by 2*dx
        if res_x_max is None or res_x_max <= 0:
            stride = 1
        else:
            desired_dx = 1.0 / res_x_max
            stride = int(np.ceil(desired_dx / (2.0 * dx)))
            stride = max(1, stride)

        idx = np.nonzero(in_win)[0][::stride]
        if idx.size == 0:  # fallback to at least the closest point to mean
            idx = np.array([np.argmin(np.abs(x_all - x_mean))])

        k_right = k_right_all[idx]
        k_left  = n_steps - k_right
        x_plot  = x_all[idx]

        # log PMF via gammaln for numerical stability
        log_p, log_q = np.log(p), np.log1p(-p)
        log_binom = gammaln(n_steps + 1) - gammaln(k_right + 1) - gammaln(k_left + 1)
        log_pmf   = log_binom + k_right * log_p + k_left * log_q
        y_plot    = np.exp(log_pmf)

        return x_plot, y_plot
    return (compute_distribution_after_nsteps,)


@app.cell
def _(alt, compute_distribution_after_nsteps):
    import pandas as pd

    def plot_distribution_after_nsteps(
        p, dx, distance_traveled,
        x_start=100.0, x_end=1000.0,
        xmin=0.0, xmax=1100.0, ymin=0, ymax=1
    ):
        x_plot, y_plot = compute_distribution_after_nsteps(
            p, dx, distance_traveled, x_start, x_end, xmin, xmax, ymin, ymax
        )
        df = pd.DataFrame({"x": x_plot, "y": y_plot})
        y_max = float(df["y"].max())

        base = alt.Chart(df).encode(
            x=alt.X("x:Q",
                    scale=alt.Scale(domain=(xmin, xmax)),
                    axis=alt.Axis(values=list(range(int(xmin), int(xmax)+1, 50)),
                                  labelAngle=45)),
            y=alt.Y("y:Q",
                    scale=alt.Scale(domain=(ymin, y_max)),
                    axis=None),
            tooltip=["x", "y"]
        )

        area = base.mark_area(color="steelblue", opacity=0.35, interpolate="step")  # or "monotone"
        line = base.mark_line(color="steelblue", strokeWidth=2, interpolate="step")

        return (area + line).properties(width=700, height=300)
    return (plot_distribution_after_nsteps,)


@app.cell
def _(mo, np):
    p_slider_1d = mo.ui.slider(steps = np.arange(0.1, 1.1, 0.1), value=0.6, label="`p = `")
    dx_slider_1d = mo.ui.slider(steps = np.geomspace(0.01, 100.0, 5), value=10.0, label="`dx = `")
    # distance_slider_1d = mo.ui.slider(steps = np.arange(0, 7100.0, 10.0), value=40.0, label="`distance = `")
    return dx_slider_1d, p_slider_1d


@app.cell
def _(dx_slider_1d, mo, np):
    _dx = dx_slider_1d.value

    distance_slider_1d = mo.ui.slider(steps = np.arange(0, 5000 / _dx + 10 / _dx, 10 / _dx), value=4500 / _dx, label="`n_steps = `")
    return (distance_slider_1d,)


@app.cell
def _(
    distance_slider_1d,
    dx_slider_1d,
    mo,
    p_slider_1d,
    plot_distribution_after_nsteps,
):
    _fig = plot_distribution_after_nsteps(p_slider_1d.value, dx_slider_1d.value, distance_slider_1d.value * dx_slider_1d.value)

    mo.vstack([
        p_slider_1d,
        dx_slider_1d,
        distance_slider_1d,
        mo.ui.altair_chart(_fig)
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Experimental timeit function""")
    return


@app.cell(hide_code=True)
def _():
    import gc
    from time import perf_counter
    from statistics import mean, median, pstdev

    # --- helpers ---------------------------------------------------------------

    def _human_time(sec: float) -> str:
        """SI prefixes, picks a clean unit; supports ns…h."""
        if sec < 0:
            return f"-{_human_time(-sec)}"
        units = [(1e-9,"ns"), (1e-6,"µs"), (1e-3,"ms"), (1.0,"s"), (60.0,"min"), (3600.0,"h")]
        if sec < 1e-9:
            return f"{sec:.3e} s"
        for scale, name in reversed(units):
            if sec >= scale:
                val = sec / scale
                if val >= 100: return f"{val:.0f} {name}"
                if val >= 10:  return f"{val:.1f} {name}"
                return f"{val:.3g} {name}"
        return f"{sec:.3g} s"

    def _pick_repeats(first_run: float) -> int:
        if first_run >= 3.0: return 1
        if first_run < 0.01: return 100   # <10 ms
        if first_run < 1.0:  return 10    # [10 ms, 1 s)
        return 3

    def _short_repr(x, maxlen=120):
        r = repr(x)
        return r if len(r) <= maxlen else r[:maxlen-3] + "..."

    # --- decorator -------------------------------------------------------------

    def timeit(_func=None, *, disable_gc=True, print_stats=True, return_stats=False,
               attach_attr="__timing__", pick_repeats=_pick_repeats):
        """
        Decorator form of your timer.

        Usage:
          @timeit
          def f(...): ...

          @timeit(disable_gc=False, return_stats=True)
          def g(...): ...

        Options:
          - disable_gc: temporarily disable GC during timing
          - print_stats: pretty print summary
          - return_stats: if True, return (result, stats_dict)
          - attach_attr: store last stats on function via this attribute (or None to skip)
          - pick_repeats: function(first_run_seconds) -> int
        """
        def _decorate(func):
            def wrapper(*args, **kwargs):
                gc_was_enabled = gc.isenabled()
                if disable_gc and gc_was_enabled:
                    gc.disable()
                try:
                    # first run
                    t0 = perf_counter()
                    result = func(*args, **kwargs)
                    t1 = perf_counter()
                    times = [t1 - t0]
                    repeats = pick_repeats(times[0])

                    # remaining runs
                    for _ in range(repeats - 1):
                        t0 = perf_counter()
                        _ = func(*args, **kwargs)
                        t1 = perf_counter()
                        times.append(t1 - t0)
                finally:
                    if disable_gc and gc_was_enabled:
                        gc.enable()

                per_call = mean(times)
                tot = sum(times)
                md = median(times)
                sd = pstdev(times) if len(times) > 1 else 0.0
                best = min(times)
                worst = max(times)

                name = getattr(func, "__name__", "<callable>")
                arg_str = f"args={_short_repr(args)}, kwargs={_short_repr(kwargs)}"
                if print_stats:
                    print(
                        f"[timeit] {name}\n"
                        f"[timeit] repeats={repeats}  per-call: {_human_time(per_call)}  "
                        f"(median {_human_time(md)}, min {_human_time(best)}, max {_human_time(worst)})\n"
                        f"[timeit] total: {_human_time(tot)}  | per-call (sci): {per_call:.3e} s\n"
                        f"[timeit] {arg_str}\n"
                    )

                stats = {
                    "function": name,
                    "repeats": repeats,
                    "times": times,
                    "per_call_mean": per_call,
                    "per_call_median": md,
                    "per_call_std": sd,
                    "per_call_min": best,
                    "per_call_max": worst,
                    "total": tot,
                    "args": args,
                    "kwargs": kwargs,
                }
                if attach_attr:
                    setattr(wrapper, attach_attr, stats)

                if return_stats:
                    return result, stats
                return result
            return wrapper

        # bare @timeit vs @timeit(...)
        return _decorate if _func is None else _decorate(_func)
    return


@app.cell
def _(np, paths):
    def _plot_paths(paths, dx, max_samples=1000):
        import plotly.graph_objects as go
        subsampled_paths = []
        # Get the paths as the non-nan values, but be sure to subsample them.
        for p in paths:
            f_not_nan =  ~np.isnan(p)
            n_steps = f_not_nan.sum() # number of steps in this path (this varies per path)
            x = 1 + np.arange(len(p)).astype(float)
            x *= dx
            if n_steps > max_samples + 1:
                n_skip = n_steps // max_samples + 1
                x = x[::n_skip]
                p = p[f_not_nan][::n_skip]
            subsampled_paths.append((x, p))


        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 5000], y=[100, 1000], name="EV", mode="lines", line_color="#FF746C"))
        n = len(subsampled_paths)
        for i, (x, p) in enumerate(subsampled_paths, 1):
            # keep hue=200 (blue), saturation=80%, vary lightness from 30%→80%
            lightness = 10 + (i-1) * (50 / max(1, n-1))
            color = f"hsl(200, 80%, {lightness}%)"

            fig.add_trace(
                go.Scatter(
                    x=x, y=p,
                    mode="lines",
                    line=dict(color="#3E4772"),
                    name=f"Sim {i}",
                    opacity=0.5
                )
            )

        # Set ax limits
        fig.update_layout(
                xaxis_range = [0, 7000],
                yaxis_range = [0, 1000],
            )

        # Aesthetics
        fig.update_layout(
            xaxis=dict(
                showline=True,
                linecolor="black",
                showgrid=False,
                zeroline=False,
                ticks="inside",
                ticklen=7,  # major ticks
                minor=dict(
                    showgrid=False,
                    ticks="inside",
                    ticklen=4  # minor ticks (shorter)
                )
            ),
            yaxis=dict(
                showline=True,
                linecolor="black",
                showgrid=False,
                zeroline=False,
                ticks="inside",
                ticklen=7,
                minor=dict(
                    showgrid=False,
                    ticks="inside",
                    ticklen=4
                )
            ),
            plot_bgcolor="white"
        )

        # Axis titles
        fig.update_layout(
            xaxis_title=r"$\text{Total Distance Traveled (m)}$",
            yaxis_title=r"$\text{Position } X \text{ (m)}$"
        )
        return fig


    _dx = 10
    _plot_paths(paths[_dx], _dx)
    return


@app.cell
def _():
    return


@app.cell
def _(np, paths):
    def _():
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        dx = 100
        _paths = paths[dx]
        for p in _paths:
            f_not_nan = ~np.isnan(p)
            print(f_not_nan.sum())
            path = p[f_not_nan]
            x = 1 + np.arange(len(path))
            x *= dx
            plt.plot(x, path)
        return fig

    _()
    return


if __name__ == "__main__":
    app.run()
