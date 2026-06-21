# 1. Setup
using Pkg; Pkg.activate(@__DIR__)

import TulipaIO          as TIO
import TulipaEnergyModel as TEM
import TulipaClustering  as TC
import MathOptInterface  as MOI
using DuckDB, DataFrames, Plots, Random, JuMP, Gurobi
using Statistics, Logging
include("artificial-worst-case-clustering.jl")
include("real-worst-case-clustering.jl")

# 2. Configuration
const INPUT_DIR  = joinpath(@__DIR__, "data", "tutorial-9")
const OUTPUT_DIR = joinpath(INPUT_DIR, "results")  # "" = skip saving

period_duration  = 24
min_clusters     = 2
max_clusters     = 0        # 0 = use all periods
step_sizes = [
    (50,   2),
    (200,  5),
    (400,  25),
    (99999, 150),
]
n_seeds          = 5      
target_years     = [2030]
target_scenarios = []
const LAYOUT = TC.ProfilesTableLayout(;
    year             = :year,
    cols_to_groupby  = [:year],
    cols_to_crossby  = [:scenario],
)

methods = [:k_medoids, :k_medoids_wc_frac, :k_medoids_wc_unit,
           :k_medoids_rwc_netload, :k_medoids_rwc_elgersma, :k_medoids_rwc_ens]
baseline_method = :k_medoids

do_regret = true

# Weight fraction given to the artificial period (:k_medoids_wc_frac).
const WC_FRACTION = 0.1

METHOD_STYLES = Dict(
    :k_medoids              => (color=:black,        stroke=:solid,   marker=:circle,    lw=2.0, ms=4.5),
    :k_medoids_wc_unit      => (color=:royalblue,    stroke=:dash,    marker=:square,    lw=2.0, ms=4.5),
    :k_medoids_wc_frac      => (color=:mediumpurple, stroke=:dot,     marker=:diamond,   lw=2.0, ms=4.5),
    :k_medoids_rwc_netload  => (color=:darkorange,   stroke=:solid,   marker=:utriangle, lw=2.0, ms=4.5),
    :k_medoids_rwc_elgersma => (color=:crimson,      stroke=:dash,    marker=:dtriangle, lw=2.0, ms=4.5),
    :k_medoids_rwc_ens      => (color=:seagreen,     stroke=:dashdot, marker=:star5,     lw=2.0, ms=4.5),
)

# 3. Optimizer
# Set Threads and suppress all parameter messages on the environment level.
# This stops the repeated "Set parameter Threads to value 12" spam.
const GRB_ENV = Gurobi.Env()
Gurobi.GRBsetintparam(GRB_ENV, "OutputFlag", 0)
Gurobi.GRBsetintparam(GRB_ENV, "Threads", 12)
const OPTIMIZER  = () -> Gurobi.Optimizer(GRB_ENV)
const OPT_PARAMS = Dict{String,Any}(
    "OutputFlag"   => 0,
    "LogToConsole" => 0,
    "Threads"      => 12,
)


# 4. Data helpers

function load_connection()
    c = DBInterface.connect(DuckDB.DB)
    TIO.read_csv_folder(c, INPUT_DIR)
    TC.transform_wide_to_long!(c, "profiles_wide", "profiles";
        exclude_columns = ["milestone_year", "timestep", "scenario"])
    DuckDB.execute(c, "ALTER TABLE profiles RENAME COLUMN milestone_year TO year")
    isempty(target_years)     || DuckDB.execute(c,
        "DELETE FROM profiles WHERE year NOT IN ($(join(target_years, ',')))")
    isempty(target_scenarios) || DuckDB.execute(c,
        "DELETE FROM profiles WHERE scenario NOT IN ($(join(target_scenarios, ',')))")
    return c
end

# TC.cluster! writes tables with a 'year' column; TEM expects 'milestone_year'.
function fix_milestone_year!(c)
    yr = isempty(target_years) ? 0 : target_years[1]
    for t in ("profiles_rep_periods", "rep_periods_data", "rep_periods_mapping", "timeframe_data")
        try; DuckDB.execute(c, "ALTER TABLE $t RENAME COLUMN year TO milestone_year"); catch; end
        DuckDB.execute(c, "UPDATE $t SET milestone_year = $yr WHERE milestone_year IS NULL")
    end
end


# 5. Metric helpers

"""
Computes Loss of Load Expectation (h/yr) and Expected Energy Not Served (MWh/yr).

LOLE = Σ  block_hours × weight          [hours/year]
EENS = Σ  solution × block_hours × weight  [MWh/year]

where the sum is over all ENS flow blocks with solution > 0,
block_hours = time_block_end − time_block_start + 1,
weight = number of real periods the rep-period represents.
"""
function compute_lole(c)
    ens = DuckDB.query(c, """
        SELECT rep_period,
               milestone_year,
               (time_block_end - time_block_start + 1)            AS block_hours,
               solution * (time_block_end - time_block_start + 1) AS energy
        FROM   var_flow
        WHERE  lower(from_asset) LIKE '%ens%'
          AND  solution > 0
    """) |> DataFrame
    nrow(ens) == 0 && return (0.0, 0.0)
    weights = DuckDB.query(c, """
        SELECT rep_period, milestone_year, SUM(weight) AS w
        FROM   rep_periods_mapping
        GROUP  BY rep_period, milestone_year
    """) |> DataFrame
    m = leftjoin(ens, weights; on = [:rep_period, :milestone_year])
    m.w .= coalesce.(m.w, 0.0)
    lole = sum(m.block_hours .* m.w)
    eens = sum(m.energy      .* m.w)
    return (lole, eens)
end

"""
Total annualised investment cost (EUR/yr) of the solved plan:
cost = Σ solution × annualized_cost
"""
function investment_cost(c)
    total = 0.0
    tbls = all(t -> (DuckDB.query(c,
        "SELECT COUNT(*) AS n FROM information_schema.tables WHERE table_name='$t'") |>
        DataFrame).n[1] > 0, ("var_assets_investment", "t_objective_assets"))
    if tbls
        df = DuckDB.query(c,
            "SELECT SUM(r.solution * o.annualized_cost) AS cost
             FROM var_assets_investment r
             JOIN t_objective_assets o ON r.asset = o.asset
             AND r.milestone_year = o.milestone_year") |> DataFrame
        total += coalesce(df.cost[1], 0.0)
    end
    if (DuckDB.query(c,
            "SELECT COUNT(*) AS n FROM information_schema.tables WHERE table_name='var_flows_investment'"
            ) |> DataFrame).n[1] > 0
        df = DuckDB.query(c,
            "SELECT SUM(r.solution * o.annualized_cost) AS cost
             FROM var_flows_investment r
             JOIN t_objective_flows o ON r.from_asset = o.from_asset
             AND r.to_asset = o.to_asset
             AND r.milestone_year = o.milestone_year") |> DataFrame
        total += coalesce(df.cost[1], 0.0)
    end
    return total
end


# 6. Regret helpers

function fix_investments!(bench, reduced)
    for sym in (:assets_investment, :assets_investment_energy, :flows_investment)
        haskey(bench.variables,   sym) || continue
        haskey(reduced.variables, sym) || continue
        for (var, ref) in zip(bench.variables[sym].container, reduced.variables[sym].container)
            JuMP.fix(var, JuMP.value(ref); force=true)
        end
    end
end

function unfix_investments!(bench)
    for sym in (:assets_investment, :assets_investment_energy, :flows_investment)
        haskey(bench.variables, sym) || continue
        for var in bench.variables[sym].container
            JuMP.unfix(var)
        end
    end
end


# 7. run_once

# Each call opens its own connection so re-clustering never sees stale tables.
function run_once(k; method=:k_medoids, return_problem=false)
    c  = load_connection()
    t0 = time_ns()
    TC.cluster!(c, period_duration, k; method, layout = LAYOUT)
    fix_milestone_year!(c)
    TEM.populate_with_defaults!(c)
    prob = try
        with_logger(NullLogger()) do
            TEM.run_scenario(c; optimizer=OPTIMIZER, optimizer_parameters=OPT_PARAMS,
                             show_log=false)
        end
    catch
        elapsed = (time_ns() - t0) / 1e9
        DBInterface.close!(c)
        nan = (cost=NaN, inv=NaN, lole=NaN, eens=NaN, time=elapsed)
        return return_problem ? merge(nan, (problem=nothing,)) : nan
    end
    elapsed = (time_ns() - t0) / 1e9
    inv     = investment_cost(c)
    DBInterface.close!(c)
    result = (cost=prob.objective_value, inv=inv, lole=NaN, eens=NaN, time=elapsed)
    return return_problem ? merge(result, (problem=prob,)) : result
end


# 8. run_experiment!

init_results() = Dict(m => (clusters=Int[], cost=Float64[], inv=Float64[],
    regret=Float64[], lole=Float64[], eens=Float64[], time=Float64[]) for m in methods)

function run_experiment!(steps, results, wc_df;
                         bench_prob=nothing, bench_conn=nothing, bench_cost=NaN,
                         netload_score_df=nothing, elgersma_score_df=nothing)
    try
        for k in steps, m in methods
            r = if m == :k_medoids_wc_unit
                    run_once_wc(k, wc_df; weight_type=:wc_unit, return_problem=do_regret)
                elseif m == :k_medoids_wc_frac
                    run_once_wc(k, wc_df; weight_type=:wc_frac, return_problem=do_regret)
                elseif m == :k_medoids_rwc_netload
                    run_once_rwc(k, netload_score_df, :net_load_score;
                                 return_problem=do_regret)
                elseif m == :k_medoids_rwc_elgersma
                    run_once_rwc(k, elgersma_score_df, :elgersma_score;
                                 return_problem=do_regret)
                elseif m == :k_medoids_rwc_ens
                    run_once_rwc_ens(k; return_problem=do_regret)
                else
                    run_once(k; method=m, return_problem=do_regret)
                end
            regret = NaN
            lole   = NaN
            eens   = NaN
            if do_regret && r.problem !== nothing && bench_prob !== nothing && isfinite(bench_cost)
                fix_investments!(bench_prob, r.problem)
                JuMP.optimize!(bench_prob.model)
                st  = JuMP.termination_status(bench_prob.model)
                agg = st in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED) ?
                          JuMP.objective_value(bench_prob.model) : NaN
                regret = isfinite(agg) ? (agg - bench_cost) / bench_cost * 100 : NaN
                if st in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED) && bench_conn !== nothing
                    TEM.save_solution!(bench_prob; compute_duals=false)
                    lole, eens = compute_lole(bench_conn)
                end
                unfix_investments!(bench_prob)
            end
            res = results[m]
            push!(res.clusters, k);    push!(res.cost,   r.cost)
            push!(res.inv,      r.inv); push!(res.regret, regret)
            push!(res.lole,     lole);  push!(res.eens,   eens)
            push!(res.time,     r.time)
            println(isfinite(r.cost) ?
                "$m  k=$k  cost=$(round(r.cost,digits=2))  regret=$(round(regret,digits=3))%" :
                "$m  k=$k  infeasible")
        end
    catch e
        e isa InterruptException && return :interrupted
        rethrow()
    end
    return :ok
end


# 9. Plotting

function build_plots(results; ref=nothing)
    ZOOM_MIN = 50

    fa = (
        titlefontsize  = 10,
        guidefontsize  = 9,
        tickfontsize   = 8,
        legendfontsize = 6,
        grid           = true,
        gridalpha      = 0.15,
        frame          = :box,
        dpi            = 300,
    )

    function add!(plt, m, xs, ys)
        s = get(METHOD_STYLES, m, (color=:auto, stroke=:solid, marker=:none, lw=1.5, ms=4.0))
        plot!(plt, xs, ys;
            label=replace(string(m), "_"=>"-"),
            color=s.color, linestyle=s.stroke, marker=s.marker,
            linewidth=s.lw, markersize=s.ms, markerstrokewidth=0,
            alpha=0.85, markeralpha=0.5)
    end

    inv_dev(r) = (ref !== nothing && isfinite(ref.inv)) ?
        [isfinite(v) ? (v - ref.inv)/ref.inv*100.0 : NaN for v in r.inv] : r.inv

    # ── linear (originals) ────────────────────────────────────────────────────
    p_regret = plot(; title="Regret (%)",          xlabel="Clusters", ylabel="Regret [%]",                    fa..., legend=:best)
    p_time   = plot(; title="Solve Time (s)",      xlabel="Clusters", ylabel="Time [s]",                      fa..., legend=:best)
    p_lole   = plot(; title="LOLE Full-Res",       xlabel="Clusters", ylabel="LOLE [h/yr]",                   fa..., legend=:best)
    p_eens   = plot(; title="EENS Full-Res",       xlabel="Clusters", ylabel="EENS [MWh/yr]",                 fa..., legend=:best)
    p_inv    = plot(; title="Investment Deviation", xlabel="Clusters", ylabel="Deviation from benchmark [%]", fa..., legend=:best)

    # ── log-x ─────────────────────────────────────────────────────────────────
    p_regret_logx = plot(; title="Regret (%) — Log x",     xlabel="Clusters (log)", ylabel="Regret [%]",            fa..., legend=:best, xscale=:log10)
    p_lole_logx   = plot(; title="LOLE — Log x",           xlabel="Clusters (log)", ylabel="LOLE [h/yr]",           fa..., legend=:best, xscale=:log10)
    p_inv_logx    = plot(; title="Investment Dev. — Log x", xlabel="Clusters (log)", ylabel="Deviation [%]",         fa..., legend=:best, xscale=:log10)
    p_time_logx   = plot(; title="Solve Time — Log x",     xlabel="Clusters (log)", ylabel="Time [s]",              fa..., legend=:best, xscale=:log10)

    # ── log-y ─────────────────────────────────────────────────────────────────
    p_regret_logy = plot(; title="Regret (%) — Log y",     xlabel="Clusters", ylabel="Regret [%] (log)",       fa..., legend=:best, yscale=:log10)
    p_lole_logy   = plot(; title="LOLE — Log y",           xlabel="Clusters", ylabel="LOLE [h/yr] (log)",      fa..., legend=:best, yscale=:log10)

    # ── zoomed k >= ZOOM_MIN ──────────────────────────────────────────────────
    p_regret_zoom = plot(; title="Regret (%) — k≥$(ZOOM_MIN)",        xlabel="Clusters", ylabel="Regret [%]",       fa..., legend=:best)
    p_lole_zoom   = plot(; title="LOLE — k≥$(ZOOM_MIN)",              xlabel="Clusters", ylabel="LOLE [h/yr]",      fa..., legend=:best)
    p_inv_zoom    = plot(; title="Investment Dev. — k≥$(ZOOM_MIN)",   xlabel="Clusters", ylabel="Deviation [%]",    fa..., legend=:best)

    # ── excluding wc_frac ─────────────────────────────────────────────────────
    p_regret_nofrac     = plot(; title="Regret (%) — excl. wc-frac",         xlabel="Clusters", ylabel="Regret [%]",        fa..., legend=:best)
    p_regret_nofrac_log = plot(; title="Regret (%) — excl. wc-frac, Log y",  xlabel="Clusters", ylabel="Regret [%] (log)",  fa..., legend=:best, yscale=:log10)

    all_r_zoom, all_l_zoom = Float64[], Float64[]
    for m in methods
        r   = results[m]
        xs  = max.(1.0, Float64.(r.clusters))
        idx = findall(k -> k >= ZOOM_MIN, r.clusters)
        add!(p_regret, m, r.clusters, r.regret)
        add!(p_time,   m, r.clusters, r.time)
        add!(p_lole,   m, r.clusters, r.lole)
        add!(p_eens,   m, r.clusters, r.eens)
        add!(p_inv,    m, r.clusters, inv_dev(r))
        add!(p_regret_logx, m, xs, r.regret)
        add!(p_lole_logx,   m, xs, r.lole)
        add!(p_inv_logx,    m, xs, inv_dev(r))
        add!(p_time_logx,   m, xs, r.time)
        add!(p_regret_logy, m, r.clusters, max.(r.regret, 1e-5))
        add!(p_lole_logy,   m, r.clusters, max.(r.lole,   1.0))
        if !isempty(idx)
            add!(p_regret_zoom, m, r.clusters[idx], r.regret[idx])
            add!(p_lole_zoom,   m, r.clusters[idx], r.lole[idx])
            add!(p_inv_zoom,    m, r.clusters[idx], inv_dev(r)[idx])
            append!(all_r_zoom, filter(isfinite, r.regret[idx]))
            append!(all_l_zoom, filter(isfinite, r.lole[idx]))
        end
        if m != :k_medoids_wc_frac
            add!(p_regret_nofrac,     m, r.clusters, r.regret)
            add!(p_regret_nofrac_log, m, r.clusters, max.(r.regret, 1e-5))
        end
    end

    # smart ylims on zoomed plots — clips top 7% so spikes don't compress the rest
    isempty(all_r_zoom) || ylims!(p_regret_zoom, -0.02, quantile(all_r_zoom, 0.93))
    isempty(all_l_zoom) || ylims!(p_lole_zoom,    0.0,  quantile(all_l_zoom, 0.93))

    hline!(p_regret,          [0.0]; label=false, linestyle=:solid, color=:gray, linewidth=1.0)
    hline!(p_regret_logx,     [0.0]; label=false, linestyle=:solid, color=:gray, linewidth=1.0)
    hline!(p_regret_zoom,     [0.0]; label=false, linestyle=:solid, color=:gray, linewidth=1.0)
    hline!(p_regret_nofrac,   [0.0]; label=false, linestyle=:solid, color=:gray, linewidth=1.0)
    hline!(p_regret_logy,     [0.1]; label="0.1%", linestyle=:dash, color=:gray, linewidth=1.0)
    hline!(p_regret_nofrac_log,[0.1]; label="0.1%", linestyle=:dash, color=:gray, linewidth=1.0)
    hline!(p_inv,      [0.0]; label=false, linestyle=:dash, color=:black, linewidth=1.0)
    hline!(p_inv_logx, [0.0]; label=false, linestyle=:dash, color=:black, linewidth=1.0)
    hline!(p_inv_zoom, [0.0]; label=false, linestyle=:dash, color=:black, linewidth=1.0)

    if ref !== nothing
        rpa = (linestyle=:dash, color=:black, linewidth=1.2)
        for p in (p_lole, p_lole_logx, p_lole_logy, p_lole_zoom)
            hline!(p, [ref.lole]; label="Full-Res", rpa...)
            hline!(p, [4.0]; label="TenneT (4 h/yr)", linestyle=:dot, color=:red, linewidth=1.5)
        end
        hline!(p_eens, [ref.eens]; label="Full-Res", rpa...)
        hline!(p_inv,      [0.0]; label="Full-Res Baseline (0%)", rpa...)
        hline!(p_inv_logx, [0.0]; label="Full-Res Baseline (0%)", rpa...)
        hline!(p_inv_zoom, [0.0]; label="Full-Res Baseline (0%)", rpa...)
    end

    return (
        regret=p_regret, time=p_time, lole=p_lole, eens=p_eens, inv=p_inv,
        regret_logx=p_regret_logx, lole_logx=p_lole_logx,
        inv_logx=p_inv_logx, time_logx=p_time_logx,
        regret_logy=p_regret_logy, lole_logy=p_lole_logy,
        regret_zoom=p_regret_zoom, lole_zoom=p_lole_zoom, inv_zoom=p_inv_zoom,
        regret_nofrac=p_regret_nofrac, regret_nofrac_log=p_regret_nofrac_log,
    )
end


# 10. The main call + helpers for saving results

function next_run_dir()
    OUTPUT_DIR == "" && return ""
    mkpath(OUTPUT_DIR)
    i = 1
    while isdir(joinpath(OUTPUT_DIR, "run_$(lpad(i,3,'0'))")); i += 1; end
    d = joinpath(OUTPUT_DIR, "run_$(lpad(i,3,'0'))"); mkpath(d); return d
end

save_fig(fig, name, dir) = dir != "" && savefig(fig, joinpath(dir, "$name.png"))

function save_csv(results, run_dir)
    run_dir == "" && return
    for m in methods
        r = results[m]
        open(joinpath(run_dir, "$(m)_results.csv"), "w") do io
            println(io, "clusters,cost,investment,regret,lole_full_h,eens_full_mwh,time_s")
            for i in eachindex(r.clusters)
                println(io, join((r.clusters[i], r.cost[i], r.inv[i], r.regret[i],
                                  r.lole[i], r.eens[i], r.time[i]), ","))
            end
        end
    end
end

function save_benchmark_csv(bench_conn, bench_cost, n_periods, run_dir,
                             bench_inv=NaN, bench_lole=NaN, bench_eens=NaN)
    (run_dir == "" || bench_conn === nothing) && return
    open(joinpath(run_dir, "benchmark_full_res.csv"), "w") do io
        println(io, "clusters,cost,investment,lole_full_h,eens_full_mwh")
        println(io, "$n_periods,$bench_cost,$bench_inv,$bench_lole,$bench_eens")
    end
end

function main()
    Random.seed!(42)
    run_dir = next_run_dir()

    probe     = load_connection()
    n_scenarios = (DuckDB.query(probe,
        "SELECT COUNT(DISTINCT scenario) AS n FROM profiles") |> DataFrame).n[1]
    max_ts      = (DuckDB.query(probe,
        "SELECT MAX(timestep) AS m FROM profiles") |> DataFrame).m[1]
    n_periods   = ceil(Int, n_scenarios * max_ts / period_duration)
    DBInterface.close!(probe)

    wc_conn = load_connection()
    wc_df   = compute_elgersma_wc(wc_conn)
    DBInterface.close!(wc_conn)

    score_conn        = load_connection()
    cap_weights       = load_capacity_weights(score_conn)
    netload_score_df  = compute_period_net_loads(score_conn,      cap_weights)
    elgersma_score_df = compute_period_elgersma_scores(score_conn)
    DBInterface.close!(score_conn)

    max_k = max_clusters <= 0 ? n_periods : min(max_clusters, n_periods)
    steps = Int[]
    current_k = min_clusters
    for (phase_limit, p_step) in step_sizes
        # Cap the phase limit at max_k so 99999 cleanly scales down to your actual maximum
        limit = min(phase_limit, max_k)
        if current_k <= limit
            append!(steps, collect(current_k:p_step:limit))
            steps[end] >= max_k && break
            current_k = steps[end] + p_step
        end
    end
    unique!(filter!(k -> k <= max_k, steps))
    sort!(steps)

    bench_prob, bench_cost, bench_conn = nothing, NaN, nothing
    if do_regret
        bench_conn = load_connection()
        TC.cluster!(bench_conn, period_duration, n_periods; method=baseline_method, layout = LAYOUT)
        fix_milestone_year!(bench_conn)
        TEM.populate_with_defaults!(bench_conn)
        bench_prob = with_logger(NullLogger()) do
            TEM.run_scenario(bench_conn; optimizer=OPTIMIZER,
                             optimizer_parameters=OPT_PARAMS, show_log=false)
        end
        bench_cost = JuMP.objective_value(bench_prob.model)
        println("Benchmark (full-res, $baseline_method): cost = $(round(bench_cost, digits=2))")
    end

    bench_inv  = bench_conn !== nothing ? investment_cost(bench_conn) : NaN
    bench_lole, bench_eens =
        bench_conn !== nothing ? compute_lole(bench_conn) : (NaN, NaN)

    ref = bench_conn !== nothing ?
        (inv  = bench_inv,
         lole = bench_lole,
         eens = bench_eens) : nothing

    all_results = [init_results() for _ in 1:n_seeds]
    completed, interrupted = 0, false
    for s in 1:n_seeds
        Random.seed!(s)
        status = run_experiment!(
            steps, all_results[s], wc_df;
            bench_prob,
            bench_conn,
            bench_cost,
            netload_score_df,
            elgersma_score_df,
        )
        completed = s
        if status == :interrupted; interrupted = true; break; end
    end
    completed == 0 && (println("No seeds completed."); return nothing)

    results = init_results()
    for m in methods
        append!(results[m].clusters, all_results[1][m].clusters)
        for field in (:cost, :inv, :regret, :lole, :eens, :time)
            vals = completed == 1 ?
                getproperty(all_results[1][m], field) :
                vec(mean(hcat([getproperty(all_results[s][m], field)
                               for s in 1:completed]...), dims=2))
            append!(getproperty(results[m], field), vals)
        end
    end
    plts = build_plots(results; ref)

    combined        = plot(plts.regret, plts.time,       plts.lole,
                           plts.eens,   plts.inv;
                           layout=(2,3), size=(1650,800), margin=6Plots.mm, dpi=300)
    combined_logx   = plot(plts.regret_logx,  plts.time_logx, plts.lole_logx,
                           plts.inv_logx;
                           layout=(2,2), size=(1400,800), margin=6Plots.mm, dpi=300)
    combined_detail = plot(plts.regret_zoom,  plts.regret_logy, plts.lole_zoom,
                           plts.lole_logy,    plts.inv_zoom;
                           layout=(2,3), size=(1650,800), margin=6Plots.mm, dpi=300)
    combined_nofrac = plot(plts.regret_nofrac, plts.regret_nofrac_log;
                           layout=(1,2), size=(1200,500), margin=6Plots.mm, dpi=300)
    display(combined)

    for (fig, name) in (
        (combined,              "combined"),
        (combined_logx,         "combined_logx"),
        (combined_detail,       "combined_detail"),
        (combined_nofrac,       "combined_nofrac"),
        (plts.regret,           "regret"),
        (plts.time,             "time"),
        (plts.lole,             "lole"),
        (plts.eens,             "eens"),
        (plts.inv,              "investment"),
        (plts.regret_logx,      "regret_logx"),
        (plts.lole_logx,        "lole_logx"),
        (plts.inv_logx,         "investment_logx"),
        (plts.time_logx,        "time_logx"),
        (plts.regret_logy,      "regret_logy"),
        (plts.lole_logy,        "lole_logy"),
        (plts.regret_zoom,      "regret_zoom"),
        (plts.lole_zoom,        "lole_zoom"),
        (plts.inv_zoom,         "investment_zoom"),
        (plts.regret_nofrac,    "regret_nofrac"),
        (plts.regret_nofrac_log,"regret_nofrac_logy"),
    )
        save_fig(fig, name, run_dir)
    end
    save_csv(results, run_dir)
    save_benchmark_csv(bench_conn, bench_cost, n_periods, run_dir,
                       bench_inv, bench_lole, bench_eens)
    bench_conn !== nothing && DBInterface.close!(bench_conn)
    interrupted ? @warn("Interrupted – partial results saved to $run_dir") :
                  println("Done – results saved to $run_dir")
    return results
end

main()