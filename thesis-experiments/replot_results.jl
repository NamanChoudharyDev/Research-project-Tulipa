using Pkg
Pkg.activate(@__DIR__)
using CSV, DataFrames, Plots, Statistics

# ── Paths ──────────────────────────────────────────────────────────────────────
const RUN_DIR      = joinpath(@__DIR__, "data", "tutorial-9", "results", "5-seeded run")
const OUT_MAIN     = joinpath(RUN_DIR, "publication_plots_5seed", "main")
const OUT_APPENDIX = joinpath(RUN_DIR, "publication_plots_5seed", "appendix")
mkpath(OUT_MAIN)
mkpath(OUT_APPENDIX)

# ── Method catalogue ───────────────────────────────────────────────────────────
const METHODS = [
    (id=:k_medoids,              label="k-Medoids",           color=:black,        stroke=:solid,   marker=:circle,    lw=2.0, ms=4.5),
    (id=:k_medoids_wc_unit,      label="WC unit weight",      color=:royalblue,    stroke=:dash,    marker=:square,    lw=2.0, ms=4.5),
    (id=:k_medoids_wc_frac,      label="WC frac. weight",     color=:mediumpurple, stroke=:dot,     marker=:diamond,   lw=2.0, ms=4.5),
    (id=:k_medoids_rwc_netload,  label="Real Net-Load",       color=:darkorange,   stroke=:solid,   marker=:utriangle, lw=2.0, ms=4.5),
    (id=:k_medoids_rwc_elgersma, label="Real Elgersma-Score", color=:crimson,      stroke=:dash,    marker=:dtriangle, lw=2.0, ms=4.5),
    (id=:k_medoids_rwc_ens,      label="ENS-Guided",          color=:seagreen,     stroke=:dashdot, marker=:star5,     lw=2.0, ms=4.5),
]

# The two baselines that appear in every contributed-method comparison plot
const BASELINES  = [:k_medoids, :k_medoids_wc_unit]

# The three contributed real worst-case methods with their short names (used in filenames)
const REAL_METHODS = [
    (:k_medoids_rwc_netload,  "netload"),
    (:k_medoids_rwc_elgersma, "elgersma"),
    (:k_medoids_rwc_ens,      "ens"),
]

# For the fractional-weight group (3 lines: both baselines + wc_frac)
const FRAC_GROUP = [:k_medoids, :k_medoids_wc_unit, :k_medoids_wc_frac]

# ── Data loading ───────────────────────────────────────────────────────────────
function load_all()
    d = Dict{Symbol, DataFrame}()
    for m in METHODS
        p = joinpath(RUN_DIR, "$(m.id)_results.csv")
        isfile(p) && (d[m.id] = CSV.read(p, DataFrame))
    end
    return d
end

function load_bench()
    p = joinpath(RUN_DIR, "benchmark_full_res.csv")
    isfile(p) || return nothing
    df = CSV.read(p, DataFrame)
    return (cost=df.cost[1], inv=df.investment[1],
            lole=df.lole_full_h[1], eens=df.eens_full_mwh[1])
end

const DATA  = load_all()
const BENCH = load_bench()

# ── Shared theme ───────────────────────────────────────────────────────────────
const FA = (
    titlefontsize=11, guidefontsize=10, tickfontsize=9,
    legendfontsize=8, grid=true, gridalpha=0.12, gridstyle=:dot,
    frame=:box, tickdirection=:out, dpi=300,
)

# ── Helpers ────────────────────────────────────────────────────────────────────
function inv_dev_vec(df)
    BENCH === nothing && return Float64.(df.investment)
    [(isfinite(v) ? (v - BENCH.inv)/BENCH.inv*100.0 : NaN) for v in df.investment]
end

function slice_range(df, lo, hi)
    idx = findall(k -> lo <= k <= hi, df.clusters)
    (Float64.(df.clusters[idx]), idx)
end

function smart_hi(vals; pct=0.95)
    f = filter(isfinite, vals)
    isempty(f) ? NaN : quantile(f, pct)
end

save_main(fig, name)     = savefig(fig, joinpath(OUT_MAIN,     "$name.png"))
save_appendix(fig, name) = savefig(fig, joinpath(OUT_APPENDIX, "$name.png"))

function add_method!(p, m, xs, ys)
    plot!(p, xs, ys;
        label=m.label, color=m.color, linestyle=m.stroke, marker=m.marker,
        linewidth=m.lw, markersize=m.ms, markerstrokewidth=0,
        alpha=0.85, markeralpha=0.5)
end

function method_by_id(id)
    i = findfirst(m -> m.id == id, METHODS)
    i === nothing && error("Unknown method id: $id")
    METHODS[i]
end

# ── Core: plot one group of IDs for a given metric and k-range ────────────────
function make_plot(ids, ycol::Symbol, title_str, ylabel_str, lo, hi;
                   legend_pos=:topright,
                   ref_hlines=[],
                   smart_pct=0.95,
                   ylo_override=nothing)
    p = plot(; title=title_str, xlabel="k", ylabel=ylabel_str,
               FA..., legend=legend_pos,
               legendbackgroundcolor=RGBA(1,1,1,0.7))
    for (val, lbl, col, ls) in ref_hlines
        hline!(p, [val]; label=lbl, linestyle=ls, color=col, linewidth=1.5)
    end
    ys_acc = Float64[]
    for id in ids
        m = method_by_id(id)
        haskey(DATA, id) || continue
        df = DATA[id]
        xs, idx = slice_range(df, lo, hi)
        isempty(idx) && continue
        ys = if ycol == :inv_dev
                 inv_dev_vec(df)[idx]
             else
                 Float64.(df[!, ycol][idx])
             end
        append!(ys_acc, filter(isfinite, ys))
        add_method!(p, m, xs, ys)
    end
    hi_y = smart_hi(ys_acc; pct=smart_pct)
    lo_y = ylo_override !== nothing ? ylo_override :
           (ycol == :inv_dev ? (isempty(ys_acc) ? -15.0 :
               let fv = filter(isfinite, ys_acc); isempty(fv) ? -15.0 : minimum(fv)*1.15 end) :
           0.0)
    isnan(hi_y) || ylims!(p, lo_y, hi_y)
    xlims!(p, lo*0.97, hi*1.02)
    return p
end

# ════════════════════════════════════════════════════════════════════════════════
# MAIN PAPER PLOTS
# Each real worst-case method plotted against its two baselines only (3 lines).
# ════════════════════════════════════════════════════════════════════════════════

println("=== Generating main plots ===")

# ── Regret: per-method, two zoom ranges ───────────────────────────────────────
# Produces 6 files:
#   netload_regret_mid.png   elgersma_regret_mid.png   ens_regret_mid.png
#   netload_regret_late.png  elgersma_regret_late.png  ens_regret_late.png
for (rid, rname) in REAL_METHODS
    ids = vcat(BASELINES, [rid])

    p_mid = make_plot(ids, :regret,
        "Regret (%) — $(method_by_id(rid).label) (k 50-252)",
        "Regret [%]", 50, 252;
        ref_hlines=[(0.0, false, :gray, :dash)])
    save_main(p_mid, "$(rname)_regret_mid")
    println("Saved: $(rname)_regret_mid.png")

    p_late = make_plot(ids, :regret,
        "Regret (%) — $(method_by_id(rid).label) (k 202-1002)",
        "Regret [%]", 202, 1002;
        ref_hlines=[(0.0, false, :gray, :dash)])
    save_main(p_late, "$(rname)_regret_late")
    println("Saved: $(rname)_regret_late.png")
end

# ── Fractional-weight failure: regret and investment (already 3 lines) ────────
p_frac_r = make_plot(FRAC_GROUP, :regret,
    "Regret (%) — WC frac. weight vs baselines (k 50-1002)",
    "Regret [%]", 50, 1002;
    ref_hlines=[(0.0, false, :gray, :dash)],
    smart_pct=0.99)
save_main(p_frac_r, "wc_frac_regret")
println("Saved: wc_frac_regret.png")

p_frac_i = make_plot(FRAC_GROUP, :inv_dev,
    "Investment deviation — WC frac. weight vs baselines (k 50-1002)",
    "Deviation from benchmark [%]", 50, 1002;
    ref_hlines=[(0.0, false, :gray, :dash)],
    smart_pct=0.99)
save_main(p_frac_i, "wc_frac_inv")
println("Saved: wc_frac_inv.png")

# ── LOLE: per-method, medium/high k ───────────────────────────────────────────
# Produces 3 files:
#   netload_lole_late.png   elgersma_lole_late.png   ens_lole_late.png
lole_refs = BENCH === nothing ? [] :
    [(BENCH.lole, "Full-res ($(Int(BENCH.lole)) h/yr)", :black, :dash)]

for (rid, rname) in REAL_METHODS
    ids = vcat(BASELINES, [rid])
    p_l = make_plot(ids, :lole_full_h,
        "LOLE — $(method_by_id(rid).label) (k 252-852)",
        "LOLE [h/yr]", 252, 852;
        ref_hlines=lole_refs,
        smart_pct=0.97,
        ylo_override=290.0)
    save_main(p_l, "$(rname)_lole_late")
    println("Saved: $(rname)_lole_late.png")
end

# ── ENS-Guided regret vs baselines (already 3 lines, k=50-402) ───────────────
p_ens_r = make_plot([:k_medoids, :k_medoids_wc_unit, :k_medoids_rwc_ens],
    :regret,
    "Regret (%) — ENS-Guided vs baselines (k 50-402)",
    "Regret [%]", 50, 402;
    ref_hlines=[(0.0, false, :gray, :dash)])
save_main(p_ens_r, "ens_regret_zoom")
println("Saved: ens_regret_zoom.png")

# ── Solve time: all 6 methods (time plot, 6 lines is fine here) ───────────────
begin
    p = plot(; title="Solve time per k-value — all methods",
               xlabel="k", ylabel="Time [s]",
               FA..., legend=:topleft,
               legendbackgroundcolor=RGBA(1,1,1,0.7))
    for m in METHODS
        haskey(DATA, m.id) || continue
        df = DATA[m.id]
        plot!(p, Float64.(df.clusters), Float64.(df.time_s);
            label=m.label, color=m.color, linestyle=m.stroke, marker=m.marker,
            linewidth=m.lw, markersize=m.ms, markerstrokewidth=0,
            alpha=0.85, markeralpha=0.5)
    end
    xlims!(p, 0, 1100)
    save_main(p, "time_all")
    println("Saved: time_all.png")
end

println("\n=== Main plots done ($(length(readdir(OUT_MAIN)))) ===\n")

# ════════════════════════════════════════════════════════════════════════════════
# APPENDIX PLOTS — full range + zooms, 3 lines per plot (method + 2 baselines)
# ════════════════════════════════════════════════════════════════════════════════

println("=== Generating appendix plots ===")

const APP_R_EARLY = (2,   50)
const APP_R_LATE  = (50,  252)
const APP_L_EARLY = (2,   50)
const APP_L_MID   = (50,  152)
const APP_L_LATE  = (152, 702)
const APP_I_EARLY = (2,   50)
const APP_I_LATE  = (50,  402)

lole_refs_full = BENCH === nothing ? [] : [
    (BENCH.lole, "Full-res ($(Int(BENCH.lole)) h/yr)", :black, :dash),
    (4.0,        "TenneT (4 h/yr)",                    :red,   :dot),
]

function build_appendix_group(ids, group_name)
    # ── regret ────────────────────────────────────────────────────────────────
    p_rf = make_plot(ids, :regret,
        "Regret (%) — $(group_name) (full range)", "Regret [%]", 2, 1095;
        ref_hlines=[(0.0, false, :gray, :dash)], smart_pct=0.95)
    p_re = make_plot(ids, :regret,
        "Regret (%) — $(group_name) (k $(APP_R_EARLY[1])–$(APP_R_EARLY[2]))",
        "Regret [%]", APP_R_EARLY...;
        ref_hlines=[(0.0, false, :gray, :dash)], smart_pct=0.93)
    p_rl = make_plot(ids, :regret,
        "Regret (%) — $(group_name) (k $(APP_R_LATE[1])–$(APP_R_LATE[2]))",
        "Regret [%]", APP_R_LATE...;
        ref_hlines=[(0.0, false, :gray, :dash)])
    save_appendix(p_rf, "$(group_name)_regret_full")
    save_appendix(p_re, "$(group_name)_regret_early")
    save_appendix(p_rl, "$(group_name)_regret_late")

    # ── LOLE ──────────────────────────────────────────────────────────────────
    p_lf = make_plot(ids, :lole_full_h,
        "LOLE — $(group_name) (full range)", "LOLE [h/yr]", 2, 1095;
        ref_hlines=lole_refs_full, smart_pct=0.93)
    p_le = make_plot(ids, :lole_full_h,
        "LOLE — $(group_name) (k $(APP_L_EARLY[1])–$(APP_L_EARLY[2]))",
        "LOLE [h/yr]", APP_L_EARLY...;
        ref_hlines=lole_refs_full, smart_pct=0.93)
    p_lm = make_plot(ids, :lole_full_h,
        "LOLE — $(group_name) (k $(APP_L_MID[1])–$(APP_L_MID[2]))",
        "LOLE [h/yr]", APP_L_MID...;
        ref_hlines=lole_refs_full)
    p_ll = make_plot(ids, :lole_full_h,
        "LOLE — $(group_name) (k $(APP_L_LATE[1])–$(APP_L_LATE[2]))",
        "LOLE [h/yr]", APP_L_LATE...;
        ref_hlines=lole_refs_full)
    save_appendix(p_lf, "$(group_name)_lole_full")
    save_appendix(p_le, "$(group_name)_lole_early")
    save_appendix(p_lm, "$(group_name)_lole_mid")
    save_appendix(p_ll, "$(group_name)_lole_late")

    # ── investment deviation ──────────────────────────────────────────────────
    p_if = make_plot(ids, :inv_dev,
        "Investment deviation — $(group_name) (full range)", "Deviation [%]", 2, 1095;
        ref_hlines=[(0.0, false, :gray, :dash)], smart_pct=0.95)
    p_ie = make_plot(ids, :inv_dev,
        "Investment deviation — $(group_name) (k $(APP_I_EARLY[1])–$(APP_I_EARLY[2]))",
        "Deviation [%]", APP_I_EARLY...;
        ref_hlines=[(0.0, false, :gray, :dash)], smart_pct=0.93)
    p_il = make_plot(ids, :inv_dev,
        "Investment deviation — $(group_name) (k $(APP_I_LATE[1])–$(APP_I_LATE[2]))",
        "Deviation [%]", APP_I_LATE...;
        ref_hlines=[(0.0, false, :gray, :dash)])
    save_appendix(p_if, "$(group_name)_inv_full")
    save_appendix(p_ie, "$(group_name)_inv_early")
    save_appendix(p_il, "$(group_name)_inv_late")

    println("Saved appendix group: $(group_name)  (10 PNGs)")
end

build_appendix_group(vcat(BASELINES, [:k_medoids_rwc_netload]),  "netload")
build_appendix_group(vcat(BASELINES, [:k_medoids_rwc_elgersma]), "elgersma")
build_appendix_group(vcat(BASELINES, [:k_medoids_rwc_ens]),      "ens")
build_appendix_group(vcat(BASELINES, [:k_medoids_wc_frac]),      "wc_frac")

# ── Appendix: full solve time (all 6 methods) ─────────────────────────────────
begin
    p = plot(; title="Solve time — all methods (full range)", xlabel="k", ylabel="Time [s]",
               FA..., legend=:topleft, legendbackgroundcolor=RGBA(1,1,1,0.7))
    for m in METHODS
        haskey(DATA, m.id) || continue
        df = DATA[m.id]
        plot!(p, Float64.(df.clusters), Float64.(df.time_s);
            label=m.label, color=m.color, linestyle=m.stroke, marker=m.marker,
            linewidth=m.lw, markersize=m.ms, markerstrokewidth=0,
            alpha=0.85, markeralpha=0.5)
    end
    xlims!(p, 0, 1100)
    save_appendix(p, "time_full")
    println("Saved: time_full.png (appendix)")
end

println("\nAll done.")
println("Main plots     ($(length(readdir(OUT_MAIN))))    → $(OUT_MAIN)")
println("Appendix plots ($(length(readdir(OUT_APPENDIX)))) → $(OUT_APPENDIX)")