#   :k_medoids_rwc_netload
#       Select top-n_wc periods by highest capacity-weighted net load.
#       Net load(p) = Σ_h [ demand(p,h) − Σ_g capacity_g × availability_g(p,h) ]
#       Sums residual demand across all 24 hours. Higher = more total stress.
#
#   :k_medoids_rwc_elgersma
#       Select top-n_wc periods by the per-generator Elgersma residual score:
#
#       Step 1 — peak demand in period:
#           D_max(p) = max_{h ∈ p}( demand(p,h) )
#
#       Step 2 — worst renewable ratio per generator (mirrors Elgersma eq. 2):
#           γ_g(p) = min_{h ∈ p}( A_g(p,h) / demand(p,h) )
#
#       Step 3 — worst-case availability at peak demand (mirrors Elgersma eq. 3):
#           A_g_worst(p) = D_max(p) × γ_g(p) × cap_g
#
#       Score(p) = D_max(p) − Σ_g A_g_worst(p)
#       Higher score → harder period.
#
#   :k_medoids_rwc_ens       — Algorithm F
#       Phase 1: run k medoids → solve → find rep_period with highest total ENS →
#       among mapped periods pick p* = highest net load.
#       Phase 2: run k−1 medoids → append p* as its own representative → solve.
#       Zero ENS: return pilot result directly.


# 0. Configurable parameters
const N_WC_REAL = 1


# 1. Capacity weight helper

"""
Returns Dict{profile_name → capacity} for availability/inflows profiles only.
"""
function load_capacity_weights(c)
    df = DuckDB.query(c, """
        SELECT ap.profile_name,
               a.capacity
        FROM   assets_profiles ap
        JOIN   asset            a  ON ap.asset = a.asset
        WHERE  ap.profile_type IN ('availability', 'inflows')
    """) |> DataFrame
    return Dict(row.profile_name => Float64(row.capacity) for row in eachrow(df))
end


# 2. Period scoring helpers

"""
Computes the net load score for every period:
    net_load_score(p) = Σ_h [ demand(p,h) − Σ_g capacity_g × availability_g(p,h) ]
Higher score → higher stress.
"""
function compute_period_net_loads(c, capacity_weights)
    cap_rows = join(["('$(k)', $(v))" for (k,v) in capacity_weights], ", ")
    return DuckDB.query(c, """
        WITH scen_rank AS (
            SELECT scenario,
                   ROW_NUMBER() OVER (ORDER BY scenario) - 1 AS sr
            FROM (SELECT DISTINCT scenario FROM profiles)
        ),
        max_ts AS (SELECT MAX(timestep) AS m FROM profiles),
        demand_agg AS (
            SELECT CAST(CEIL((sr.sr * mt.m + p.timestep) / $(period_duration).0)
                        AS INTEGER)          AS period,
                   SUM(p.value)              AS total_demand
            FROM profiles p
            JOIN scen_rank sr ON p.scenario = sr.scenario
            CROSS JOIN max_ts mt
            WHERE p.profile_name = 'demand'
            GROUP BY period
        ),
        avail_agg AS (
            SELECT CAST(CEIL((sr.sr * mt.m + p.timestep) / $(period_duration).0)
                        AS INTEGER)                              AS period,
                   SUM(p.value * cw.capacity)                   AS total_avail
            FROM profiles p
            JOIN scen_rank sr ON p.scenario = sr.scenario
            CROSS JOIN max_ts mt
            JOIN (VALUES $cap_rows) AS cw(profile_name, capacity)
              ON p.profile_name = cw.profile_name
            WHERE p.profile_name != 'demand'
            GROUP BY period
        )
        SELECT d.period,
               d.total_demand - COALESCE(a.total_avail, 0.0) AS net_load_score
        FROM   demand_agg d
        LEFT JOIN avail_agg a ON d.period = a.period
        ORDER  BY d.period
    """) |> DataFrame
end

"""
Computes the per-generator Elgersma residual score for every period.
    D_max(p)     = max_{h ∈ p}( demand(p,h) )
    γ_g(p)       = min_{h ∈ p}( A_g(p,h) / demand(p,h) )   [per generator]
    A_g_worst(p) = D_max(p) × γ_g(p) × cap_g
    Score(p)     = D_max(p) − Σ_g A_g_worst(p)
demand = 0 rows excluded from ratio computation to avoid division by zero.
"""
function compute_period_elgersma_scores(c)
    return DuckDB.query(c, """
        WITH scen_rank AS (
            SELECT scenario,
                   ROW_NUMBER() OVER (ORDER BY scenario) - 1 AS sr
            FROM (SELECT DISTINCT scenario FROM profiles)
        ),
        max_ts AS (SELECT MAX(timestep) AS m FROM profiles),
        with_period AS (
            SELECT p.scenario,
                   p.timestep,
                   p.profile_name,
                   p.value,
                   CAST(CEIL((sr.sr * mt.m + p.timestep) / $(period_duration).0)
                        AS INTEGER) AS period
            FROM profiles p
            JOIN scen_rank sr ON p.scenario = sr.scenario
            CROSS JOIN max_ts mt
        ),
        d AS (
            SELECT period, timestep, scenario, value AS dval
            FROM with_period WHERE profile_name = 'demand'
        ),
        max_d AS (
            SELECT period, MAX(dval) AS max_demand FROM d GROUP BY period
        ),
        min_ratio AS (
            SELECT a.period,
                   MIN(a.value / NULLIF(d.dval, 0.0)) AS min_r
            FROM with_period a
            JOIN d ON a.period = d.period
                  AND a.timestep = d.timestep
                  AND a.scenario = d.scenario
            WHERE a.profile_name != 'demand' AND d.dval > 0.0
            GROUP BY a.period, a.profile_name
        ),
        worst_avail AS (
            SELECT mr.period,
                   SUM(md.max_demand * mr.min_r) AS total_worst
            FROM min_ratio mr
            JOIN max_d md ON mr.period = md.period
            GROUP BY mr.period
        )
        SELECT md.period,
               md.max_demand - COALESCE(wa.total_worst, 0.0) AS elgersma_score
        FROM   max_d md
        LEFT JOIN worst_avail wa ON md.period = wa.period
        ORDER  BY md.period
    """) |> DataFrame
end


# 3. Table helpers

function _write_rwc_table!(c, df, table_name)
    tmp = "t_rwc_$(table_name)"
    DuckDB.register_data_frame(c, df, tmp)
    DuckDB.execute(c, "CREATE OR REPLACE TABLE $table_name AS FROM $tmp")
    DuckDB.execute(c, "DROP VIEW $tmp")
end

"""
Handles table construction for the k = 1 case of the real worst-case method.
Every real period maps to the single worst real period with weight 1.0.
"""
function create_rwc_tables_k1!(c, global_period_idx, n_periods, yr)
    n_ts      = (DuckDB.query(c,
        "SELECT MAX(timestep) AS m FROM profiles") |> DataFrame).m[1]
    n_local   = ceil(Int, n_ts / period_duration)
    scen_list = sort((DuckDB.query(c,
        "SELECT DISTINCT scenario FROM profiles ORDER BY scenario") |> DataFrame).scenario)
    scen_rank = div(global_period_idx - 1, n_local)
    local_p   = mod(global_period_idx - 1, n_local) + 1
    scen_val  = scen_list[scen_rank + 1]

    # Fetch the 24 hourly rows for this specific scenario and local period
    raw = DuckDB.query(c, """
        SELECT (((timestep - 1) % $(period_duration)) + 1) AS local_ts,
               profile_name,
               value
        FROM   profiles
        WHERE  CAST(CEIL(timestep / $(period_duration).0) AS INTEGER) = $local_p
          AND  scenario = $scen_val
    """) |> DataFrame

    rpp = DataFrame(
        rep_period     = fill(1,  nrow(raw)),
        timestep       = raw.local_ts,
        milestone_year = fill(yr, nrow(raw)),
        profile_name   = raw.profile_name,
        value          = raw.value,
    )
    _write_rwc_table!(c, rpp, "profiles_rep_periods")

    _write_rwc_table!(c,
        DataFrame(
            milestone_year = yr,
            rep_period     = 1,
            num_timesteps  = period_duration,
            resolution     = 1.0,
        ),
        "rep_periods_data")

    mapping_df = DuckDB.query(c, """
        SELECT DISTINCT
            $yr                                                     AS milestone_year,
            scenario,
            CAST(CEIL(timestep / $(period_duration).0) AS INTEGER) AS period,
            1                                                       AS rep_period,
            1.0                                                     AS weight
        FROM profiles
        ORDER BY scenario, period
    """) |> DataFrame
    _write_rwc_table!(c, mapping_df, "rep_periods_mapping")

    timeframe_df = DuckDB.query(c, """
        SELECT DISTINCT
            $yr                                                     AS milestone_year,
            scenario,
            CAST(CEIL(timestep / $(period_duration).0) AS INTEGER) AS period,
            $period_duration                                        AS num_timesteps
        FROM profiles
        ORDER BY scenario, period
    """) |> DataFrame
    _write_rwc_table!(c, timeframe_df, "timeframe_data")
end


"""
Redirects mapping rows for a globally-indexed period to new rep_period slots.
"""
function append_real_periods_to_tables!(c, global_period_indices, k_base, yr)
    # Resolve scenario and local period from global index
    n_ts     = (DuckDB.query(c, "SELECT MAX(timestep) AS m FROM profiles") |> DataFrame).m[1]
    n_local  = ceil(Int, n_ts / period_duration)

    scenario_list = sort((DuckDB.query(c,
        "SELECT DISTINCT scenario FROM profiles ORDER BY scenario") |> DataFrame).scenario)

    for (i, global_p) in enumerate(global_period_indices)
        new_rp        = k_base + i
        scen_rank     = div(global_p - 1, n_local)
        local_period  = mod(global_p - 1, n_local) + 1
        scen_val      = scenario_list[scen_rank + 1]

        DuckDB.execute(c, """
            UPDATE rep_periods_mapping
            SET    rep_period = $new_rp
            WHERE  period         = $local_period
              AND  scenario       = $scen_val
              AND  milestone_year = $yr
        """)

        raw = DuckDB.query(c, """
            SELECT (((timestep - 1) % $(period_duration)) + 1) AS local_ts,
                profile_name,
                value
            FROM   profiles
            WHERE  CAST(CEIL(timestep / $(period_duration).0) AS INTEGER) = $local_period
              AND  scenario = $scen_val
        """) |> DataFrame

        rpp_new = DataFrame(
            rep_period     = fill(new_rp, nrow(raw)),
            timestep       = raw.local_ts,
            milestone_year = fill(yr,     nrow(raw)),
            profile_name   = raw.profile_name,
            value          = raw.value,
        )
        tmp = "t_rwc_rpp_new_$(new_rp)"
        DuckDB.register_data_frame(c, rpp_new, tmp)
        DuckDB.execute(c, "INSERT INTO profiles_rep_periods SELECT * FROM $tmp")
        DuckDB.execute(c, "DROP VIEW $tmp")

        DuckDB.execute(c, """
            INSERT INTO rep_periods_data (milestone_year, rep_period, num_timesteps, resolution)
            VALUES ($yr, $new_rp, $period_duration, 1.0)
        """)
    end
end


# 4. run_once_rwc

"""
score_col  — :net_load_score or :elgersma_score; highest = worst-case.
k ≤ n_wc   → tables built from scratch with worst real period as sole representative.
k > n_wc   → TC.cluster! on k−n_wc medoids, top-n_wc real periods appended.
"""
function run_once_rwc(
    k,
    score_df,
    score_col::Symbol;
    n_wc           = N_WC_REAL,
    return_problem = false,
)
    c  = load_connection()
    t0 = time_ns()
    yr = isempty(target_years) ? 0 : target_years[1]
    n_sc      = (DuckDB.query(c,
        "SELECT COUNT(DISTINCT scenario) AS n FROM profiles") |> DataFrame).n[1]
    n_ts      = (DuckDB.query(c,
        "SELECT MAX(timestep) AS m FROM profiles") |> DataFrame).m[1]
    n_periods = ceil(Int, n_sc * n_ts / period_duration)

    n_select = min(n_wc, nrow(score_df))
    sorted   = sort(score_df, score_col; rev=true)
    selected = sorted.period[1:n_select]

    if k <= n_select
        create_rwc_tables_k1!(c, selected[1], n_periods, yr)
    else
        k_base = k - n_select
        TC.cluster!(c, period_duration, k_base; method=:k_medoids, layout = LAYOUT)
        fix_milestone_year!(c)
        append_real_periods_to_tables!(c, selected, k_base, yr)
    end

    TEM.populate_with_defaults!(c)
    prob = try
        with_logger(NullLogger()) do
            TEM.run_scenario(c; optimizer=OPTIMIZER,
                             optimizer_parameters=OPT_PARAMS, show_log=false)
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


# 5. run_once_rwc_ens

"""
Two-phase model-guided real worst-case selection.
Phase 1: run k medoids → solve → find rep with highest ENS → pick p* = max net load in that cluster.
Phase 2: run k−1 medoids → append p* → solve again.
Zero ENS: return pilot result directly. Elapsed time includes both solves.
"""
function run_once_rwc_ens(k; return_problem = false)
    yr = isempty(target_years) ? 0 : target_years[1]
    t0 = time_ns()

    # Phase 1: pilot solve
    c_pilot   = load_connection()
    n_sc_pilot = (DuckDB.query(c_pilot,
        "SELECT COUNT(DISTINCT scenario) AS n FROM profiles") |> DataFrame).n[1]
    n_ts_pilot = (DuckDB.query(c_pilot,
        "SELECT MAX(timestep) AS m FROM profiles") |> DataFrame).m[1]
    n_periods  = ceil(Int, n_sc_pilot * n_ts_pilot / period_duration)
    n_local    = ceil(Int, n_ts_pilot / period_duration)

    cap_weights = load_capacity_weights(c_pilot)

    TC.cluster!(c_pilot, period_duration, k; method=:k_medoids, layout = LAYOUT)
    fix_milestone_year!(c_pilot)
    TEM.populate_with_defaults!(c_pilot)

    pilot_prob = try
        with_logger(NullLogger()) do
            TEM.run_scenario(c_pilot; optimizer=OPTIMIZER,
                             optimizer_parameters=OPT_PARAMS, show_log=false)
        end
    catch
        elapsed = (time_ns() - t0) / 1e9
        DBInterface.close!(c_pilot)
        nan = (cost=NaN, inv=NaN, lole=NaN, eens=NaN, time=elapsed)
        return return_problem ? merge(nan, (problem=nothing,)) : nan
    end

    ens_by_rp = DuckDB.query(c_pilot, """
        SELECT rep_period,
               SUM(solution * (time_block_end - time_block_start + 1)) AS total_ens
        FROM   var_flow
        WHERE  lower(from_asset) LIKE '%ens%'
          AND  solution > 0
        GROUP  BY rep_period
        ORDER  BY total_ens DESC
        LIMIT  1
    """) |> DataFrame

    # Zero ENS: system feasible, return pilot result directly
    if nrow(ens_by_rp) == 0
        elapsed = (time_ns() - t0) / 1e9
        inv     = investment_cost(c_pilot)
        DBInterface.close!(c_pilot)
        result = (cost=pilot_prob.objective_value, inv=inv, lole=NaN, eens=NaN, time=elapsed)
        return return_problem ? merge(result, (problem=pilot_prob,)) : result
    end

    worst_rp = ens_by_rp.rep_period[1]

    mapped_periods = DuckDB.query(c_pilot, """
        SELECT period, scenario
        FROM   rep_periods_mapping
        WHERE  rep_period     = $worst_rp
          AND  milestone_year = $yr
    """) |> DataFrame

    scen_list_pilot = sort((DuckDB.query(c_pilot,
        "SELECT DISTINCT scenario FROM profiles ORDER BY scenario") |> DataFrame).scenario)
    mapped_global = Set(
        (findfirst(==(row.scenario), scen_list_pilot) - 1) * n_local + row.period
        for row in eachrow(mapped_periods)
    )

    net_loads  = compute_period_net_loads(c_pilot, cap_weights)
    cluster_nl = filter(r -> r.period in mapped_global, net_loads)

    if nrow(cluster_nl) == 0
        elapsed = (time_ns() - t0) / 1e9
        inv     = investment_cost(c_pilot)
        DBInterface.close!(c_pilot)
        result = (cost=pilot_prob.objective_value, inv=inv, lole=NaN, eens=NaN, time=elapsed)
        return return_problem ? merge(result, (problem=pilot_prob,)) : result
    end

    p_star = cluster_nl[argmax(cluster_nl.net_load_score), :period]

    DBInterface.close!(c_pilot)

    # Phase 2: final solve
    c_final = load_connection()

    if k == 1
        create_rwc_tables_k1!(c_final, p_star, n_periods, yr)
    else
        TC.cluster!(c_final, period_duration, k - 1; method=:k_medoids, layout = LAYOUT)
        fix_milestone_year!(c_final)
        append_real_periods_to_tables!(c_final, [p_star], k - 1, yr)
    end

    TEM.populate_with_defaults!(c_final)

    prob = try
        with_logger(NullLogger()) do
            TEM.run_scenario(c_final; optimizer=OPTIMIZER,
                             optimizer_parameters=OPT_PARAMS, show_log=false)
        end
    catch
        elapsed = (time_ns() - t0) / 1e9
        DBInterface.close!(c_final)
        nan = (cost=NaN, inv=NaN, lole=NaN, eens=NaN, time=elapsed)
        return return_problem ? merge(nan, (problem=nothing,)) : nan
    end

    elapsed = (time_ns() - t0) / 1e9
    inv     = investment_cost(c_final)
    DBInterface.close!(c_final)

    result = (cost=prob.objective_value, inv=inv, lole=NaN, eens=NaN, time=elapsed)
    return return_problem ? merge(result, (problem=prob,)) : result
end