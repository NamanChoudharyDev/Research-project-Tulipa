# Elgersma artificial worst-case period benchmarks.
#   :k_medoids_wc_unit — appended with weight = 1.0 (one extra day-equivalent)
#   :k_medoids_wc_frac — appended with weight = WC_FRACTION * n_periods;
#                        existing weights scaled by (1−WC_FRACTION); total stays n_periods

# 1. Elgersma worst-case construction

"""
Constructs one artificial 24-hour worst-case period from the full profile
dataset using the Elgersma construction:

  Per local hour h ∈ 1..period_duration:
    D_wc(h)   = max  demand(t)              over all t with local_hour(t) = h
    γ_g(h)    = min  A_g(t) / demand(t)    over all t with local_hour(t) = h
                                            and demand(t) > 0
    A_wc_g(h) = D_wc(h) × γ_g(h)
"""
function compute_elgersma_wc(c)
    profiles = DuckDB.query(c, """
        SELECT timestep, profile_name, value, scenario
        FROM profiles
    """) |> DataFrame
    profiles.local_hour = mod.(profiles.timestep .- 1, period_duration) .+ 1

    demand = filter(r -> r.profile_name == "demand", profiles)
    avail  = filter(r -> r.profile_name != "demand", profiles)

    # peak demand across all days and all scenarios
    D_wc = combine(groupby(demand, :local_hour), :value => maximum => :demand_wc)

    # worst ratio across all days and all scenarios
    joined = leftjoin(avail,
        select(demand, :timestep, :scenario, :value => :demand);
        on = [:timestep, :scenario])
    filter!(r -> !ismissing(r.demand) && r.demand > 0, joined)
    joined.ratio = joined.value ./ joined.demand
    min_ratios = combine(groupby(joined, [:profile_name, :local_hour]),
                         :ratio => minimum => :min_ratio)

    wc_avail = leftjoin(min_ratios, D_wc; on = :local_hour)

    demand_rows = DataFrame(
        timestep     = D_wc.local_hour,
        profile_name = fill("demand", nrow(D_wc)),
        value        = D_wc.demand_wc,
    )
    avail_rows = DataFrame(
        timestep     = wc_avail.local_hour,
        profile_name = wc_avail.profile_name,
        value        = wc_avail.demand_wc .* wc_avail.min_ratio,
    )

    wc_df = vcat(demand_rows, avail_rows)
    sort!(wc_df, [:profile_name, :timestep])
    return wc_df
end


# 2. Table helpers

function _write_wc_table!(c, df, table_name)
    tmp = "t_wc_$(table_name)"
    DuckDB.register_data_frame(c, df, tmp)
    DuckDB.execute(c, "CREATE OR REPLACE TABLE $table_name AS FROM $tmp")
    DuckDB.execute(c, "DROP VIEW $tmp")
end

"""
Handles table construction for the k = 1 case of the artificial worst-case method.
Every one of the n_periods original periods maps to rep_period = 1 with weight = 1.0.
"""
function create_wc_tables_k1!(c, wc_df, n_periods, yr)
    rpp = copy(wc_df)
    rpp.rep_period     .= 1
    rpp.milestone_year .= yr
    select!(rpp, :rep_period, :timestep, :milestone_year, :profile_name, :value)
    _write_wc_table!(c, rpp, "profiles_rep_periods")

    _write_wc_table!(c,
        DataFrame(milestone_year = yr, rep_period = 1,
                  num_timesteps  = period_duration, resolution = 1.0),
        "rep_periods_data")

    mapping_df = DuckDB.query(c, """
        SELECT DISTINCT
            $yr                                                         AS milestone_year,
            scenario,
            CAST(CEIL(timestep / $(period_duration).0) AS INTEGER)     AS period,
            1                                                           AS rep_period,
            1.0                                                         AS weight
        FROM profiles
        ORDER BY scenario, period
    """) |> DataFrame
    _write_wc_table!(c, mapping_df, "rep_periods_mapping")

    timeframe_df = DuckDB.query(c, """
        SELECT DISTINCT
            $yr                                                         AS milestone_year,
            scenario,
            CAST(CEIL(timestep / $(period_duration).0) AS INTEGER)     AS period,
            $period_duration                                            AS num_timesteps
        FROM profiles
        ORDER BY scenario, period
    """) |> DataFrame
    _write_wc_table!(c, timeframe_df, "timeframe_data")
end

"""
Appends the Elgersma worst-case period as the k-th representative.
:wc_unit — weight=1.0. Total weight becomes n_periods+1.
:wc_frac — weight=WC_FRACTION×n_periods. Existing weights ×(1−WC_FRACTION); total stays n_periods.
"""
function append_wc_to_tables!(c, wc_df, k, n_periods, yr; weight_type = :wc_unit)
    rpp = copy(wc_df)
    rpp.rep_period     .= k
    rpp.milestone_year .= yr
    select!(rpp, :rep_period, :timestep, :milestone_year, :profile_name, :value)
    DuckDB.register_data_frame(c, rpp, "t_wc_append")
    DuckDB.execute(c, "INSERT INTO profiles_rep_periods SELECT * FROM t_wc_append")
    DuckDB.execute(c, "DROP VIEW t_wc_append")

    DuckDB.execute(c, """
        INSERT INTO rep_periods_data
        VALUES ($yr, $k, $period_duration, 1.0)
    """)

    if weight_type == :wc_frac
        DuckDB.execute(c,
            "UPDATE rep_periods_mapping SET weight = weight * $(1.0 - WC_FRACTION)")
    end
end


# 3. run_once_wc

"""
Runs k_medoids on k − 1 clusters then appends the Elgersma period as the k-th representative.
"""
function run_once_wc(k, wc_df; weight_type = :wc_unit, return_problem = false)
    c  = load_connection()
    t0 = time_ns()
    yr = isempty(target_years) ? 0 : target_years[1]
    n_sc      = (DuckDB.query(c,
        "SELECT COUNT(DISTINCT scenario) AS n FROM profiles") |> DataFrame).n[1]
    n_ts      = (DuckDB.query(c,
        "SELECT MAX(timestep) AS m FROM profiles") |> DataFrame).m[1]
    n_periods = ceil(Int, n_sc * n_ts / period_duration)

    if k == 1
        create_wc_tables_k1!(c, wc_df, n_periods, yr)
    else
        TC.cluster!(c, period_duration, k - 1; method = :k_medoids, layout = LAYOUT)
        fix_milestone_year!(c)
        append_wc_to_tables!(c, wc_df, k, n_periods, yr; weight_type)
    end

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