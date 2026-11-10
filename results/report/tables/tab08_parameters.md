| section | parameter | value |
| --- | --- | --- |
| run | seed | 42 |
| run | annualization | 252 |
| run | rf | 0.0 |
| data | start_date | 2010-01-04 |
| data | end_date | 2024-12-20 |
| data | ffill_limit | 3 |
| data | max_missing_frac | 0.05 |
| data | suspect_abs_logret | 0.4 |
| data | winsor_lower | 0.001 |
| data | winsor_upper | 0.999 |
| eda | stress_date | 2020-03-31 |
| eda | corr_window_days | 21 |
| eda | index_vol_window_days | 21 |
| eda | highlight_years | [2020, 2022] |
| eda | frontier_horizon_days | 252 |
| eda | frontier_points | 50 |
| stats | block_length | 10 |
| stats | n_boot | 5000 |
| stats | sensitivity_block_lengths | [5, 10, 20] |
| stats | references | [GMV-SAMPLE, EW, GMV-LW] |
| stats | alpha | 0.05 |
| stats | mde_power | 0.8 |
| regimes | covid_start | 2020-02-19 |
| regimes | covid_end | 2020-12-31 |
| regimes | stress_year | 2022 |
| regimes | vol_window_days | 63 |
| regimes | min_history_days | 252 |
| universe | n_assets | 100 |
| universe | lookback_days | 504 |
| universe | dollar_volume_window | 252 |
| backtest | rebalance_days | 21 |
| backtest | cost_bps | 10.0 |
| backtest | max_weight | none |
| backtest | mu_convention | arithmetic |
| pca | k_rule | mp |
| pca | k_fixed | none |
| pca | baing_kmax | 10 |
| hrp | linkage | single |
| sensitivity | n_assets | [50, 100, 200] |
| sensitivity | lookback_days | [252, 504, 756] |
| sensitivity | cost_bps | [0.0, 5.0, 10.0, 20.0] |
| sensitivity | max_weight | [0.1] |
| sensitivity | pca_k_fixed | [1, 3, 5, 10] |
| sensitivity | hrp_linkage | [single, ward, average] |
| simulation | n_assets | 100 |
| simulation | n_groups | 2 |
| simulation | sample_sizes | [126, 252, 504, 1000] |
| simulation | n_reps | 200 |
| simulation | distributions | [gaussian, student_t] |
| simulation | t_df | 5.0 |
| simulation | market_vol | 0.15 |
| simulation | group_vols | [0.08, 0.06] |
| simulation | loading_low | 0.5 |
| simulation | loading_high | 1.5 |
| simulation | idio_vol_low | 0.15 |
| simulation | idio_vol_high | 0.45 |
| simulation | alpha | 0.05 |
| simulation | time_varying | False |
| simulation | horizon_days | 21 |
| simulation | vol_log_sd | 0.4 |
| simulation | vol_persistence | 0.98 |
| simulation | loading_drift_sd | 0.2 |
| simulation | loading_persistence | 0.996 |
| pipeline_check | n_symbols | 150 |
| pipeline_check | n_days | 2500 |
| pipeline_check | n_panels | 4 |
| pipeline_check | dv_persistence | 0.995 |
| pipeline_check | dv_sd | 1.0 |
| pipeline_check | z2_tolerance | 3.0 |
| pipeline_check | identity_tolerance | 1e-06 |
| nonstationarity | halflives | [126, 63] |
| nonstationarity | calibration_block_days | 21 |
| nonstationarity | sim_sample_sizes | [252, 504] |
| nonstationarity | sim_n_reps | 100 |
| residual | n_perm | 200 |
| residual | sim_n_groups | 10 |
| residual | sim_group_vol | 0.08 |
| residual | sim_label_error | 0.2 |
| residual | sim_sample_sizes | [252, 504] |
| residual | sim_n_reps | 100 |
| residual | n_placebo | 20 |
| forecast | n_portfolios | 500 |
| forecast | dirichlet_alphas | [1.0, 0.1] |
| forecast | hac_lag | 3 |
| regularized | ridge_lambdas | [0.1, 1.0, 10.0] |
| regularized | shrink_gammas | [0.25, 0.5, 0.75] |
| regularized | l1_bounds | [1.25, 1.5, 2.0] |
| regularized | enet_bound | 1.5 |
| regularized | enet_lambda | 1.0 |
| ml | momentum_days | 252 |
| ml | skip_days | 21 |
| ml | reversal_days | 21 |
| ml | vol_days | 63 |
| ml | horizon_days | 21 |
| ml | purge_days | 21 |
| ml | min_train_dates | 24 |
| ml | retrain_every | 6 |
| ml | ridge_alpha | 0.1 |
| ml | rf_trees | 200 |
| ml | rf_max_depth | 4 |
| ml | rf_min_leaf | 50 |
| ml | rf_max_features | 0.5 |
| ml | ic_gate_t | 2.0 |
| ml | force_portfolio | True |
