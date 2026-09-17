# Modelo de dados

## fighters

- fighter_id
- first_name
- last_name
- nickname
- height_cm
- weight_lbs
- reach_cm
- stance
- dob

## events

- event_id
- name
- date
- location

## fights

- fight_id
- event_id
- fighter_a_id
- fighter_b_id
- result_a
- result_b
- weight_class
- is_title_bout
- is_interim_title
- is_tournament
- method
- ending_round
- ending_time
- time_format

## round_stats

- fight_id
- round
- fighter_id
- kd
- sig_str_landed
- sig_str_attempted
- sig_str_pct
- total_str_landed
- total_str_attempted
- td_landed
- td_attempted
- td_pct
- sub_att
- reversals
- control_seconds
- head_landed
- head_attempted
- body_landed
- body_attempted
- leg_landed
- leg_attempted
- distance_landed
- distance_attempted
- clinch_landed
- clinch_attempted
- ground_landed
- ground_attempted
