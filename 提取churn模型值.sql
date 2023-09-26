```
{biz_date_pre3days}： 截点三天前
{biz_date_pre1years}:  截点一年前
{biz_date_pre1halfyears}:  截点1.5年前
```

---初始化排序表格，用于找出最后两单的时间
drop table if exists tmp.flat_tld_detail_rank_tmp;
create table if not exists tmp.flat_tld_detail_rank_tmp as 
with ranktable as
(select *, dense_rank() over(partition by user_code order by transaction_start_time desc) as rank_val
from dw_kfc.flat_tld_detail
where user_code is not null and p_biz_date between '${biz_date_pre1years}' and '${biz_date_pre3days}')

select distinct transaction_guid
    , transaction_amount
    , yumid, user_code
    , p_biz_date, daypart_name,rank_val from ranktable where rank_val <= 2;



---定义最后一天的特征，考虑内存原因只抽取1000000的用户
drop table if exists tmp.churn_user_feats_new;
create table if not exists tmp.churn_user_feats_new
stored as parquet as
select distinct * from
(select *,
ROW_NUMBER() OVER (PARTITION BY t.yumid ORDER BY t.p_biz_date DESC) AS userRank
from aie_feature_center.kfc_recsys_user_stats_daily_v2 t
where p_biz_date between '${biz_date_pre1years}' and '${biz_date_pre3days}'
and yumid is not null
) a
where a.userRank = 1
order by rand()
limit 1000000;


---找出cltv2的最后一单天数差值
create table tmp.cltv2_date_diff_new as
with last_order as 
(select yumid, user_code, p_biz_date as max_biz_day 
from tmp.flat_tld_detail_rank_tmp
where rank_val = 1),
prev_order as
(select yumid, user_code, p_biz_date as min_biz_day
from tmp.flat_tld_detail_rank_tmp
where rank_val = 2) 

select b.user_code, abs(datediff(to_date(concat(substr(max_biz_day,1,4),'-',substr(max_biz_day,5,2),'-',substr(max_biz_day,7))),to_date(concat(substr(min_biz_day,1,4),'-',substr(min_biz_day,5,2),'-',substr(min_biz_day,7))))) as diff_date, b.max_biz_day
from (select yumid
from aie_kfc_cltv.old_member_pool where cltv_flag = 'cltv2') a
inner join last_order as b
on a.yumid = b.yumid
inner join prev_order as c
on a.yumid = c.yumid
join tmp.churn_user_feats_new d
on a.yumid = d.yumid;


---在用户画像的主表上加上churn作为y值，这里是考虑了60天100天180天以及100天可召回的用户作为y值可以去比较 (列出一些column的名字防止内存报错 select * 会内存报错)
drop table if exists tmp.churn_with_feats_new;
create table if not exists tmp.churn_with_feats_new
stored as parquet as
select t2.biz_date,t2.usercode_number,t2.yumid,t2.u_user_code, tc,ta,t2.breakfast_tc, nonbreakfast_tc, morning_tc,lunch_tc,afternoon_tc,dinner_tc,latenight_tc,
    mon_tc, tue_tc, wen_tc, thu_tc, fri_tc, sat_tc, sun_tc, tier1_tc, tier2_tc, tier3_tc, tier4_tc, tier5_tc, tier6_tc, breakfast_maxta, nonbreakfast_maxta, 
    morning_maxta,lunch_maxta, afternoon_maxta, dinner_maxta, latenight_maxta, mon_maxta, tue_maxta, wen_maxta, thu_maxta, fri_maxta, sat_maxta, sun_maxta,breakfast_minta, nonbreakfast_minta, morning_minta,lunch_minta, afternoon_minta, dinner_minta, latenight_minta, mon_minta, tue_minta, wen_minta, thu_minta, fri_minta, sat_minta, sun_minta ,
    breakfast_avgta, nonbreakfast_avgta, morning_avgta,lunch_avgta, afternoon_avgta, dinner_avgta, latenight_avgta, mon_avgta, tue_avgta, wen_avgta, thu_avgta, fri_avgta, sat_avgta, sun_avgta,
    breakfast_sumta, nonbreakfast_sumta, morning_sumta,lunch_sumta, afternoon_sumta, dinner_sumta, latenight_sumta, mon_sumta, tue_sumta, wen_sumta, thu_sumta, fri_sumta, sat_sumta, sun_sumta,
    breakfast_std_ta, nonbreakfast_std_ta, morning_std_ta,lunch_std_ta, afternoon_std_ta, dinner_std_ta, latenight_std_ta, mon_std_ta, tue_std_ta, wen_std_ta, thu_std_ta, fri_std_ta, sat_std_ta, sun_std_ta, avg_discount, max_discount, min_discount, sum_discount, std_discount, 
    avg_ta_by_ps, std_ta_by_ps, avg_city_tier, max_city_tier, min_city_tier, std_city_tier, avg_party_size, max_party_size,min_party_size, cor_ta_da, distinct_daypart, distinct_city, distinct_work_day, distinct_store, delivery_tc, delivery_ta, avg_preorder_party_size,  max_preorder_party_size, min_preorder_party_size, std_preorder_party_size,
    side_sold, coffee_sold, congee_sold, nutrition_sold, panini_sold, riceroll_sold, dabing_sold, burger_sold, chickensnack_sold, cob_sold,csd_sold, eggtart_sold,icecream_sold, t2.sidefrenchfries_sold, sideothers_sold, tea_sold, twister_sold, wing_sold, waffle_sold, croissant_sold, nonfood_sold, pie_sold, juice_sold, rice_sold, lto_sold, city_tier_set_repeat25tc,city_tier_set_repeat50tc, city_tier_set_repeat75tc,
    party_size_set_repeat25tc, party_size_set_repeat50tc,party_size_set_repeat75tc,t2.day_of_week_set_repeat25tc,t2.day_of_week_set_repeat50tc, t2.day_of_week_set_repeat75tc, daypart_name_set_repeat25tc,  daypart_name_set_repeat50tc,daypart_name_set_repeat75tc, t2.occasion_name_set_repeat25tc,t2.occasion_name_set_repeat50tc, t2.occasion_name_set_repeat75tc, t2.p_biz_date,
    case when abs(datediff(biz_date, current_date())) > 100 then 1 else 0 end as churn_100, case when abs(datediff(biz_date, current_date())) > 180 then 1 else 0 end as churn_180, case when t3.diff_date>100 and t3.user_code is not null then 1 else 0 end as returnable_churn_100, case when abs(datediff(biz_date, current_date())) > 60 then 1 else 0 end as churn_60
from (select yumid
from aie_kfc_cltv.old_member_pool where cltv_flag = 'cltv2') t1
join tmp.churn_user_feats_new t2
on t1.yumid = t2.yumid
left join tmp.cltv2_date_diff_new t3
on t2.usercode_number = t3.user_code;

select sum(churn_100)/count(churn_100)
from tmp.churn_with_feats_new;


---加上性别和年龄特征
create table tmp.churn_feats_personal_info_new as 
select t2.biz_date,t2.usercode_number,t2.yumid,sex_flag_text,derived_age,  tc,ta,t2.breakfast_tc, nonbreakfast_tc, morning_tc,lunch_tc,afternoon_tc,dinner_tc,latenight_tc,
    mon_tc, tue_tc, wen_tc, thu_tc, fri_tc, sat_tc, sun_tc, tier1_tc, tier2_tc, tier3_tc, tier4_tc, tier5_tc, tier6_tc, breakfast_maxta, nonbreakfast_maxta, 
    morning_maxta,lunch_maxta, afternoon_maxta, dinner_maxta, latenight_maxta, mon_maxta, tue_maxta, wen_maxta, thu_maxta, fri_maxta, sat_maxta, sun_maxta,breakfast_minta, nonbreakfast_minta, morning_minta,lunch_minta, afternoon_minta, dinner_minta, latenight_minta, mon_minta, tue_minta, wen_minta, thu_minta, fri_minta, sat_minta, sun_minta ,
    breakfast_avgta, nonbreakfast_avgta, morning_avgta,lunch_avgta, afternoon_avgta, dinner_avgta, latenight_avgta, mon_avgta, tue_avgta, wen_avgta, thu_avgta, fri_avgta, sat_avgta, sun_avgta,
    breakfast_sumta, nonbreakfast_sumta, morning_sumta,lunch_sumta, afternoon_sumta, dinner_sumta, latenight_sumta, mon_sumta, tue_sumta, wen_sumta, thu_sumta, fri_sumta, sat_sumta, sun_sumta,
    breakfast_std_ta, nonbreakfast_std_ta, morning_std_ta,lunch_std_ta, afternoon_std_ta, dinner_std_ta, latenight_std_ta, mon_std_ta, tue_std_ta, wen_std_ta, thu_std_ta, fri_std_ta, sat_std_ta, sun_std_ta, avg_discount, max_discount, min_discount, sum_discount, std_discount, 
    avg_ta_by_ps, std_ta_by_ps, avg_city_tier, max_city_tier, min_city_tier, std_city_tier, avg_party_size, max_party_size,min_party_size, cor_ta_da, distinct_daypart, distinct_city, distinct_work_day, distinct_store, delivery_tc, delivery_ta, avg_preorder_party_size,  max_preorder_party_size, min_preorder_party_size, std_preorder_party_size,
    side_sold, coffee_sold, congee_sold, nutrition_sold, panini_sold, riceroll_sold, dabing_sold, burger_sold, chickensnack_sold, cob_sold,csd_sold, eggtart_sold,icecream_sold, t2.sidefrenchfries_sold, sideothers_sold, tea_sold, twister_sold, wing_sold, waffle_sold, croissant_sold, nonfood_sold, pie_sold, juice_sold, rice_sold, lto_sold, city_tier_set_repeat25tc,city_tier_set_repeat50tc, city_tier_set_repeat75tc,
    party_size_set_repeat25tc, party_size_set_repeat50tc,party_size_set_repeat75tc,t2.day_of_week_set_repeat25tc,t2.day_of_week_set_repeat50tc, t2.day_of_week_set_repeat75tc, daypart_name_set_repeat25tc,  daypart_name_set_repeat50tc,daypart_name_set_repeat75tc, t2.occasion_name_set_repeat25tc,t2.occasion_name_set_repeat50tc, t2.occasion_name_set_repeat75tc, t2.p_biz_date,churn_100, churn_180,returnable_churn_100,churn_60
from tmp.churn_with_feats_new t2
left join dw_kfc.flat_user_info t1
on t2.usercode_number = t1.user_code;


---增加12类人群字段
drop table if exists tmp.churn_feats_with_mkt_new;
create table if not exists tmp.churn_feats_with_mkt_new as 
select t1.*, his_family_flag, his_travel_flag, student_flag, his_value_seeker_flag, his_delivery_flag, his_lto_flag, his_working_day_lunch_flag, his_breakfast_flag, his_afternoon_flag, his_dinner_flag, his_coffee_flag, elderly_flag 
from tmp.churn_feats_personal_info_new t1
left join (select yumid, his_family_flag, his_travel_flag, student_flag, his_value_seeker_flag, his_delivery_flag, his_lto_flag, his_working_day_lunch_flag, his_breakfast_flag, his_afternoon_flag, his_dinner_flag, his_coffee_flag, elderly_flag 
from dw_kfc.flat_user_trans_profile_his) t2
on t1.yumid = t2.yumid;






---增加ai券有关信息 (半年内获得几张券，用了几张券，用券的平均party)  注：所有券是最后加上的在代码最后
---- 1. 抽取券的信息放在tmp表里
create table tmp.sx_redeem_tmp as
select * from etl_kfc.t_kfc_coupon_redeem_dwbi_day
where day between '${biz_date_pre1years}' and '${biz_date_pre3days}';


---- 2.加入券，此处只考虑推荐组的ai字段券
drop table if exists tmp.redeem_party_size_new;
create table if not exists tmp.redeem_party_size_new as
with half_yr_coupon as 
(select t1.*, t2.*, t3.*, abs(datediff(biz_day,coupon_biz_day)) as date_diff
from (select *, to_date(concat(substr(p_biz_date,1,4),'-',substr(p_biz_date,5,2),'-',substr(p_biz_date,7))) as biz_day from tmp.churn_feats_with_mkt_new) t1
left join (select distinct p_day, usercode, activityid,to_date(concat(substr(p_day,1,4),'-',substr(p_day,5,2),'-',substr(p_day,7))) as coupon_biz_day  from aie_kfc_cltv.ai_place_coupon
where p_day between '${biz_date_pre1halfyears}' and '${biz_date_pre3days}') t2
on t1.usercode_number = t2.usercode 
left join (select distinct coupon_usercode,coupon_activityid from tmp.sx_redeem_tmp) t3
on t2.usercode = t3.coupon_usercode and t2.activityid = t3.coupon_activityid)

select usercode_number, avg(case when coupon_usercode is not null then avg_party_size else null end) as ai_redeemed_avg_party_size, count(usercode) as ai_num_coupon, count(coupon_usercode) as ai_num_redeem
from (select distinct usercode_number,avg_party_size, coupon_usercode, coupon_activityid, usercode, activityid from half_yr_coupon where date_diff < 180) t
group by usercode_number;


----3. 如果用户没有收到券，则在ai_num_redeem赋值-1，和收到券没有用的人分开
drop table if exists tmp.churn_feats_with_coupon_new;
create table if not exists tmp.churn_feats_with_coupon_new as 
select t1.*, case when t2.ai_redeemed_avg_party_size is null then -1 else t2.ai_redeemed_avg_party_size end as ai_redeemed_avg_party_size, case when t2.ai_num_coupon is null then 0 else t2.ai_num_coupon end as ai_num_coupon, case when t2.ai_num_coupon is null then -1 else t2.ai_num_redeem end as ai_num_redeem
from tmp.churn_feats_with_mkt_new t1
left join tmp.redeem_party_size_new t2
on t1.usercode_number = t2.usercode_number;





---随机取250000数据量，包括是否是100天流失，100天后被召回
drop table if exists tmp.sx_churn_feature_tmp_new;
create table if not exists tmp.sx_churn_feature_tmp_new as
select distinct usercode_number,  sex_flag_text, derived_age,his_family_flag, his_travel_flag, student_flag, his_value_seeker_flag, his_delivery_flag, his_lto_flag, his_working_day_lunch_flag, his_breakfast_flag, his_afternoon_flag, his_dinner_flag, his_coffee_flag, elderly_flag ,ai_num_coupon, ai_num_redeem, ai_redeemed_avg_party_size, tc,ta,breakfast_tc, nonbreakfast_tc, morning_tc,lunch_tc,afternoon_tc,dinner_tc,latenight_tc,
    mon_tc, tue_tc, wen_tc, thu_tc, fri_tc, sat_tc, sun_tc, tier1_tc, tier2_tc, tier3_tc, tier4_tc, tier5_tc, tier6_tc, breakfast_maxta, nonbreakfast_maxta, 
    morning_maxta,lunch_maxta, afternoon_maxta, dinner_maxta, latenight_maxta, mon_maxta, tue_maxta, wen_maxta, thu_maxta, fri_maxta, sat_maxta, sun_maxta,breakfast_minta, nonbreakfast_minta, morning_minta,lunch_minta, afternoon_minta, dinner_minta, latenight_minta, mon_minta, tue_minta, wen_minta, thu_minta, fri_minta, sat_minta, sun_minta ,
    breakfast_avgta, nonbreakfast_avgta, morning_avgta,lunch_avgta, afternoon_avgta, dinner_avgta, latenight_avgta, mon_avgta, tue_avgta, wen_avgta, thu_avgta, fri_avgta, sat_avgta, sun_avgta,
    breakfast_sumta, nonbreakfast_sumta, morning_sumta,lunch_sumta, afternoon_sumta, dinner_sumta, latenight_sumta, mon_sumta, tue_sumta, wen_sumta, thu_sumta, fri_sumta, sat_sumta, sun_sumta,
    breakfast_std_ta, nonbreakfast_std_ta, morning_std_ta,lunch_std_ta, afternoon_std_ta, dinner_std_ta, latenight_std_ta, mon_std_ta, tue_std_ta, wen_std_ta, thu_std_ta, fri_std_ta, sat_std_ta, sun_std_ta, avg_discount, max_discount, min_discount, sum_discount, std_discount, 
    avg_ta_by_ps, std_ta_by_ps, avg_city_tier, max_city_tier, min_city_tier, std_city_tier, avg_party_size, max_party_size,min_party_size, cor_ta_da, distinct_daypart, distinct_city, distinct_work_day, distinct_store, delivery_tc, delivery_ta, avg_preorder_party_size,  max_preorder_party_size, min_preorder_party_size, std_preorder_party_size,
    side_sold, coffee_sold, congee_sold, nutrition_sold, panini_sold, riceroll_sold, dabing_sold, burger_sold, chickensnack_sold, cob_sold,csd_sold, eggtart_sold,icecream_sold,   sidefrenchfries_sold, sideothers_sold, tea_sold, twister_sold, wing_sold, waffle_sold, croissant_sold, nonfood_sold, pie_sold, juice_sold, rice_sold, lto_sold, city_tier_set_repeat50tc,
    party_size_set_repeat50tc,   day_of_week_set_repeat50tc,  daypart_name_set_repeat50tc,   occasion_name_set_repeat50tc,   churn_100, churn_180,returnable_churn_100,churn_60
from tmp.churn_feats_with_coupon_new
order by rand()
limit 250000;




---前面数据跑出来usercode重复率可能0.003，保证一个用户数据为一行：
drop table if exists tmp.sx_distinct_churn_feature_new;
create table if not exists tmp.sx_distinct_churn_feature_new as
select t1.*
from tmp.sx_churn_feature_tmp_new t1
join (select usercode_number 
    from tmp.sx_churn_feature_tmp_new
    group by usercode_number
    having count(usercode_number)=1) t2
on t1.usercode_number = t2.usercode_number
limit 200000;





---加上ta和tc_diff的值：
----1. 定义选取的数据最后一天（sx_distinct_churn_feature_new）
create table tmp.tld_header_last_day_new as 
select t2.*, ROW_NUMBER() OVER (PARTITION BY t2.user_code ORDER BY t2.p_biz_date DESC) AS userRank
from tmp.sx_distinct_churn_feature_new t1
join (select * from dw_kfc.flat_tld_header where p_biz_date between '${biz_date_pre1year}' and '${biz_date_pre3days}') t2
on t1.usercode_number = t2.user_code;


----2. 选取前三个月后三个月的ta、tc
create table tmp.sx_churn_feat_addition_ta_tc_new as
with last_day as 
(select user_code, biz_date
from tmp.tld_header_last_day_new
where userRank = 1)

select t2.user_code, sum(distinct case when t2.biz_date between date_sub(t1.biz_date, 60) and t1.biz_date then transaction_amount else 0 end) as ta_after,
    sum(distinct case when t2.biz_date between date_sub(t1.biz_date, 120) and date_sub(t1.biz_date,60) then transaction_amount else 0 end) as ta_prev,
    count(distinct case when t2.biz_date between date_sub(t1.biz_date, 60) and t1.biz_date then transaction_guid else null end) as tc_after,
    count(distinct case when t2.biz_date between date_sub(t1.biz_date, 120) and date_sub(t1.biz_date,60) then transaction_guid else null end) as tc_prev
from last_day t1
join (select distinct transaction_guid, user_code, transaction_amount,biz_date from dw_kfc.flat_tld_header where p_biz_date between '${biz_date_pre1halfyears}' and '${biz_date_pre3days}') t2
on t1.user_code = t2.user_code
group by t2.user_code;


----3. 加上ta_diff和tc_diff到主表
create table if not exists tmp.sx_distinct_churn_feature_diff_new as 
select usercode_number,  sex_flag_text, derived_age,his_family_flag, his_travel_flag, student_flag, his_value_seeker_flag, his_delivery_flag, his_lto_flag, his_working_day_lunch_flag, his_breakfast_flag, his_afternoon_flag, his_dinner_flag, his_coffee_flag, elderly_flag ,ai_num_coupon, ai_num_redeem, ai_redeemed_avg_party_size, tc,ta,
    (tc_after-tc_prev) as tc_diff, (ta_after-ta_prev) as ta_diff, breakfast_tc, nonbreakfast_tc, morning_tc,lunch_tc,afternoon_tc,dinner_tc,latenight_tc,
    mon_tc, tue_tc, wen_tc, thu_tc, fri_tc, sat_tc, sun_tc, tier1_tc, tier2_tc, tier3_tc, tier4_tc, tier5_tc, tier6_tc, breakfast_maxta, nonbreakfast_maxta, 
    morning_maxta,lunch_maxta, afternoon_maxta, dinner_maxta, latenight_maxta, mon_maxta, tue_maxta, wen_maxta, thu_maxta, fri_maxta, sat_maxta, sun_maxta,breakfast_minta, nonbreakfast_minta, morning_minta,lunch_minta, afternoon_minta, dinner_minta, latenight_minta, mon_minta, tue_minta, wen_minta, thu_minta, fri_minta, sat_minta, sun_minta ,
    breakfast_avgta, nonbreakfast_avgta, morning_avgta,lunch_avgta, afternoon_avgta, dinner_avgta, latenight_avgta, mon_avgta, tue_avgta, wen_avgta, thu_avgta, fri_avgta, sat_avgta, sun_avgta,
    breakfast_sumta, nonbreakfast_sumta, morning_sumta,lunch_sumta, afternoon_sumta, dinner_sumta, latenight_sumta, mon_sumta, tue_sumta, wen_sumta, thu_sumta, fri_sumta, sat_sumta, sun_sumta,
    breakfast_std_ta, nonbreakfast_std_ta, morning_std_ta,lunch_std_ta, afternoon_std_ta, dinner_std_ta, latenight_std_ta, mon_std_ta, tue_std_ta, wen_std_ta, thu_std_ta, fri_std_ta, sat_std_ta, sun_std_ta, avg_discount, max_discount, min_discount, sum_discount, std_discount, 
    avg_ta_by_ps, std_ta_by_ps, avg_city_tier, max_city_tier, min_city_tier, std_city_tier, avg_party_size, max_party_size,min_party_size, cor_ta_da, distinct_daypart, distinct_city, distinct_work_day, distinct_store, delivery_tc, delivery_ta, avg_preorder_party_size,  max_preorder_party_size, min_preorder_party_size, std_preorder_party_size,
    side_sold, coffee_sold, congee_sold, nutrition_sold, panini_sold, riceroll_sold, dabing_sold, burger_sold, chickensnack_sold, cob_sold,csd_sold, eggtart_sold,icecream_sold,   sidefrenchfries_sold, sideothers_sold, tea_sold, twister_sold, wing_sold, waffle_sold, croissant_sold, nonfood_sold, pie_sold, juice_sold, rice_sold, lto_sold, city_tier_set_repeat50tc,
    party_size_set_repeat50tc,   day_of_week_set_repeat50tc,  daypart_name_set_repeat50tc,   occasion_name_set_repeat50tc,   churn_100, churn_180,returnable_churn_100,churn_60
from tmp.sx_distinct_churn_feature_new t1
join tmp.sx_churn_feat_addition_ta_tc_new t2
on t1.usercode_number = t2.user_code;




---计算所有的券 (之前只计算了ai的券)
drop table if exists tmp.redeem_party_size_all_coupon_new;
create table if not exists tmp.redeem_party_size_all_coupon_new as
with half_yr_coupon as 
(select t1.*, t2.*, t3.*, abs(datediff(biz_day,coupon_biz_day)) as date_diff
from (select *, to_date(concat(substr(p_biz_date,1,4),'-',substr(p_biz_date,5,2),'-',substr(p_biz_date,7))) as biz_day from tmp.churn_feats_with_mkt_new) t1
left join (select distinct p_day, user_code, activity_id,to_date(concat(substr(p_day,1,4),'-',substr(p_day,5,2),'-',substr(p_day,7))) as coupon_biz_day  from dw_kfc.flat_coupon_activated_account_info
where p_day between '${biz_date_pre1halfyears}' and '${biz_date_pre3days}') t2
on t1.usercode_number = t2.user_code 
left join (select distinct coupon_usercode,coupon_activityid from tmp.sx_redeem_tmp) t3
on t2.user_code = t3.coupon_usercode and t2.activity_id = t3.coupon_activityid)

select usercode_number, avg(case when coupon_usercode is not null then avg_party_size else null end) as all_redeemed_avg_party_size, count(user_code) as all_num_coupon, count(coupon_usercode) as all_num_redeem
from (select distinct usercode_number,avg_party_size, coupon_usercode, coupon_activityid, user_code, activity_id from half_yr_coupon where date_diff < 180) t
group by usercode_number;


---结合到原表
create table tmp.sx_distinct_churn_feature_diff_final as
select      t1.usercode_number,  sex_flag_text, derived_age,his_family_flag, his_travel_flag, student_flag, his_value_seeker_flag, his_delivery_flag, his_lto_flag, his_working_day_lunch_flag, his_breakfast_flag, his_afternoon_flag, his_dinner_flag, his_coffee_flag, elderly_flag 
    ,case when t2.all_redeemed_avg_party_size is null then -1 else t2.all_redeemed_avg_party_size end as all_redeemed_avg_party_size, case when t2.all_num_coupon is null then 0 else t2.all_num_coupon end as all_num_coupon, case when t2.all_num_coupon is null then -1 else t2.all_num_redeem end as all_num_redeem
    , ai_redeemed_avg_party_size,  ai_num_coupon, ai_num_redeem
    , tc,ta,
    tc_diff, ta_diff, breakfast_tc, nonbreakfast_tc, morning_tc,lunch_tc,afternoon_tc,dinner_tc,latenight_tc,
    mon_tc, tue_tc, wen_tc, thu_tc, fri_tc, sat_tc, sun_tc, tier1_tc, tier2_tc, tier3_tc, tier4_tc, tier5_tc, tier6_tc, breakfast_maxta, nonbreakfast_maxta, 
    morning_maxta,lunch_maxta, afternoon_maxta, dinner_maxta, latenight_maxta, mon_maxta, tue_maxta, wen_maxta, thu_maxta, fri_maxta, sat_maxta, sun_maxta,breakfast_minta, nonbreakfast_minta, morning_minta,lunch_minta, afternoon_minta, dinner_minta, latenight_minta, mon_minta, tue_minta, wen_minta, thu_minta, fri_minta, sat_minta, sun_minta ,
    breakfast_avgta, nonbreakfast_avgta, morning_avgta,lunch_avgta, afternoon_avgta, dinner_avgta, latenight_avgta, mon_avgta, tue_avgta, wen_avgta, thu_avgta, fri_avgta, sat_avgta, sun_avgta,
    breakfast_sumta, nonbreakfast_sumta, morning_sumta,lunch_sumta, afternoon_sumta, dinner_sumta, latenight_sumta, mon_sumta, tue_sumta, wen_sumta, thu_sumta, fri_sumta, sat_sumta, sun_sumta,
    breakfast_std_ta, nonbreakfast_std_ta, morning_std_ta,lunch_std_ta, afternoon_std_ta, dinner_std_ta, latenight_std_ta, mon_std_ta, tue_std_ta, wen_std_ta, thu_std_ta, fri_std_ta, sat_std_ta, sun_std_ta, avg_discount, max_discount, min_discount, sum_discount, std_discount, 
    avg_ta_by_ps, std_ta_by_ps, avg_city_tier, max_city_tier, min_city_tier, std_city_tier, avg_party_size, max_party_size,min_party_size, cor_ta_da, distinct_daypart, distinct_city, distinct_work_day, distinct_store, delivery_tc, delivery_ta, avg_preorder_party_size,  max_preorder_party_size, min_preorder_party_size, std_preorder_party_size,
    side_sold, coffee_sold, congee_sold, nutrition_sold, panini_sold, riceroll_sold, dabing_sold, burger_sold, chickensnack_sold, cob_sold,csd_sold, eggtart_sold,icecream_sold,   sidefrenchfries_sold, sideothers_sold, tea_sold, twister_sold, wing_sold, waffle_sold, croissant_sold, nonfood_sold, pie_sold, juice_sold, rice_sold, lto_sold, city_tier_set_repeat50tc,
    party_size_set_repeat50tc,   day_of_week_set_repeat50tc,  daypart_name_set_repeat50tc,   occasion_name_set_repeat50tc,   churn_100, churn_180,returnable_churn_100,churn_60
from tmp.sx_distinct_churn_feature_diff_new t1    
left join tmp.redeem_party_size_all_coupon_new t2    
on t1.usercode_number = t2.usercode_number;




---以下是增加的字段：

----增加主表信息
create table tmp.sx_user_info_from_churn_final_yumid as
select t.*, transaction_guid, occasion_name, online_trans_channel_lv2, online_trans_channel_lv3, product_class_name,unit_sold,sell_name, is_combo_flag, abs(datediff(biz_day,transaction_biz_day)) as date_diff
from tmp.sx_distinct_churn_feature_diff_final t
join (select yumid,usercode_number, distinct_store,to_date(concat(substr(p_biz_date,1,4),'-',substr(p_biz_date,5,2),'-',substr(p_biz_date,7))) as biz_day from tmp.churn_with_feats_new) t1
on t1.usercode_number = t.usercode_number
join (select * , to_date(concat(substr(p_biz_date,1,4),'-',substr(p_biz_date,5,2),'-',substr(p_biz_date,7))) as transaction_biz_day from dw_kfc.flat_tld_detail where p_biz_date between 20220701 and 20230701) t2
on t1.yumid = t2.yumid;




```
第一部分： 商圈类型
```
create table if not exists tmp.sx_churn_feats_trade_zone as
with half_yr_tc_trade_zone as 
(select t1.usercode_number, t2.yumid,t2.transaction_guid, t3.*, abs(datediff(biz_day,transaction_biz_day)) as date_diff
from (select yumid,usercode_number, distinct_store,to_date(concat(substr(p_biz_date,1,4),'-',substr(p_biz_date,5,2),'-',substr(p_biz_date,7))) as biz_day from tmp.churn_with_feats_new) t1
join (select usercode_number from tmp.sx_reduce_churn_risk_exploration_data) t
on t1.usercode_number = t.usercode_number
join (select yumid,transaction_guid, store_code, to_date(concat(substr(p_biz_date,1,4),'-',substr(p_biz_date,5,2),'-',substr(p_biz_date,7))) as transaction_biz_day from dw_kfc.flat_tld_header where p_biz_date between 20220701 and 20230701) t2
on t1.yumid = t2.yumid
join dw_kfc.dim_store t3
on t2.store_code = t3.store_code)

select usercode_number, count(CASE WHEN trade_zone IN ('O+C','O+D','O','C+O','O+D','D+O') THEN transaction_guid else null end)  as working_trade_zone,
    count(CASE WHEN trade_zone IN ('X/B','X/A','A','C','C+X','B','A+X','C+医院','C+专业市场','C+市内公共交通') THEN transaction_guid else null end) as traditional_business_district,
    count(CASE WHEN trade_zone IN ('WM+U','U','u','大学') THEN transaction_guid else null end) as university,
    count(CASE WHEN trade_zone IN ('SM+D','SM/A','SM/B','SM','SM+C','购物中心+C','购物中心+D','购物中心/A','购物中心/B','购物中心') THEN transaction_guid else null end) as shopping_center,
    count(CASE WHEN trade_zone IN ('RS','BS','Apt','R','HW','机场','客运站+C','火车站','客运站','客运站+D','高速服务区') THEN transaction_guid else null end) as traffic_center,
    count(CASE WHEN trade_zone IN ('T') THEN transaction_guid else null end) as travel_trade_zone,
    count(CASE WHEN trade_zone IN ('X+D','X+C','X') THEN transaction_guid else null end) as shopping_mall,
    count(CASE WHEN trade_zone IN ('OT','s','S','Z','开发区','Others') THEN transaction_guid else null end) as other_business_district,
    count(CASE WHEN trade_zone IN ('D+x','D','D+H','D+X','D+HP','D+医院','D+专业市场','D+市内公共交通','D+客运站') THEN transaction_guid else null end) as social_business_district,
    count(CASE WHEN trade_zone IN ('Tpc+D','Tpc','TPC','TpC','市内公共交通+C','市内公共交通+D','市内公共交通/A','市内公共交通','市内公共交通/B') THEN transaction_guid else null end) as city_public_traffic,
    count(CASE WHEN trade_zone IN ('HP','HP+D','医院+D','医院','医院+C') THEN transaction_guid else null end) as hospital,
    count(CASE WHEN trade_zone IN ('M+D','M','专业市场','专业市场+C','专业市场+D') THEN transaction_guid else null end) as market_trade_zone
from (select distinct usercode_number, transaction_guid, trade_zone from half_yr_tc_trade_zone where date_diff<180) t
group by usercode_number;



```
第二部分： 策略品，牛堡and全鸡
```
drop table if exists tmp.sx_churn_feats_product_class_type;
create table if not exists tmp.sx_churn_feats_product_class_type as
select usercode_number, 
    sum(case when product_class_name = "Beef Burger" then unit_sold else 0 end) as num_beef_burger,
    sum(case when product_class_name = "全鸡" then unit_sold else 0 end) as num_whole_chicken,
    sum(case when product_class_name = '现磨咖啡推广版' then unit_sold else 0 end) as num_coffee,
    sum(case when product_class_name = '通用套餐头' AND is_combo_flag = 1
    AND (sell_name like '%APP%专享%' or sell_name like '%OK%') then unit_sold else 0 end) as num_ok_can,
    count(distinct case when product_class_name = "Beef Burger" then transaction_guid else null end) as beef_burger_tc,
    count(distinct case when product_class_name = "全鸡" then transaction_guid else null end) as whole_chicken_tc,
    count(distinct case when product_class_name = '现磨咖啡推广版' then transaction_guid else null end) as coffee_tc,
    count(distinct case when product_class_name = '通用套餐头' AND is_combo_flag = 1
    AND (sell_name like '%APP%专享%' or sell_name like '%OK%') then transaction_guid else null end) as ok_can_tc
from tmp.sx_user_info_from_churn_final_yumid t
group by usercode_number;




```
第三部分： 三方以及app小程序tc绝对值
```
create table tmp.sx_churn_feats_sanfang_type as
select usercode_number, 
    count(case when occasion_name = 'Pro delivery CSC' and (online_trans_channel_lv2 = '3rd' or online_trans_channel_lv2 is null) then 1 else null end) as num_sanfang
from (select distinct usercode_number, transaction_guid, occasion_name,online_trans_channel_lv2 
    from tmp.sx_user_info_from_churn_final_yumid where date_diff<180) t1
group by usercode_number;




---到这里
--- 中高频有宅急送 自有渠道订单
drop table tmp.sx_churn_feats_hightc_percent;
create table tmp.sx_churn_feats_hightc_percent as
with cte as 
(select usercode_number,
    count(distinct case when occasion_name = 'Pro delivery CSC' then transaction_guid else null end) as delivery_count,---宅急送消费次数
    count(distinct case when (online_trans_channel_lv2 in ("SUPERAPP","MOBILE") and (online_trans_channel_lv3 is null or online_trans_channel_lv3 like "%APP%")) or (online_trans_channel_lv2 in ("WECHATH5","WECHATMINI","MOBILE","ALIMINI","ALIPAY") and (online_trans_channel_lv3 is null or online_trans_channel_lv3 like "%WECHAT%" or online_trans_channel_lv3 like "%ALIPAY%" or online_trans_channel_lv3 like "%支付宝%")) then transaction_guid else null end) as app_wechat_high_tc
from (select distinct usercode_number, transaction_guid, occasion_name,online_trans_channel_lv2,online_trans_channel_lv3
    from tmp.sx_user_info_from_churn_final_yumid where date_diff<180 and tc>=8) t
group by usercode_number)

select t1.usercode_number, 
    case when t2.usercode_number is null then -1 when app_wechat_high_tc >tc then 1 else (app_wechat_high_tc)/tc end as high_app_wechat_tc_percent
from tmp.sx_distinct_churn_feature_diff_final t1
left join (select * from cte where delivery_count>0) t2
on t1.usercode_number = t2.usercode_number;



---app小程序订单绝对值及占比
create table tmp.sx_churn_feats_app_wechat_tc as
select usercode_number, 
    count(distinct case when online_trans_channel_lv2 in ("SUPERAPP","MOBILE") and (online_trans_channel_lv3 is null or online_trans_channel_lv3 like "%APP%") then transaction_guid else null end) as APP_tc,
    count(distinct case when online_trans_channel_lv2 in ("WECHATH5","WECHATMINI","MOBILE","ALIMINI","ALIPAY") and (online_trans_channel_lv3 is null or online_trans_channel_lv3 like "%WECHAT%" or online_trans_channel_lv3 like "%ALIPAY%" or online_trans_channel_lv3 like "%支付宝%") then transaction_guid else null end) as wechat_tc
from (select distinct usercode_number, tc, transaction_guid, online_trans_channel_lv2, online_trans_channel_lv3,occasion_name from tmp.sx_user_info_from_churn_final_yumid where date_diff<180) t
group by usercode_number;






```
第四部分：券 


create table if not exists tmp.sx_churn_feats_douyin_meituan_coupon as
with half_yr_coupon_tmp as 
(select t1.*, t2.*, t3.*, abs(datediff(biz_day,coupon_biz_day)) as date_diff
from (select *, to_date(concat(substr(p_biz_date,1,4),'-',substr(p_biz_date,5,2),'-',substr(p_biz_date,7))) as biz_day from tmp.churn_feats_with_mkt_new) t1
left join (select distinct p_day, user_code, activity_id,activity_name, activity_group, place_channel_name, to_date(concat(substr(p_day,1,4),'-',substr(p_day,5,2),'-',substr(p_day,7))) as coupon_biz_day  from dw_kfc.flat_coupon_activated_account_info
where p_day between '${biz_date_pre1halfyears}' and '${biz_date_pre3days}') t2
on t1.usercode_number = t2.user_code 
left join (select distinct coupon_usercode,coupon_activityid from etl_kfc.t_kfc_coupon_redeem_dwbi_day where day between '${biz_date_pre1halfyears}' and '${biz_date_pre3days}') t3
on t2.user_code = t3.coupon_usercode and t2.activity_id = t3.coupon_activityid)

select usercode_number, 
    count(case when activity_name like '%抖音%' or activity_group = 'KFC抖音团购券' then coupon_usercode else null end) as douyin_redeem,
    count(case when place_channel_name = '美团预付券' then coupon_usercode else null end) as meituan_redeem
from (select distinct usercode_number,activity_group, activity_name,place_channel_name,coupon_usercode from tmp.half_yr_coupon_tmp where date_diff < 180) t
group by usercode_number;

```
create table tmp.half_yr_coupon_tmp as
select t1.*, t2.*, t3.*, abs(datediff(biz_day,coupon_biz_day)) as date_diff
from (select *, to_date(concat(substr(p_biz_date,1,4),'-',substr(p_biz_date,5,2),'-',substr(p_biz_date,7))) as biz_day from tmp.churn_feats_with_mkt_new) t1
left join (select distinct p_day, user_code, activity_id,activity_name, activity_group, place_channel_name, to_date(concat(substr(p_day,1,4),'-',substr(p_day,5,2),'-',substr(p_day,7))) as coupon_biz_day  from dw_kfc.flat_coupon_activated_account_info
where p_day between 20220101 and 20230701) t2
on t1.usercode_number = t2.user_code 
left join (select distinct coupon_usercode,coupon_activityid from etl_kfc.t_kfc_coupon_redeem_dwbi_day where day between 20220101 and 20230701) t3
on t2.user_code = t3.coupon_usercode and t2.activity_id = t3.coupon_activityid;


create table if not exists tmp.sx_churn_feats_douyin_coupon as
select usercode_number, 
    count(case when activity_name like '%抖音%' or activity_group = 'KFC抖音团购券' then user_code else null end) as num_douyin,
    count(case when activity_name like '%抖音%' or activity_group = 'KFC抖音团购券' then coupon_usercode else null end) as douyin_redeem
from (select distinct usercode_number,activity_group, activity_name,place_channel_name,coupon_usercode,user_code from tmp.half_yr_coupon_tmp where date_diff < 180) t
where activity_name like '%抖音%' or activity_group = 'KFC抖音团购券'
group by usercode_number;

create table if not exists tmp.sx_churn_feats_meituan_coupon as
select usercode_number, 
    count(case when place_channel_name = '美团点评预付券' then user_code else null end) as num_meituan,
    count(case when place_channel_name = '美团点评预付券' then coupon_usercode else null end) as meituan_redeem
from (select distinct usercode_number,activity_group, activity_name,place_channel_name,coupon_usercode,user_code from tmp.half_yr_coupon_tmp where date_diff < 180) t
where place_channel_name = '美团点评预付券'
group by usercode_number;






---结合在一起
---beef_burger_tc/tc as beef_burger_tc_percent,whole_chicken_tc/tc as whole_chicken_tc_percent, coffee_tc/tc as coffee_tc_percent,ok_can_tc/tc as ok_can_tc_percent,
drop table tmp.sx_reduce_churn_risk_exploration_data_new;
create table tmp.sx_reduce_churn_risk_exploration_data_new as
select t1.usercode_number,  sex_flag_text, derived_age,his_family_flag, his_travel_flag, student_flag, his_value_seeker_flag, his_delivery_flag, his_lto_flag, his_working_day_lunch_flag, his_breakfast_flag, his_afternoon_flag, his_dinner_flag, his_coffee_flag, elderly_flag 
    ,all_redeemed_avg_party_size, all_num_coupon, all_num_redeem
    , ai_redeemed_avg_party_size,  ai_num_coupon, ai_num_redeem,
    working_trade_zone/tc as working_trade_zone,traditional_business_district/tc as traditional_business_district, university/tc as university, shopping_center/tc as shopping_center, traffic_center/tc as traffic_center, travel_trade_zone/tc as travel_trade_zone, shopping_mall/tc as shopping_mall, other_business_district/tc as other_business_district, 
    social_business_district/tc as social_business_district, city_public_traffic/tc as city_public_traffic, hospital/tc as hospital, market_trade_zone/tc as market_trade_zone,
    high_app_wechat_tc_percent
    , case when t7.num_douyin is null then 0 else num_douyin end as num_douyin, case when t7.usercode_number is null then -1 else douyin_redeem end as douyin_redeem,case when t7.usercode_number is null then -1 else douyin_redeem/tc end as douyin_redeem_percent
    , case when t8.num_meituan is null then 0 else num_meituan end as num_meituan, case when t8.usercode_number is null then -1 else meituan_redeem end as meituan_redeem,case when t8.usercode_number is null then -1 else meituan_redeem/tc end as meituan_redeem_percent
    , case when num_beef_burger > 0 then 1 else 0 end as if_beef_burger, case when num_whole_chicken > 0 then 1 else 0 end as if_whole_chicken,case when num_coffee > 0 then 1 else 0 end as if_coffee,case when num_ok_can > 0 then 1 else 0 end as if_ok_can,
    beef_burger_tc, whole_chicken_tc, coffee_tc, ok_can_tc, 
    num_sanfang, num_sanfang/tc as sanfang_percent,
    APP_tc/tc as app_tc_percent, wechat_tc/tc as wechat_tc_percent, APP_tc as num_app_tc, wechat_tc as num_wechat_tc
    , tc,ta,
    tc_diff, ta_diff, breakfast_tc, nonbreakfast_tc, morning_tc,lunch_tc,afternoon_tc,dinner_tc,latenight_tc,
    mon_tc, tue_tc, wen_tc, thu_tc, fri_tc, sat_tc, sun_tc, tier1_tc, tier2_tc, tier3_tc, tier4_tc, tier5_tc, tier6_tc, breakfast_maxta, nonbreakfast_maxta, 
    morning_maxta,lunch_maxta, afternoon_maxta, dinner_maxta, latenight_maxta, mon_maxta, tue_maxta, wen_maxta, thu_maxta, fri_maxta, sat_maxta, sun_maxta,breakfast_minta, nonbreakfast_minta, morning_minta,lunch_minta, afternoon_minta, dinner_minta, latenight_minta, mon_minta, tue_minta, wen_minta, thu_minta, fri_minta, sat_minta, sun_minta ,
    breakfast_avgta, nonbreakfast_avgta, morning_avgta,lunch_avgta, afternoon_avgta, dinner_avgta, latenight_avgta, mon_avgta, tue_avgta, wen_avgta, thu_avgta, fri_avgta, sat_avgta, sun_avgta,
    breakfast_sumta, nonbreakfast_sumta, morning_sumta,lunch_sumta, afternoon_sumta, dinner_sumta, latenight_sumta, mon_sumta, tue_sumta, wen_sumta, thu_sumta, fri_sumta, sat_sumta, sun_sumta,
    breakfast_std_ta, nonbreakfast_std_ta, morning_std_ta,lunch_std_ta, afternoon_std_ta, dinner_std_ta, latenight_std_ta, mon_std_ta, tue_std_ta, wen_std_ta, thu_std_ta, fri_std_ta, sat_std_ta, sun_std_ta, avg_discount, max_discount, min_discount, sum_discount, std_discount, 
    avg_ta_by_ps, std_ta_by_ps, avg_city_tier, max_city_tier, min_city_tier, std_city_tier, avg_party_size, max_party_size,min_party_size, cor_ta_da, distinct_daypart, distinct_city, distinct_work_day, distinct_store, delivery_tc, delivery_ta, avg_preorder_party_size,  max_preorder_party_size, min_preorder_party_size, std_preorder_party_size,
    side_sold, coffee_sold, congee_sold, nutrition_sold, panini_sold, riceroll_sold, dabing_sold, burger_sold, chickensnack_sold, cob_sold,csd_sold, eggtart_sold,icecream_sold,   sidefrenchfries_sold, sideothers_sold, tea_sold, twister_sold, wing_sold, waffle_sold, croissant_sold, nonfood_sold, pie_sold, juice_sold, rice_sold, lto_sold, city_tier_set_repeat50tc,
    party_size_set_repeat50tc,   day_of_week_set_repeat50tc,  daypart_name_set_repeat50tc,   occasion_name_set_repeat50tc,   churn_100, churn_180,returnable_churn_100,churn_60
from tmp.sx_distinct_churn_feature_diff_final t1
join tmp.sx_churn_feats_trade_zone t2
on t1.usercode_number = t2.usercode_number
join tmp.sx_churn_feats_product_class_type t3
on t1.usercode_number = t3.usercode_number
join tmp.sx_churn_feats_sanfang_type t4
on t1.usercode_number = t4.usercode_number
join tmp.sx_churn_feats_hightc_percent t5
on t1.usercode_number = t5.usercode_number
join tmp.sx_churn_feats_app_wechat_tc t6
on t1.usercode_number = t6.usercode_number
left join tmp.sx_churn_feats_douyin_coupon t7
on t1.usercode_number = t7.usercode_number
left join tmp.sx_churn_feats_meituan_coupon t8
on t1.usercode_number = t8.usercode_number;