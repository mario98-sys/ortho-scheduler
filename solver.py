# solver.py
import json
import calendar
from datetime import date
import random
import pandas as pd
from ortools.sat.python import cp_model

# Python date.weekday(): Mon=0 ... Sun=6
WEEKDAY = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}

ROLE_LABEL = {
    "first": "First On-Call",
    "second": "Second On-Call",
    "half": "Half Shift",
    "maccabi": "Maccabi Center",
}

ROLE_KEYS = ["first", "second", "half", "maccabi"]
WEEKEND_DAYS = {"Fri", "Sat"}  # Israel weekend


def load_cfg(path="input.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def month_days(year, month):
    _, last = calendar.monthrange(year, month)
    days = []
    for d in range(1, last + 1):
        wd = WEEKDAY[date(year, month, d).weekday()]
        days.append((d, wd))
    return days


def is_half_day(weekday: str) -> bool:
    return weekday in {"Sun", "Mon", "Tue", "Wed", "Thu"}


def is_role_required(role: str, weekday: str, cfg: dict) -> bool:
    rr = cfg.get("role_required", {}) or {}

    if role == "half":
        mode = rr.get("half", "SUN_THU")
        if mode == "SUN_THU":
            return is_half_day(weekday)
        if mode == "ALL_DAYS":
            return True
        if mode == "NONE":
            return False
        return is_half_day(weekday)

    mode = rr.get(role, "ALL_DAYS")
    if mode == "ALL_DAYS":
        return True
    if mode == "NONE":
        return False
    return True


def precheck_coverage(cfg):
    year, month = int(cfg["year"]), int(cfg["month"])
    days = month_days(year, month)
    roles = cfg.get("roles", ROLE_KEYS)
    people = [p for p in cfg.get("people", []) if not p.get("excluded", False)]

    holes = []
    for (day_num, weekday) in days:
        for r in roles:
            if not is_role_required(r, weekday, cfg):
                continue
            if r == "half" and not is_half_day(weekday):
                continue

            candidates = []
            for p in people:
                if day_num in set(p.get("unavailable_dates", [])):
                    continue
                if r not in set(p.get("can", [])):
                    continue
                candidates.append(p["name"])

            if len(candidates) == 0:
                holes.append((day_num, weekday, r))
    return holes


def compute_max_consecutive(worked_days_sorted):
    if not worked_days_sorted:
        return 0
    best = 1
    cur = 1
    for i in range(1, len(worked_days_sorted)):
        if worked_days_sorted[i] == worked_days_sorted[i - 1] + 1:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
    return best


def build_summary(cfg, schedule_rows):
    label_to_role = {v: k for k, v in ROLE_LABEL.items()}
    roles = ["first", "second", "half", "maccabi"]

    people = [p for p in cfg["people"] if not p.get("excluded", False)]
    people_by_name = {p["name"]: p for p in people}

    counts = {n: {r: 0 for r in roles} for n in people_by_name.keys()}
    worked_days = {n: [] for n in people_by_name.keys()}
    weekend_counts = {n: 0 for n in people_by_name.keys()}

    pref_hits_generic = {n: 0 for n in people_by_name.keys()}
    pref_hits_role = {n: {r: 0 for r in roles} for n in people_by_name.keys()}

    for row in schedule_rows:
        day_num = int(str(row["Date"]).split("/")[0])
        weekday = row["Day"]
        is_weekend = weekday in WEEKEND_DAYS

        for label, role_key in label_to_role.items():
            assignee = (row.get(label, "") or "").strip()
            if not assignee or assignee not in people_by_name:
                continue

            counts[assignee][role_key] += 1
            worked_days[assignee].append(day_num)
            if is_weekend:
                weekend_counts[assignee] += 1

            p = people_by_name[assignee]

            if day_num in set(p.get("prefer_dates", [])):
                pref_hits_generic[assignee] += 1
            if weekday in set(p.get("prefer_weekdays", [])):
                pref_hits_generic[assignee] += 1

            if day_num in set(p.get(f"prefer_{role_key}_dates", [])):
                pref_hits_role[assignee][role_key] += 1
            if weekday in set(p.get(f"prefer_{role_key}_weekdays", [])):
                pref_hits_role[assignee][role_key] += 1

    summary_rows = []
    for person_name in sorted(people_by_name.keys()):
        total = sum(counts[person_name].values())
        max_consec = compute_max_consecutive(sorted(set(worked_days[person_name])))

        summary_rows.append({
            "Name": person_name,
            "First": counts[person_name]["first"],
            "Second": counts[person_name]["second"],
            "Half": counts[person_name]["half"],
            "Maccabi": counts[person_name]["maccabi"],
            "Total": total,
            "Weekend Shifts": weekend_counts[person_name],
            "Max Consecutive Days": max_consec,
            "Preference Hits (Generic)": pref_hits_generic[person_name],
            "Preference Hits (Role-Specific)": sum(pref_hits_role[person_name].values()),
        })

    return pd.DataFrame(summary_rows)


def _build_existing_map(existing_rows):
    """(day_num, role_key) -> name"""
    label_to_role = {v: k for k, v in ROLE_LABEL.items()}
    m = {}
    for row in existing_rows or []:
        try:
            day_num = int(str(row["Date"]).split("/")[0])
        except Exception:
            continue
        for label, role_key in label_to_role.items():
            nm = (row.get(label, "") or "").strip()
            if nm:
                m[(day_num, role_key)] = nm
    return m


def solve(cfg, time_limit_sec=20):
    year, month = int(cfg["year"]), int(cfg["month"])
    rules = cfg.get("rules", {}) or {}
    weights = cfg.get("weights", {}) or {}
    locked = cfg.get("locked_assignments", []) or []

    # -------------------
    # Weights (defaults)
    # -------------------
    W_PREF_DATE = int(weights.get("pref_date", 30))
    W_PREF_WD = int(weights.get("pref_weekday", 15))
    W_ROLE_PREF_DATE = int(weights.get("role_pref_date", 30))
    W_ROLE_PREF_WD = int(weights.get("role_pref_weekday", 15))
    W_QUOTA_DEV = int(weights.get("quota_dev", 50))

    W_WEEKEND_BAL = int(weights.get("weekend_balance", 20))
    W_MACCABI_BAL = int(weights.get("maccabi_balance", 15))
    W_CONSEC_WEEKEND = int(weights.get("consecutive_weekend_penalty", 120))
    W_KEEP = int(weights.get("keep_existing", 120))

    # NEW: human-friendliness weights
    W_ADJ_ANY = int(weights.get("adjacent_any_work_penalty", 80))
    W_ADJ_SAME_ROLE = int(weights.get("adjacent_same_role_penalty", 140))
    W_REPEAT_FIRST_3 = int(weights.get("repeat_first_within3_penalty", 90))
    W_REPEAT_SECOND_3 = int(weights.get("repeat_second_within3_penalty", 70))
    W_REPEAT_MACCABI_2 = int(weights.get("repeat_maccabi_within2_penalty", 40))
    W_MAC_FRI_SAT_BONUS = int(weights.get("maccabi_fri_sat_same_bonus", 120))
    W_AVOID_SWAP = int(weights.get("avoid_first_second_swap_penalty", 60))

    # random tie-break (keep tiny)
    seed = int(cfg.get("random_seed", 0) or 0)
    rnd = random.Random(seed if seed != 0 else None)
    W_RANDOM = int(weights.get("random_tiebreak", 1))

    # Reschedule fields
    res_cfg = cfg.get("reschedule", {}) or {}
    res_enabled = bool(res_cfg.get("enabled", False))
    res_from_date = int(res_cfg.get("from_date", 0) or 0)
    res_sick_name = (res_cfg.get("sick_name", "") or "").strip()
    res_hard_lock_non_sick_future = bool(res_cfg.get("hard_lock_non_sick_future", True))

    existing_rows = cfg.get("existing_schedule", []) or []
    existing_map = _build_existing_map(existing_rows) if (res_enabled and existing_rows) else {}

    days = month_days(year, month)
    roles = cfg.get("roles", ROLE_KEYS) or ROLE_KEYS

    people = [p for p in cfg.get("people", []) if not p.get("excluded", False)]
    P = range(len(people))
    D = range(len(days))
    R = roles

    names = [people[i]["name"] for i in P]
    name_to_idx = {names[i]: i for i in P}

    can = [set(people[i].get("can", [])) for i in P]
    unavail = [set(people[i].get("unavailable_dates", [])) for i in P]

    prefer_dates = [set(people[i].get("prefer_dates", [])) for i in P]
    prefer_weekdays = [set(people[i].get("prefer_weekdays", [])) for i in P]

    role_pref_dates = []
    role_pref_weekdays = []
    for i in P:
        dmap = {}
        wmap = {}
        for r in roles:
            dmap[r] = set(people[i].get(f"prefer_{r}_dates", []))
            wmap[r] = set(people[i].get(f"prefer_{r}_weekdays", []))
        role_pref_dates.append(dmap)
        role_pref_weekdays.append(wmap)

    quota = [people[i].get("quota", {}) or {} for i in P]

    model = cp_model.CpModel()

    # -------------------
    # Decision vars x[p,d,r]
    # -------------------
    x = {}
    for p in P:
        for d in D:
            day_num, weekday = days[d]
            for r in R:
                if not is_role_required(r, weekday, cfg):
                    continue
                if r == "half" and not is_half_day(weekday):
                    continue
                if r not in can[p]:
                    continue
                if day_num in unavail[p]:
                    continue
                x[(p, d, r)] = model.NewBoolVar(f"x_p{p}_d{d}_r{r}")

    def var(p, d, r):
        return x.get((p, d, r), None)

    # worked[p,d] = 1 if person works any role that day
    worked = {}
    for p in P:
        for d in D:
            todays = [var(p, d, r) for r in R if var(p, d, r) is not None]
            worked[(p, d)] = model.NewBoolVar(f"worked_p{p}_d{d}")
            if not todays:
                model.Add(worked[(p, d)] == 0)
            else:
                model.Add(sum(todays) >= 1).OnlyEnforceIf(worked[(p, d)])
                model.Add(sum(todays) == 0).OnlyEnforceIf(worked[(p, d)].Not())

    # -------------------
    # HARD CONSTRAINTS
    # -------------------

    # (A) Fill required roles exactly once
    for d in D:
        day_num, weekday = days[d]
        for r in R:
            if not is_role_required(r, weekday, cfg):
                continue
            if r == "half" and not is_half_day(weekday):
                continue
            candidates = [var(p, d, r) for p in P if var(p, d, r) is not None]
            model.Add(sum(candidates) == 1)

    # (B) No two roles same day per person
    if rules.get("no_two_roles_same_day", True):
        for p in P:
            for d in D:
                todays = [var(p, d, r) for r in R if var(p, d, r) is not None]
                if todays:
                    model.Add(sum(todays) <= 1)

    # (C) Cooldown: first/second blocks next day first/second + optionally maccabi
    for p in P:
        for d in range(len(days) - 1):
            fs_today_vars = []
            for rr in ("first", "second"):
                v = var(p, d, rr)
                if v is not None:
                    fs_today_vars.append(v)

            if not fs_today_vars:
                continue

            fs_today = model.NewBoolVar(f"fs_today_p{p}_d{d}")
            model.Add(sum(fs_today_vars) >= 1).OnlyEnforceIf(fs_today)
            model.Add(sum(fs_today_vars) == 0).OnlyEnforceIf(fs_today.Not())

            if rules.get("first_second_blocks_next_day_first_second", True):
                for rr in ("first", "second"):
                    v_next = var(p, d + 1, rr)
                    if v_next is not None:
                        model.Add(v_next == 0).OnlyEnforceIf(fs_today)

            if rules.get("first_second_blocks_next_day_maccabi", True):
                v_next = var(p, d + 1, "maccabi")
                if v_next is not None:
                    model.Add(v_next == 0).OnlyEnforceIf(fs_today)

    # (D) Max consecutive working days (any role)
    max_consec = rules.get("max_consecutive_days", None)
    if max_consec is not None:
        max_consec = int(max_consec)
        window = max_consec + 1
        for p in P:
            for start in range(len(days) - window + 1):
                model.Add(sum(worked[(p, dd)] for dd in range(start, start + window)) <= max_consec)

    # (E) Minimum rest days after ANY shift (hard)
    min_rest = int(rules.get("min_rest_days_any", 0) or 0)
    if min_rest > 0:
        for p in P:
            for d in range(len(days) - 1):
                for k in range(1, min_rest + 1):
                    if d + k < len(days):
                        model.Add(worked[(p, d)] + worked[(p, d + k)] <= 1)

    # (F) Locked assignments (manual pins)
    for lk in locked:
        day_num = int(lk["date"])
        role = lk["role"]
        person_name = lk["name"]

        d_idx = None
        for i, (dn, wd) in enumerate(days):
            if dn == day_num:
                d_idx = i
                break
        if d_idx is None:
            continue

        if role == "half":
            _, wd = days[d_idx]
            if not is_half_day(wd):
                model.Add(0 == 1)

        if person_name not in name_to_idx:
            model.Add(0 == 1)
            continue

        p_idx = name_to_idx[person_name]
        v = var(p_idx, d_idx, role)
        if v is None:
            model.Add(0 == 1)
            continue

        model.Add(v == 1)
        for p in P:
            if p == p_idx:
                continue
            v2 = var(p, d_idx, role)
            if v2 is not None:
                model.Add(v2 == 0)

    # (G) Partial re-schedule locks (keep existing)
    if res_enabled and existing_map:
        for d in D:
            day_num, weekday = days[d]
            for r in R:
                old_name = existing_map.get((day_num, r))
                if not old_name:
                    continue

                # always lock before sick date
                if day_num < res_from_date:
                    if old_name in name_to_idx:
                        p_idx = name_to_idx[old_name]
                        v = var(p_idx, d, r)
                        if v is None:
                            model.Add(0 == 1)
                        else:
                            model.Add(v == 1)
                    else:
                        model.Add(0 == 1)
                    continue

                # from sick date onward
                if day_num >= res_from_date:
                    if old_name == res_sick_name:
                        continue
                    if res_hard_lock_non_sick_future:
                        if old_name in name_to_idx:
                            p_idx = name_to_idx[old_name]
                            v = var(p_idx, d, r)
                            if v is not None:
                                model.Add(v == 1)

    # -------------------
    # SOFT (OPTIMIZATION)
    # -------------------
    objective_terms = []
    penalty_terms = []

    def add_and_bool(a, b, name):
        """creates z = a AND b"""
        z = model.NewBoolVar(name)
        model.AddBoolAnd([a, b]).OnlyEnforceIf(z)
        model.AddBoolOr([a.Not(), b.Not()]).OnlyEnforceIf(z.Not())
        return z

    # (1) Preferences + tiny random tie-break
    for p in P:
        for d in D:
            day_num, weekday = days[d]
            for r in R:
                v = var(p, d, r)
                if v is None:
                    continue

                if day_num in prefer_dates[p]:
                    objective_terms.append(W_PREF_DATE * v)
                if weekday in prefer_weekdays[p]:
                    objective_terms.append(W_PREF_WD * v)

                if day_num in role_pref_dates[p].get(r, set()):
                    objective_terms.append(W_ROLE_PREF_DATE * v)
                if weekday in role_pref_weekdays[p].get(r, set()):
                    objective_terms.append(W_ROLE_PREF_WD * v)

                if W_RANDOM > 0:
                    objective_terms.append(rnd.randint(0, W_RANDOM) * v)

    # (2) Keep existing schedule (soft reward)
    if res_enabled and existing_map:
        for d in D:
            day_num, weekday = days[d]
            for r in R:
                old_name = existing_map.get((day_num, r))
                if not old_name:
                    continue
                if day_num >= res_from_date and old_name == res_sick_name:
                    continue
                if old_name not in name_to_idx:
                    continue
                p_idx = name_to_idx[old_name]
                v = var(p_idx, d, r)
                if v is not None:
                    objective_terms.append(W_KEEP * v)

    # (3) Quotas deviation
    for p in P:
        for r in R:
            if r not in quota[p]:
                continue
            target = int(quota[p][r])

            assigned_vars = [var(p, d, r) for d in D if var(p, d, r) is not None]
            assigned_sum = model.NewIntVar(0, len(days), f"count_p{p}_r{r}")
            model.Add(assigned_sum == (sum(assigned_vars) if assigned_vars else 0))

            dev = model.NewIntVar(0, len(days), f"dev_p{p}_r{r}")
            model.AddAbsEquality(dev, assigned_sum - target)
            penalty_terms.append((W_QUOTA_DEV, dev))

    # (4) WEEKEND fairness
    weekend_day_indices = [d for d in D if days[d][1] in WEEKEND_DAYS]
    if weekend_day_indices:
        weekend_counts = []
        for p in P:
            wsum = model.NewIntVar(0, len(weekend_day_indices), f"weekend_count_p{p}")
            model.Add(wsum == sum(worked[(p, d)] for d in weekend_day_indices))
            weekend_counts.append(wsum)

        n_people = max(1, len(list(P)))
        avg_floor = len(weekend_day_indices) // n_people
        avg_ceil = (len(weekend_day_indices) + n_people - 1) // n_people

        for p in P:
            dev_low = model.NewIntVar(0, len(weekend_day_indices), f"weekend_dev_low_p{p}")
            dev_high = model.NewIntVar(0, len(weekend_day_indices), f"weekend_dev_high_p{p}")
            model.Add(dev_low >= avg_floor - weekend_counts[p])
            model.Add(dev_low >= 0)
            model.Add(dev_high >= weekend_counts[p] - avg_ceil)
            model.Add(dev_high >= 0)
            penalty_terms.append((W_WEEKEND_BAL, dev_low))
            penalty_terms.append((W_WEEKEND_BAL, dev_high))

        # consecutive weekend blocks penalty
        fri_indices = [d for d in D if days[d][1] == "Fri"]
        weekend_blocks = []
        for d_fri in fri_indices:
            d_sat = None
            if d_fri + 1 < len(days) and days[d_fri + 1][1] == "Sat":
                d_sat = d_fri + 1
            weekend_blocks.append((d_fri, d_sat))

        works_weekend = {}
        for p in P:
            for w, (d_fri, d_sat) in enumerate(weekend_blocks):
                v = model.NewBoolVar(f"works_weekend_p{p}_w{w}")
                parts = [worked[(p, d_fri)]]
                if d_sat is not None:
                    parts.append(worked[(p, d_sat)])

                model.Add(sum(parts) >= 1).OnlyEnforceIf(v)
                model.Add(sum(parts) == 0).OnlyEnforceIf(v.Not())
                works_weekend[(p, w)] = v

        for p in P:
            for w in range(len(weekend_blocks) - 1):
                both = add_and_bool(works_weekend[(p, w)], works_weekend[(p, w + 1)], f"consec_weekends_p{p}_w{w}")
                penalty_terms.append((W_CONSEC_WEEKEND, both))

    # (5) Maccabi balancing
    maccabi_capable = [p for p in P if "maccabi" in can[p]]
    if maccabi_capable:
        m_counts = []
        for p in maccabi_capable:
            vs = [var(p, d, "maccabi") for d in D if var(p, d, "maccabi") is not None]
            c = model.NewIntVar(0, len(days), f"maccabi_count_p{p}")
            model.Add(c == (sum(vs) if vs else 0))
            m_counts.append((p, c))

        avg_m = int(round(len(days) / max(1, len(maccabi_capable))))
        for p, c in m_counts:
            dev = model.NewIntVar(0, len(days), f"maccabi_dev_p{p}")
            model.AddAbsEquality(dev, c - avg_m)
            penalty_terms.append((W_MACCABI_BAL, dev))

    # -------------------------
    # NEW: "doctor-style" soft constraints
    # -------------------------

    # A) discourage any adjacent-day work for same person (unless forced)
    if W_ADJ_ANY > 0:
        for p in P:
            for d in range(1, len(days)):
                both = add_and_bool(worked[(p, d - 1)], worked[(p, d)], f"adj_any_p{p}_d{d}")
                penalty_terms.append((W_ADJ_ANY, both))

    # B) discourage same role on adjacent days (stronger)
    if W_ADJ_SAME_ROLE > 0:
        for p in P:
            for d in range(1, len(days)):
                for rr in ["first", "second", "maccabi", "half"]:
                    v1 = var(p, d - 1, rr)
                    v2 = var(p, d, rr)
                    if v1 is None or v2 is None:
                        continue
                    both = add_and_bool(v1, v2, f"adj_same_role_{rr}_p{p}_d{d}")
                    penalty_terms.append((W_ADJ_SAME_ROLE, both))

    # C) discourage repeating FIRST within 3 days (human rotation)
    if W_REPEAT_FIRST_3 > 0:
        for p in P:
            for d in range(len(days)):
                v_today = var(p, d, "first")
                if v_today is None:
                    continue
                for k in (1, 2, 3):
                    if d + k >= len(days):
                        continue
                    v_next = var(p, d + k, "first")
                    if v_next is None:
                        continue
                    both = add_and_bool(v_today, v_next, f"repeat_first3_p{p}_d{d}_k{k}")
                    penalty_terms.append((W_REPEAT_FIRST_3, both))

    # D) discourage repeating SECOND within 3 days
    if W_REPEAT_SECOND_3 > 0:
        for p in P:
            for d in range(len(days)):
                v_today = var(p, d, "second")
                if v_today is None:
                    continue
                for k in (1, 2, 3):
                    if d + k >= len(days):
                        continue
                    v_next = var(p, d + k, "second")
                    if v_next is None:
                        continue
                    both = add_and_bool(v_today, v_next, f"repeat_second3_p{p}_d{d}_k{k}")
                    penalty_terms.append((W_REPEAT_SECOND_3, both))

    # E) discourage repeating MACCABI within 2 days (milder, but helps)
    if W_REPEAT_MACCABI_2 > 0:
        for p in P:
            for d in range(len(days)):
                v_today = var(p, d, "maccabi")
                if v_today is None:
                    continue
                for k in (1, 2):
                    if d + k >= len(days):
                        continue
                    v_next = var(p, d + k, "maccabi")
                    if v_next is None:
                        continue
                    both = add_and_bool(v_today, v_next, f"repeat_maccabi2_p{p}_d{d}_k{k}")
                    penalty_terms.append((W_REPEAT_MACCABI_2, both))

    # F) encourage same MACCABI across Fri+Sat (doctor-style weekend block)
    if W_MAC_FRI_SAT_BONUS > 0:
        for d in range(len(days) - 1):
            if days[d][1] == "Fri" and days[d + 1][1] == "Sat":
                for p in P:
                    v_fri = var(p, d, "maccabi")
                    v_sat = var(p, d + 1, "maccabi")
                    if v_fri is None or v_sat is None:
                        continue
                    both = add_and_bool(v_fri, v_sat, f"maccabi_frisat_same_p{p}_d{d}")
                    objective_terms.append(W_MAC_FRI_SAT_BONUS * both)

    # G) discourage swapping FIRST<->SECOND across adjacent days (less “random looking”)
    if W_AVOID_SWAP > 0:
        for p in P:
            for d in range(1, len(days)):
                v_prev_first = var(p, d - 1, "first")
                v_prev_second = var(p, d - 1, "second")
                v_now_first = var(p, d, "first")
                v_now_second = var(p, d, "second")

                # prev first -> now second
                if v_prev_first is not None and v_now_second is not None:
                    both = add_and_bool(v_prev_first, v_now_second, f"swap_f_s_p{p}_d{d}")
                    penalty_terms.append((W_AVOID_SWAP, both))
                # prev second -> now first
                if v_prev_second is not None and v_now_first is not None:
                    both = add_and_bool(v_prev_second, v_now_first, f"swap_s_f_p{p}_d{d}")
                    penalty_terms.append((W_AVOID_SWAP, both))

    # -------------------
    # Final objective
    # -------------------
    model.Maximize(sum(objective_terms) - sum(w * dev for (w, dev) in penalty_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_sec)
    solver.parameters.num_search_workers = 8

    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None, "INFEASIBLE"

    # Build schedule rows
    out = []
    for d in D:
        day_num, weekday = days[d]
        row = {
            "Date": f"{day_num:02d}/{month:02d}/{year}",
            "Day": weekday,
            ROLE_LABEL["first"]: "",
            ROLE_LABEL["second"]: "",
            ROLE_LABEL["half"]: "",
            ROLE_LABEL["maccabi"]: "",
        }

        for r in R:
            if r == "half" and not is_half_day(weekday):
                continue
            if not is_role_required(r, weekday, cfg):
                continue

            for p in P:
                v = var(p, d, r)
                if v is not None and solver.Value(v) == 1:
                    row[ROLE_LABEL[r]] = names[p]
                    break

        out.append(row)

    return out, "OK"


def export_excel(rows, cfg, path="output.xlsx"):
    schedule_df = pd.DataFrame(rows)
    summary_df = build_summary(cfg, rows)

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        schedule_df.to_excel(writer, index=False, sheet_name="Schedule")
        summary_df.to_excel(writer, index=False, sheet_name="Summary")

    return path


def export_csv(rows, path="output.csv"):
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


if __name__ == "__main__":
    cfg = load_cfg("input.json")

    holes = precheck_coverage(cfg)
    if holes:
        print("Coverage holes (guarantee INFEASIBLE):")
        for day_num, weekday, role in holes:
            print(f"  - {day_num:02d}/{cfg['month']:02d}/{cfg['year']} ({weekday}) role={role}")
        print("Fix roles/can/unavailability/exclusions first.")

    rows, status = solve(cfg)
    if status != "OK":
        print("No valid schedule:", status)
    else:
        print("Saved:", export_excel(rows, cfg, "output.xlsx"))
        print("Saved:", export_csv(rows, "output.csv"))
