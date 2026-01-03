# app.py
import os
import json
import copy
import pandas as pd
import streamlit as st

from solver import solve, build_summary, precheck_coverage

# -----------------------------
# Constants / Labels
# -----------------------------
ROLE_KEYS = ["first", "second", "half", "maccabi"]

ROLE_NAMES_HE = {
    "first": "תורן ראשון",
    "second": "תורן שני",
    "half": "תורן חצי",
    "maccabi": "מוקד מכבי",
}

WEEKDAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
WEEKDAYS_HE = {
    "Sun": "ראשון",
    "Mon": "שני",
    "Tue": "שלישי",
    "Wed": "רביעי",
    "Thu": "חמישי",
    "Fri": "שישי",
    "Sat": "שבת",
}

# Solver uses these English output labels
SOLVER_COLS = {
    "date": "Date",
    "day": "Day",
    "first": "First On-Call",
    "second": "Second On-Call",
    "half": "Half Shift",
    "maccabi": "Maccabi Center",
}

WEEKEND_DAYS = {"Fri", "Sat"}  # Israel weekend

DEFAULT_CFG = {
    "year": 2026,
    "month": 3,
    "roles": ROLE_KEYS,
    "role_required": {
        "first": "ALL_DAYS",
        "second": "ALL_DAYS",
        "maccabi": "ALL_DAYS",
        "half": "SUN_THU"
    },
    "rules": {
        "no_two_roles_same_day": True,
        "first_second_blocks_next_day_first_second": True,
        "first_second_blocks_next_day_maccabi": True,
        "maccabi_allows_next_day_anything": True,
        "half_can_follow_half": True,
        "max_consecutive_days": 2,
        "min_rest_days_any": 0
    },
    "weights": {
        "pref_date": 30,
        "pref_weekday": 15,
        "role_pref_date": 30,
        "role_pref_weekday": 15,
        "quota_dev": 50,
        "weekend_balance": 20,
        "maccabi_balance": 15,
        "consecutive_weekend_penalty": 120,
        "keep_existing": 120,
        "random_tiebreak": 2
    },
    "locked_assignments": [],
    "people": []
}

TEMPLATES_DIR = "templates"


# -----------------------------
# Utilities
# -----------------------------
def ensure_templates_dir():
    os.makedirs(TEMPLATES_DIR, exist_ok=True)


def load_cfg(path="input.json"):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return copy.deepcopy(DEFAULT_CFG)


def save_cfg(cfg, path="input.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def parse_int_list(text):
    """Accepts: '1,2,3' or '1-5, 8, 10-12'."""
    text = (text or "").strip()
    if not text:
        return []
    out = set()
    parts = [p.strip() for p in text.split(",") if p.strip()]
    for p in parts:
        if "-" in p:
            a, b = p.split("-", 1)
            a = int(a.strip())
            b = int(b.strip())
            for x in range(min(a, b), max(a, b) + 1):
                out.add(x)
        else:
            out.add(int(p))
    return sorted(out)


def int_list_to_text(lst):
    return ",".join(str(x) for x in (lst or []))


def month_last_day(year, month):
    import calendar
    _, last = calendar.monthrange(year, month)
    return last


def reset_person_inputs(cfg):
    """
    Resets ONLY per-person fields (as requested):
    - Availability (unavailable_dates)
    - Generic preferences (prefer_dates, prefer_weekdays)
    - Maccabi preferences (prefer_maccabi_dates, prefer_maccabi_weekdays)
    - Quotas (quota)
    Keeps:
    - name
    - can (roles)
    - excluded
    """
    for p in cfg.get("people", []):
        p["unavailable_dates"] = []
        p["prefer_dates"] = []
        p["prefer_weekdays"] = []
        p["prefer_maccabi_dates"] = []
        p["prefer_maccabi_weekdays"] = []
        p["quota"] = {}

        # (future proof) role-specific prefs, if exist
        for rk in ROLE_KEYS:
            if f"prefer_{rk}_dates" in p:
                p[f"prefer_{rk}_dates"] = []
            if f"prefer_{rk}_weekdays" in p:
                p[f"prefer_{rk}_weekdays"] = []


def schedule_rows_to_hebrew_df(rows):
    df = pd.DataFrame(rows).copy()

    # Convert weekday to Hebrew
    if SOLVER_COLS["day"] in df.columns:
        df[SOLVER_COLS["day"]] = df[SOLVER_COLS["day"]].map(lambda d: WEEKDAYS_HE.get(d, d))

    rename_map = {
        SOLVER_COLS["date"]: "תאריך",
        SOLVER_COLS["day"]: "יום",
        SOLVER_COLS["first"]: "תורן ראשון",
        SOLVER_COLS["second"]: "תורן שני",
        SOLVER_COLS["half"]: "תורן חצי",
        SOLVER_COLS["maccabi"]: "מוקד מכבי",
    }
    df = df.rename(columns=rename_map)

    cols = ["תאריך", "יום", "תורן ראשון", "תורן שני", "תורן חצי", "מוקד מכבי"]
    cols = [c for c in cols if c in df.columns]
    return df[cols]


def summary_to_hebrew_df(summary_df: pd.DataFrame):
    rename_map = {
        "Name": "שם",
        "First": "תורן ראשון",
        "Second": "תורן שני",
        "Half": "תורן חצי",
        "Maccabi": "מוקד מכבי",
        "Total": "סה״כ משמרות",
        "Weekend Shifts": "סופ״ש",
        "Max Consecutive Days": "מקס׳ רצופים",
        "Preference Hits (Generic)": "פגיעות העדפה (כללי)",
        "Preference Hits (Role-Specific)": "פגיעות העדפה (תפקיד)",
    }
    out = summary_df.copy().rename(columns=rename_map)
    cols = [
        "שם", "תורן ראשון", "תורן שני", "תורן חצי", "מוקד מכבי",
        "סה״כ משמרות", "סופ״ש", "מקס׳ רצופים",
        "פגיעות העדפה (כללי)", "פגיעות העדפה (תפקיד)"
    ]
    cols = [c for c in cols if c in out.columns]
    return out[cols]


def export_hebrew_files(rows, cfg, excel_path="output.xlsx", csv_path="output.csv"):
    schedule_he = schedule_rows_to_hebrew_df(rows)
    summary_he = summary_to_hebrew_df(build_summary(cfg, rows))

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        schedule_he.to_excel(writer, index=False, sheet_name="לוח עבודה")
        summary_he.to_excel(writer, index=False, sheet_name="סיכום")

    schedule_he.to_csv(csv_path, index=False, encoding="utf-8-sig")

    with open(excel_path, "rb") as f:
        excel_bytes = f.read()
    with open(csv_path, "rb") as f:
        csv_bytes = f.read()

    return schedule_he, summary_he, excel_bytes, csv_bytes


def validate_schedule(cfg, rows):
    """Basic validation report (Hebrew)."""
    issues = []
    rules = cfg.get("rules", {})
    max_consec = int(rules.get("max_consecutive_days", 2))
    min_rest = int(rules.get("min_rest_days_any", 0))

    df = pd.DataFrame(rows)
    role_cols = [SOLVER_COLS["first"], SOLVER_COLS["second"], SOLVER_COLS["half"], SOLVER_COLS["maccabi"]]

    # No double role same day
    for _, r in df.iterrows():
        day = r[SOLVER_COLS["date"]]
        assigned = []
        for c in role_cols:
            nm = str(r.get(c, "")).strip()
            if nm:
                assigned.append(nm)
        if len(assigned) != len(set(assigned)):
            issues.append({"סוג": "כפילות באותו יום", "תאריך": day, "פרטים": f"{assigned}"})

    # Half should be empty Fri/Sat
    for _, r in df.iterrows():
        weekday = str(r.get(SOLVER_COLS["day"], "")).strip()
        half = str(r.get(SOLVER_COLS["half"], "")).strip()
        if weekday in ["Fri", "Sat"] and half:
            issues.append({"סוג": "תורן חצי בשישי/שבת", "תאריך": r[SOLVER_COLS["date"]], "פרטים": f"יום={weekday}, שם={half}"})

    # Max consecutive + min rest (per person)
    worked = {}
    for _, r in df.iterrows():
        day_num = int(str(r[SOLVER_COLS["date"]]).split("/")[0])
        names = set(str(r.get(c, "")).strip() for c in role_cols)
        names.discard("")
        for nm in names:
            worked.setdefault(nm, []).append(day_num)

    for nm, days_list in worked.items():
        days_list = sorted(set(days_list))

        # max consecutive
        best = 1
        streak = 1
        for i in range(1, len(days_list)):
            if days_list[i] == days_list[i - 1] + 1:
                streak += 1
                best = max(best, streak)
            else:
                streak = 1
        if best > max_consec:
            issues.append({"סוג": "יותר מדי ימים רצופים", "תאריך": "-", "פרטים": f"{nm} רצף={best} (מותר {max_consec})"})

        # min rest
        if min_rest > 0:
            s = set(days_list)
            for d0 in days_list:
                for k in range(1, min_rest + 1):
                    if d0 + k in s:
                        issues.append({"סוג": "הפרת מינימום מנוחה", "תאריך": "-", "פרטים": f"{nm} עבד בימים {d0} ו-{d0 + k} (מינימום {min_rest})"})
                        break

    return issues


def schedule_diff_df(old_rows, new_rows):
    """Shows only changed cells between old and new schedules."""
    if not old_rows or not new_rows:
        return pd.DataFrame(columns=["תאריך", "יום", "עמדה", "לפני", "אחרי"])

    old = pd.DataFrame(old_rows).copy().set_index(SOLVER_COLS["date"])
    new = pd.DataFrame(new_rows).copy().set_index(SOLVER_COLS["date"])

    role_cols = [SOLVER_COLS["first"], SOLVER_COLS["second"], SOLVER_COLS["half"], SOLVER_COLS["maccabi"]]
    role_he = {
        SOLVER_COLS["first"]: "תורן ראשון",
        SOLVER_COLS["second"]: "תורן שני",
        SOLVER_COLS["half"]: "תורן חצי",
        SOLVER_COLS["maccabi"]: "מוקד מכבי",
    }

    common_dates = [d for d in old.index if d in new.index]
    changes = []

    for dt in common_dates:
        day_en = str(new.loc[dt].get(SOLVER_COLS["day"], "")).strip()
        day_he = WEEKDAYS_HE.get(day_en, day_en)

        for col in role_cols:
            old_v = str(old.loc[dt].get(col, "")).strip()
            new_v = str(new.loc[dt].get(col, "")).strip()
            if old_v != new_v:
                changes.append({
                    "תאריך": dt,
                    "יום": day_he,
                    "עמדה": role_he.get(col, col),
                    "לפני": old_v,
                    "אחרי": new_v
                })

    return pd.DataFrame(changes)


def apply_sick_leave_to_cfg(cfg, sick_name: str, from_date: int):
    """
    Removes a person from availability starting from from_date (inclusive)
    by adding all dates [from_date..end_of_month] to their unavailable_dates.
    """
    year = int(cfg.get("year"))
    month = int(cfg.get("month"))
    last = month_last_day(year, month)

    for p in cfg.get("people", []):
        if p.get("name") == sick_name:
            existing = set(p.get("unavailable_dates", []))
            for d in range(from_date, last + 1):
                existing.add(d)
            p["unavailable_dates"] = sorted(existing)
            break


# -----------------------------
# Streamlit setup
# -----------------------------
st.set_page_config(page_title="שיבוץ מתמחים אורטופדיה", layout="wide")

# RTL + MOBILE FIX:
# - Hide Streamlit sidebar entirely on mobile to prevent “ghost sidebar” layout bugs
# - Make buttons full width on mobile
st.markdown(
    """
    <style>
      html, body, [class*="css"]  { direction: rtl; }
      .stApp { direction: rtl; }
      h1, h2, h3, h4, h5, h6, p, div, span, label { text-align: right; }
      .stDataFrame { direction: rtl; }

      /* Optional: slightly tighter desktop spacing */
      .block-container { padding-top: 1.2rem; }

      /* MOBILE: hide sidebar completely so it can’t mess layout when “closed” */
      @media (max-width: 768px) {
        section[data-testid="stSidebar"] { display: none !important; }
        div[data-testid="collapsedControl"] { display: none !important; }

        .block-container { padding: 0.8rem 0.8rem 3rem 0.8rem; }
        h1 { font-size: 1.35rem !important; }
        h2 { font-size: 1.15rem !important; }
        h3 { font-size: 1.05rem !important; }

        .stButton>button, .stDownloadButton>button { width: 100%; }
      }
    </style>
    """,
    unsafe_allow_html=True
)

# session state
if "reset_pending" not in st.session_state:
    st.session_state.reset_pending = False
if "generated" not in st.session_state:
    st.session_state.generated = False
if "last_rows" not in st.session_state:
    st.session_state.last_rows = None
if "excel_bytes" not in st.session_state:
    st.session_state.excel_bytes = None
if "csv_bytes" not in st.session_state:
    st.session_state.csv_bytes = None
if "schedule_he" not in st.session_state:
    st.session_state.schedule_he = None
if "summary_he" not in st.session_state:
    st.session_state.summary_he = None
if "diff_df" not in st.session_state:
    st.session_state.diff_df = None

ensure_templates_dir()

st.title("שיבוץ מתמחים באורטופדיה")

cfg = load_cfg("input.json")
cfg.setdefault("rules", copy.deepcopy(DEFAULT_CFG["rules"]))
cfg.setdefault("weights", copy.deepcopy(DEFAULT_CFG["weights"]))
cfg.setdefault("locked_assignments", [])
cfg.setdefault("people", [])

# -----------------------------
# Settings (mobile-friendly)
# (We DO NOT use st.sidebar anymore)
# -----------------------------
with st.expander("⚙️ הגדרות", expanded=False):
    cY, cM = st.columns(2)
    with cY:
        cfg["year"] = int(st.number_input("שנה", min_value=2000, max_value=2100, value=int(cfg.get("year", 2026)), key="cfg_year"))
    with cM:
        cfg["month"] = int(st.number_input("חודש", min_value=1, max_value=12, value=int(cfg.get("month", 3)), key="cfg_month"))

    st.markdown("### חוקים")
    r = cfg["rules"]
    r["no_two_roles_same_day"] = st.checkbox("אין 2 עמדות באותו יום לאותו אדם", value=bool(r.get("no_two_roles_same_day", True)), key="rule_no2")
    r["first_second_blocks_next_day_first_second"] = st.checkbox("תורן 1/2 חוסם תורן 1/2 למחרת", value=bool(r.get("first_second_blocks_next_day_first_second", True)), key="rule_fs_next_fs")
    r["first_second_blocks_next_day_maccabi"] = st.checkbox("תורן 1/2 חוסם מכבי למחרת", value=bool(r.get("first_second_blocks_next_day_maccabi", True)), key="rule_fs_next_m")
    cA, cB = st.columns(2)
    with cA:
        r["max_consecutive_days"] = int(st.number_input("מקסימום ימים רצופים", min_value=1, max_value=10, value=int(r.get("max_consecutive_days", 2)), key="rule_max_consec"))
    with cB:
        r["min_rest_days_any"] = int(st.number_input("מינימום ימי מנוחה בין משמרות (0=אין)", min_value=0, max_value=7, value=int(r.get("min_rest_days_any", 0)), key="rule_min_rest"))
    cfg["rules"] = r

    st.divider()

    st.markdown("### תבניות")
    template_files = sorted([f for f in os.listdir(TEMPLATES_DIR) if f.endswith(".json")])
    selected_template = st.selectbox("בחר תבנית לטעינה", ["(ללא)"] + template_files, key="tpl_select")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("טען תבנית", use_container_width=True, key="tpl_load_btn"):
            if selected_template != "(ללא)":
                with open(os.path.join(TEMPLATES_DIR, selected_template), "r", encoding="utf-8") as f:
                    cfg_loaded = json.load(f)
                cfg = cfg_loaded
                st.success("נטענה תבנית ✅")
    with c2:
        tpl_name = st.text_input("שם לשמירה", value=f"schedule_{cfg['year']}_{cfg['month']:02d}", key="tpl_save_name")
        if st.button("שמור תבנית", use_container_width=True, key="tpl_save_btn"):
            if tpl_name.strip():
                path = os.path.join(TEMPLATES_DIR, f"{tpl_name.strip()}.json")
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(cfg, f, ensure_ascii=False, indent=2)
                st.success("נשמרה תבנית ✅")

    if st.button("שכפל חודש קודם", use_container_width=True, key="tpl_clone_prev"):
        y = int(cfg["year"])
        m = int(cfg["month"])
        if m == 1:
            py, pm = y - 1, 12
        else:
            py, pm = y, m - 1
        candidate = os.path.join(TEMPLATES_DIR, f"schedule_{py}_{pm:02d}.json")
        if os.path.exists(candidate):
            with open(candidate, "r", encoding="utf-8") as f:
                base = json.load(f)
            base["year"] = y
            base["month"] = m
            cfg = base
            st.success("שוכפל חודש קודם ✅")
        else:
            st.warning("לא נמצאה תבנית לחודש קודם בשם ברירת מחדל.")

    st.divider()
    if st.button("💾 שמור הגדרות (input.json)", use_container_width=True, key="save_cfg_top"):
        save_cfg(cfg, "input.json")
        st.success("נשמר input.json ✅")

# -----------------------------
# Tabs
# -----------------------------
tab_people, tab_locks, tab_generate = st.tabs(["מתמחים", "נעילות", "יצירה ותוצאות"])

# -----------------------------
# People tab
# -----------------------------
with tab_people:
    st.subheader("רשימת מתמחים")

    if st.button("➕ הוסף/י מתמחה", use_container_width=True):
        cfg["people"].append({
            "name": f"מתמחה חדש {len(cfg['people']) + 1}",
            "can": ["first"],
            "excluded": False,
            "unavailable_dates": [],
            "prefer_dates": [],
            "prefer_weekdays": [],
            "prefer_maccabi_dates": [],
            "prefer_maccabi_weekdays": [],
            "quota": {}
        })

    to_delete = None
    for i, p in enumerate(cfg["people"]):
        title = f"{i + 1}. {p.get('name', '(ללא שם)')}"
        with st.expander(title, expanded=False):
            # Mobile friendly: stack into rows (avoid tight 3-column on phone)
            p["name"] = st.text_input("שם", value=p.get("name", ""), key=f"name_{i}")
            cA, cB = st.columns(2)
            with cA:
                p["excluded"] = st.checkbox("הוצאה זמנית", value=bool(p.get("excluded", False)), key=f"ex_{i}")
            with cB:
                if st.button("🗑️ מחיקה", key=f"del_{i}", use_container_width=True):
                    to_delete = i

            p["can"] = st.multiselect(
                "עמדות",
                ROLE_KEYS,
                default=p.get("can", []),
                format_func=lambda r0: f"{ROLE_NAMES_HE[r0]} ({r0})",
                key=f"can_{i}"
            )

            st.markdown("#### זמינות")
            p["unavailable_dates"] = parse_int_list(st.text_input(
                "תאריכים שאי אפשר (1-5, 8, 10-12)",
                value=int_list_to_text(p.get("unavailable_dates", [])),
                key=f"unav_{i}"
            ))

            st.markdown("#### העדפות כלליות")
            p["prefer_dates"] = parse_int_list(st.text_input(
                "תאריכים מועדפים (לכל העמדות)",
                value=int_list_to_text(p.get("prefer_dates", [])),
                key=f"pd_{i}"
            ))
            p["prefer_weekdays"] = st.multiselect(
                "ימים מועדפים (לכל העמדות)",
                WEEKDAYS,
                default=p.get("prefer_weekdays", []),
                format_func=lambda d: WEEKDAYS_HE[d],
                key=f"pw_{i}"
            )

            st.markdown("#### העדפות למוקד מכבי")
            p["prefer_maccabi_dates"] = parse_int_list(st.text_input(
                "תאריכים מועדפים למכבי",
                value=int_list_to_text(p.get("prefer_maccabi_dates", [])),
                key=f"pmd_{i}"
            ))
            p["prefer_maccabi_weekdays"] = st.multiselect(
                "ימים מועדפים למכבי",
                WEEKDAYS,
                default=p.get("prefer_maccabi_weekdays", []),
                format_func=lambda d: WEEKDAYS_HE[d],
                key=f"pmw_{i}"
            )

            st.markdown("#### יעדים (Quota) — לא חובה")
            q = p.get("quota", {})

            # Mobile friendly: 2 columns instead of 4
            q1, q2 = st.columns(2)
            with q1:
                q_first = st.text_input("תורן ראשון", value=str(q.get("first", "")), key=f"qf_{i}")
                q_half = st.text_input("תורן חצי", value=str(q.get("half", "")), key=f"qh_{i}")
            with q2:
                q_second = st.text_input("תורן שני", value=str(q.get("second", "")), key=f"qs_{i}")
                q_m = st.text_input("מכבי", value=str(q.get("maccabi", "")), key=f"qm_{i}")

            quota_obj = {}
            for rk, val in [("first", q_first), ("second", q_second), ("half", q_half), ("maccabi", q_m)]:
                val = (val or "").strip()
                if val:
                    try:
                        quota_obj[rk] = int(val)
                    except ValueError:
                        pass
            p["quota"] = quota_obj

    if to_delete is not None:
        cfg["people"].pop(to_delete)

    st.divider()
    c_save, c_reset = st.columns(2)

    with c_save:
        if st.button("💾 שמור הגדרות (input.json)", use_container_width=True, key="save_cfg_people"):
            save_cfg(cfg, "input.json")
            st.success("נשמר input.json ✅")

    with c_reset:
        if not st.session_state.reset_pending:
            if st.button("♻️ איפוס (זמינות + יעדים + העדפות)", use_container_width=True, key="reset_btn"):
                st.session_state.reset_pending = True
                st.warning(
                    "אזהרה: פעולה זו תאפס עבור כל המתמחים:\n"
                    "- זמינות (תאריכים שאי אפשר)\n"
                    "- יעדים (Quota)\n"
                    "- העדפות כלליות\n"
                    "- העדפות מכבי\n\n"
                    "היא לא תיגע ב: שם, עמדות (מי יכול לעשות מה), והוצאה זמנית."
                )
        else:
            r1, r2 = st.columns(2)
            with r1:
                if st.button("✅ כן, אפס עכשיו", use_container_width=True, key="reset_yes"):
                    reset_person_inputs(cfg)
                    save_cfg(cfg, "input.json")
                    st.session_state.reset_pending = False
                    st.success("בוצע איפוס ✅")
            with r2:
                if st.button("❌ ביטול", use_container_width=True, key="reset_no"):
                    st.session_state.reset_pending = False
                    st.info("האיפוס בוטל.")

# -----------------------------
# Locks tab
# -----------------------------
with tab_locks:
    st.subheader("נעילות (Pin) — לקבוע מראש מי עושה מה ומתי")
    cfg.setdefault("locked_assignments", [])

    people_names = [p["name"] for p in cfg.get("people", []) if not p.get("excluded", False)]
    if not people_names:
        st.info("אין מתמחים פעילים. הוסף/י מתמחים קודם.")
    else:
        if st.button("➕ הוסף נעילה", use_container_width=True, key="add_lock"):
            cfg["locked_assignments"].append({"date": 1, "role": "first", "name": people_names[0]})

        to_del = None
        for i, lk in enumerate(cfg["locked_assignments"]):
            with st.expander(f"נעילה #{i + 1}", expanded=False):
                lk["date"] = int(st.number_input("תאריך בחודש", min_value=1, max_value=31, value=int(lk.get("date", 1)), key=f"lkd_{i}"))
                lk["role"] = st.selectbox(
                    "עמדה",
                    ROLE_KEYS,
                    index=ROLE_KEYS.index(lk.get("role", "first")) if lk.get("role", "first") in ROLE_KEYS else 0,
                    format_func=lambda r0: ROLE_NAMES_HE[r0],
                    key=f"lkr_{i}"
                )
                default_name = lk.get("name", people_names[0])
                if default_name not in people_names:
                    default_name = people_names[0]
                lk["name"] = st.selectbox("מתמחה", people_names, index=people_names.index(default_name), key=f"lkn_{i}")

                if st.button("🗑️ מחיקת נעילה", key=f"lkdel_{i}", use_container_width=True):
                    to_del = i

        if to_del is not None:
            cfg["locked_assignments"].pop(to_del)

        if st.button("💾 שמור נעילות", use_container_width=True, key="save_locks"):
            save_cfg(cfg, "input.json")
            st.success("נעילות נשמרו ב-input.json ✅")

# -----------------------------
# Generate tab
# -----------------------------
with tab_generate:
    st.subheader("יצירת לוח ותוצאות")

    holes = precheck_coverage(cfg)
    if holes:
        st.warning("יש חורים בכיסוי (יגרום ל-INFEASIBLE):")
        for dnum, wd, role in holes[:30]:
            st.write(f"- {dnum:02d}/{cfg['month']:02d}/{cfg['year']} ({WEEKDAYS_HE.get(wd, wd)}) — {ROLE_NAMES_HE.get(role, role)}")
        if len(holes) > 30:
            st.write(f"... ועוד {len(holes) - 30} חורים")
        st.stop()

    # Generate normal schedule
    if st.button("⚙️ צור/י לוח עבודה", use_container_width=True, key="gen_normal"):
        save_cfg(cfg, "input.json")

        rows, status = solve(cfg)
        if status != "OK":
            st.session_state.generated = False
            st.error("לא נמצא שיבוץ חוקי (INFEASIBLE). נסה/י להקל חוקים / לעדכן זמינות / להסיר נעילות בעייתיות.")
        else:
            schedule_he, summary_he, excel_bytes, csv_bytes = export_hebrew_files(
                rows, cfg, excel_path="output.xlsx", csv_path="output.csv"
            )
            st.session_state.generated = True
            st.session_state.last_rows = rows
            st.session_state.excel_bytes = excel_bytes
            st.session_state.csv_bytes = csv_bytes
            st.session_state.schedule_he = schedule_he
            st.session_state.summary_he = summary_he
            st.session_state.diff_df = None
            st.success("הלוח נוצר בהצלחה ✅ (הקבצים זמינים להורדה)")

    st.divider()

    # Sick leave section (UI only; solver.py must support it)
    st.subheader("עדכון לוח במקרה מחלה (מבלי לשבור את כל החודש)")

    active_people = [p["name"] for p in cfg.get("people", []) if not p.get("excluded", False)]
    if not active_people:
        st.info("אין מתמחים פעילים.")
    else:
        sick_name = st.selectbox("בחר/י מתמחה שיצא למחלה", active_people, key="sick_name")
        cA, cB = st.columns(2)
        with cA:
            from_date = st.number_input("מתאריך", min_value=1, max_value=31, value=15, step=1, key="sick_from")
        with cB:
            strict_mode = st.checkbox("מצב קשיח (לשמור ככל האפשר על שיבוץ קיים בעתיד)", value=True, key="sick_strict")

        if st.button("🔁 צור/י שיבוץ מעודכן (מחלה)", use_container_width=True, key="gen_sick"):
            if not st.session_state.last_rows:
                st.error("אין לוח קודם. קודם צור/י לוח עבודה רגיל.")
            else:
                prev_rows = st.session_state.last_rows

                cfg_try = copy.deepcopy(cfg)
                cfg_try["existing_schedule"] = prev_rows
                cfg_try["reschedule"] = {
                    "enabled": True,
                    "from_date": int(from_date),
                    "sick_name": sick_name,
                    "hard_lock_non_sick_future": bool(strict_mode),
                }

                apply_sick_leave_to_cfg(cfg_try, sick_name, int(from_date))

                rows_new, status_new = solve(cfg_try)

                if status_new != "OK" and strict_mode:
                    cfg_try2 = copy.deepcopy(cfg_try)
                    cfg_try2["reschedule"]["hard_lock_non_sick_future"] = False
                    rows_new, status_new = solve(cfg_try2)
                    cfg_try = cfg_try2

                if status_new != "OK":
                    st.error("לא ניתן ליצור שיבוץ מעודכן (INFEASIBLE). נסה/י להקל חוקים או לבחור תאריך מחלה אחר.")
                else:
                    st.session_state.diff_df = schedule_diff_df(prev_rows, rows_new)

                    schedule_he, summary_he, excel_bytes, csv_bytes = export_hebrew_files(
                        rows_new, cfg_try, excel_path="output.xlsx", csv_path="output.csv"
                    )

                    st.session_state.generated = True
                    st.session_state.last_rows = rows_new
                    st.session_state.excel_bytes = excel_bytes
                    st.session_state.csv_bytes = csv_bytes
                    st.session_state.schedule_he = schedule_he
                    st.session_state.summary_he = summary_he
                    st.success("נוצר שיבוץ מעודכן למחלה ✅ (הקבצים עודכנו להורדה)")

    # Outputs
    if st.session_state.generated:
        st.subheader("הורדת קבצים")
        st.download_button(
            "⬇️ הורד Excel",
            data=st.session_state.excel_bytes,
            file_name="schedule_hebrew.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
        st.download_button(
            "⬇️ הורד CSV",
            data=st.session_state.csv_bytes,
            file_name="schedule_hebrew.csv",
            mime="text/csv",
            use_container_width=True
        )

        tab_s, tab_sum, tab_val, tab_diff = st.tabs(["לוח עבודה", "סיכום", "בדיקות", "שינויים"])

        with tab_s:
            st.dataframe(st.session_state.schedule_he, use_container_width=True)

        with tab_sum:
            st.dataframe(st.session_state.summary_he, use_container_width=True)

        with tab_val:
            issues = validate_schedule(cfg, st.session_state.last_rows)
            if not issues:
                st.success("כל הבדיקות עברו ✅ אין הפרות חוקים.")
            else:
                st.error(f"נמצאו {len(issues)} בעיות:")
                st.dataframe(pd.DataFrame(issues), use_container_width=True)

        with tab_diff:
            df = st.session_state.diff_df
            if df is None or df.empty:
                st.info("אין שינויים להציג (או שלא הופעל עדכון מחלה).")
            else:
                st.success(f"נמצאו {len(df)} שינויים:")
                st.dataframe(df, use_container_width=True)

# Persist any changes in-memory (optional)
# NOTE: We intentionally do NOT auto-save to input.json to avoid overwriting by mistake.
