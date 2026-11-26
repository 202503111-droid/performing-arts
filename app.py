"""공연 음악 장르별 예매 추이 대시보드

질문: 코로나 이후 공연 음악 장르(대중음악·클래식·국악)는 얼마나, 어떤 속도로 회복했는가?
자료: 예술경영지원센터 공연예술통합전산망(KOPIS) 예매통계 장르별통계 (월 단위로 조회해 합침)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="공연 음악 장르별 예매 추이", layout="wide")

# ---------------------------------------------------------------- 설정
DATA_PATH = Path(__file__).parent / "data" / "kopis_genre_monthly.csv"
DATA_DATE = "2026-11-26"       # 자료를 내려받은 날(KOPIS 최종집계 시점)
LAST_MONTH = "2026-10"         # 마지막으로 끝난 달 (2026-11은 11/25까지의 부분 자료라 제외)
BASE_YEAR = 2019               # 지수의 기준 연도
RECOVER_QUARTERS = 2           # 회복 판정: 저점 이후 이 개수만큼 연속으로 2019년 같은 분기 수준(100) 이상

MUSIC = ["대중음악", "서양음악(클래식)", "한국음악(국악)"]
COMPARE = ["뮤지컬", "연극", "합계"]
LABEL = {"서양음악(클래식)": "클래식", "한국음악(국악)": "국악", "합계": "전체 공연"}
# 색: 검증된 범주형 팔레트의 앞 5색을 장르에 고정해서 배정(선택이 바뀌어도 같은 장르는 같은 색). 합계는 무채색.
COLORS = {"대중음악": "#2a78d6", "서양음악(클래식)": "#eb6834", "한국음악(국악)": "#1baf7a",
          "뮤지컬": "#eda100", "연극": "#e87ba4", "합계": "#9aa0a8"}
METRICS = {"순예매액 (억 원)": ("순예매액", "억 원"), "순예매수 (만 장)": ("순예매수", "만 장")}
NAVY = "#14284b"
INK = "#3a4150"
GRID = "#e8ebf0"
FONT = "Pretendard, 'Noto Sans KR', 'Apple SD Gothic Neo', 'Malgun Gothic', sans-serif"
# 참고 구간(국내 첫 확진 ~ 사회적 거리두기 전면 해제). 설명용 음영이며 분석 기준이 아니다.
COVID_SPAN = ("2020-01-20", "2022-04-18")


def name(g: str) -> str:
    return LABEL.get(g, g)


def md(text: str) -> str:
    """마크다운 문장용: ~ 사이의 글자에 취소선이 생기지 않도록 ~를 이스케이프."""
    return text.replace("~", "\\~")


def josa(word: str, with_batchim: str, without: str) -> str:
    """받침 유무에 따라 조사를 고른다(예: 은/는, 이/가, 으로/로)."""
    ch = word[-1]
    if "가" <= ch <= "힣":
        return with_batchim if (ord(ch) - 0xAC00) % 28 else without
    return with_batchim


def qlabel(ts: pd.Timestamp) -> str:
    return f"{ts.year}Q{ts.quarter}"


# ---------------------------------------------------------------- 데이터/계산
@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df = df[df["월"] <= LAST_MONTH].copy()
    df["날짜"] = pd.to_datetime(df["월"] + "-01")
    df["순예매액"] = df["총예매액_천원"] / 1e5     # 천원 → 억 원
    df["순예매수"] = df["총예매수"] / 1e4          # 건(장) → 만 장
    return df


def monthly_wide(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    return df.pivot(index="날짜", columns="장르", values=metric).sort_index()


def to_quarterly(w: pd.DataFrame) -> pd.DataFrame:
    """3개월이 모두 있는 분기만 합산. 인덱스는 분기 첫 달."""
    per = w.index.to_period("Q")
    full = w.groupby(per).size()
    full = full[full == 3].index
    q = w.groupby(per).sum().loc[full]
    q.index = q.index.to_timestamp()
    return q


def index_vs_base(s: pd.DataFrame, unit: str) -> pd.DataFrame:
    """같은 월(분기)의 BASE_YEAR 값을 100으로 한 지수. 기준값이 0 이하면 계산하지 않는다."""
    key = (lambda ix: ix.month) if unit == "월별" else (lambda ix: ix.quarter)
    base_rows = s[s.index.year == BASE_YEAR]
    base = base_rows.groupby(key(base_rows.index)).first()
    b = pd.DataFrame(base.reindex(key(s.index)).to_numpy(), index=s.index, columns=s.columns)
    return s / b.where(b > 0) * 100


def ttm_index(w: pd.DataFrame, end_ts: pd.Timestamp) -> pd.Series:
    """종료 월까지 최근 12개월 합계가 BASE_YEAR 12개월 합계의 몇 %인가."""
    base = w[w.index.year == BASE_YEAR]
    if len(base) < 12 or end_ts <= pd.Timestamp(f"{BASE_YEAR}-12-01"):   # 기준 연도 안에서는 비교가 의미 없음
        return pd.Series(np.nan, index=w.columns)
    cur = w[(w.index <= end_ts) & (w.index > end_ts - pd.DateOffset(months=12))]
    if len(cur) < 12:
        return pd.Series(np.nan, index=w.columns)
    b = base.sum()
    return cur.sum() / b.where(b > 0) * 100


def in_range(s: pd.DataFrame, unit: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    if unit == "월별":
        return s[(s.index >= start_ts) & (s.index <= end_ts)]
    return s[(s.index >= start_ts) & (s.index + pd.DateOffset(months=2) <= end_ts)]


def recovery_info(qidx: pd.DataFrame, genre: str):
    """2020년 이후 분기 지수에서 저점과 회복 분기를 찾는다. 하락이 없으면 dip=False."""
    ser = qidx[genre].dropna()
    ser = ser[ser.index.year > BASE_YEAR]
    if ser.empty:
        return None
    low_ts, low = ser.idxmin(), float(ser.min())
    out = {"low_ts": low_ts, "low": low, "dip": low < 100, "rec_ts": None, "confirmed": True}
    if low >= 100:
        return out
    after = ser[ser.index > low_ts].to_numpy()
    idx = ser[ser.index > low_ts].index
    for i in range(len(after)):
        if after[i] >= 100:
            window = after[i:i + RECOVER_QUARTERS]
            if len(window) < RECOVER_QUARTERS:
                out["confirmed"] = False       # 마지막 분기에서야 100을 넘어 연속 여부를 확인할 수 없음
                break
            if (window >= 100).all():
                out["rec_ts"] = idx[i]
                break
    return out


# ---------------------------------------------------------------- 차트
def style(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        template="plotly_white", font=dict(family=FONT, size=13, color=INK),
        paper_bgcolor="white", plot_bgcolor="white",
        hoverlabel=dict(font_family=FONT, bgcolor="white", bordercolor=GRID),
        margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.15),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    return fig


def add_covid_band(fig: go.Figure, x_min: pd.Timestamp, x_max: pd.Timestamp):
    a, b = pd.Timestamp(COVID_SPAN[0]), pd.Timestamp(COVID_SPAN[1])
    if b < x_min or a > x_max:
        return
    fig.add_shape(type="rect", xref="x", yref="paper", x0=max(a, x_min), x1=min(b, x_max), y0=0, y1=1,
                  fillcolor="#eef1f6", opacity=0.9, line_width=0, layer="below")
    fig.add_annotation(x=max(a, x_min), xref="x", y=1.0, yref="paper", yanchor="bottom", xanchor="left",
                       text="참고: 국내 첫 확진(2020.01)~거리두기 전면 해제(2022.04)", showarrow=False,
                       font=dict(size=11, color="#6b7380"))


def make_line_figure(s: pd.DataFrame, genres: list, unit: str, ytitle: str, ref100: bool) -> go.Figure:
    fig = go.Figure()
    for g in genres:
        if g not in s.columns:
            continue
        labels = [qlabel(t) if unit == "분기별" else t.strftime("%Y-%m") for t in s.index]
        fig.add_trace(go.Scatter(
            x=s.index, y=s[g], mode="lines", name=name(g), customdata=labels,
            line=dict(color=COLORS[g], width=2.5, dash="dot" if g == "합계" else "solid"),
            hovertemplate="<b>" + name(g) + "</b><br>%{customdata}<br>%{y:,.1f} " + ytitle + "<extra></extra>"))
    if ref100:
        fig.add_hline(y=100, line_dash="dash", line_color="#6b7380", line_width=1)
    fig.update_yaxes(title=ytitle)
    fig.update_xaxes(tickformat="%Y", dtick="M12", hoverformat="%Y-%m")
    if len(s):
        add_covid_band(fig, s.index.min(), s.index.max())
    return style(fig)


def make_share_figure(share: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    xs = [qlabel(t) for t in share.index]
    for g in MUSIC:
        fig.add_trace(go.Bar(
            x=xs, y=share[g], name=name(g), marker=dict(color=COLORS[g], line=dict(color="white", width=2)),
            hovertemplate="<b>" + name(g) + "</b><br>%{x}<br>%{y:.1f}%<extra></extra>"))
    fig.update_layout(barmode="stack")
    fig.update_yaxes(title="음악 3장르 중 비중 (%)", range=[0, 100])
    fig.update_xaxes(type="category", tickangle=-45)
    return style(fig)


# ---------------------------------------------------------------- 문장
def fmt_pct(x: float) -> str:
    return "-" if pd.isna(x) else f"{x:,.0f}%"


def result_sentences(ttm: pd.Series, rec: dict, share: float, metric_name: str, end: str) -> list:
    out = []
    parts = [f"{name(g)} {fmt_pct(ttm[g])}" for g in MUSIC if not pd.isna(ttm.get(g, np.nan))]
    if parts:
        out.append(f"{end} 기준 최근 12개월 {metric_name}{josa(metric_name, '은', '는')} 2019년 연간 대비 " + ", ".join(parts) + "입니다.")
    dips = {g: r for g, r in rec.items() if r and r["dip"]}
    if dips:
        deepest = min(dips, key=lambda g: dips[g]["low"])
        d = name(deepest)
        out.append(f"코로나 이후 저점이 가장 깊었던 장르는 {d}{josa(d, '으로', '로')}, {qlabel(dips[deepest]['low_ts'])}에 "
                   f"2019년 같은 분기의 {dips[deepest]['low']:.0f}% 수준이었습니다.")
        done = {g: r for g, r in dips.items() if r["rec_ts"] is not None}
        if done:
            first = min(done, key=lambda g: done[g]["rec_ts"])
            late = [f"{name(g)}({qlabel(r['rec_ts'])})" for g, r in done.items() if g != first]
            f_ = name(first)
            text = (f"2019년 같은 분기 수준을 {RECOVER_QUARTERS}개 분기 연속 넘긴 시점은 {f_}({qlabel(done[first]['rec_ts'])})"
                    f"{josa(f_, '이', '가')} 가장 빨랐습니다.")
            if late:
                text += " 이어서 " + ", ".join(late) + "입니다."
            out.append(text)
        undone = [name(g) for g, r in dips.items() if r["rec_ts"] is None]
        if undone:
            u = ", ".join(undone)
            out.append(f"{u}{josa(undone[-1], '은', '는')} 선택한 기간 안에서 2019년 같은 분기 수준을 연속으로 회복한 분기가 확인되지 않았습니다.")
    if not pd.isna(share):
        out.append(f"음악 3장르 중 대중음악의 비중은 최근 12개월 {share:.0f}%입니다.")
    return out


def limit_lines(neg_months: int, share: float, unit: str) -> list:
    lines = [
        "**순예매액은 예매에서 취소를 뺀 값이고 예매일자 기준입니다.** 티켓을 산 날짜로 집계되어, 취소가 몰린 달에는 값이 "
        "작아지거나 음수가 됩니다. 코로나 시기의 하락에는 수요 감소뿐 아니라 대규모 취소가 섞여 있습니다."
        + (f" 선택한 기간에는 장르별 월 값이 음수인 경우가 {neg_months}건 있습니다." if neg_months else ""),
        "**2019년은 기준이 낮을 수 있습니다.** 공연법에 따른 자료 수집이 본격화된 해라서, 2019년 대비 증가에는 집계 범위 확대가 "
        "섞여 있을 수 있습니다. 이 자료만으로는 그 크기를 알 수 없습니다.",
        "**규모가 한쪽으로 쏠려 있습니다.**" + (f" 최근 12개월 음악 3장르 중 대중음악이 {share:.0f}%입니다." if not pd.isna(share) else "")
        + " 규모가 작은 국악이나 클래식은 몇 건의 공연에도 지수가 크게 흔들리므로, 장르 사이의 지수를 같은 무게로 비교하면 안 됩니다.",
        "**전산 발권된 공연만 집계됩니다.** 무료 공연이나 현장 판매가 많은 장르는 실제보다 작게 보일 수 있습니다.",
        "**금액은 물가를 반영하지 않은 명목값이고, 공연 티켓 시장의 자료입니다.** 음원·스트리밍 소비와는 다릅니다.",
    ]
    if unit == "월별":
        lines.append("월별 지수는 계절성과 예매 시점의 쏠림(대형 공연의 티켓 오픈 등)으로 크게 흔들립니다. 추이를 읽을 때는 분기별을 권장합니다.")
    return lines


# ================================================================ 화면
df = load_data()
st.markdown(f"""
<style>
.block-container {{ padding-top: 1.4rem; max-width: 1280px; }}
header[data-testid="stHeader"] {{ background: transparent; }}
.banner {{ background: linear-gradient(135deg, {NAVY} 0%, #1f3d73 100%); border-radius: 10px;
          padding: 1.5rem 1.8rem 1.4rem; margin-bottom: 1rem; }}
.banner .eyebrow {{ color: #9fb8e0; font-size: .76rem; font-weight: 700; letter-spacing: .1em; margin-bottom: .35rem; }}
.banner .title {{ color: #ffffff; font-size: 2rem; font-weight: 700; letter-spacing: -0.02em; line-height: 1.25; }}
.banner .lead {{ color: #d6e2f5; font-size: 1.02rem; margin-top: .45rem; }}
h2, h3 {{ color: {NAVY}; letter-spacing: -0.01em; font-weight: 700; }}
[data-testid="stVerticalBlockBorderWrapper"] {{ background: #ffffff; border-color: #dfe4ec; border-radius: 8px; }}
[data-testid="stColumn"] [data-testid="stVerticalBlockBorderWrapper"] {{
    border-top: 3px solid {NAVY}; box-shadow: 0 1px 3px rgba(20,40,75,.08); }}
[data-testid="stMetricLabel"] p {{ color: #52606d; font-weight: 600; font-size: .86rem; }}
[data-testid="stMetricValue"] {{ color: {NAVY}; font-weight: 700; font-size: 1.9rem; }}
button[role="tab"] p {{ font-weight: 600; font-size: .98rem; }}
button[role="tab"][aria-selected="true"] {{ color: {NAVY}; }}
div[data-baseweb="tab-highlight"] {{ background-color: {NAVY}; height: 3px; }}
[data-testid="stSidebar"] {{ border-right: 1px solid #dfe4ec; }}
[data-testid="stSidebar"] h2 {{ font-size: 1.05rem; color: {NAVY}; border-bottom: 2px solid {NAVY}; padding-bottom: .4rem; }}
[data-testid="stExpander"] {{ border-color: #dfe4ec; border-radius: 8px; background: #fbfcfe; }}
</style>
<div class="banner">
<div class="eyebrow">예술경영지원센터 공연예술통합전산망(KOPIS) · 예매통계</div>
<div class="title">공연 음악 장르별 예매 추이 대시보드</div>
<div class="lead">코로나 이후 공연 음악 장르(대중음악·클래식·국악)는 얼마나, 어떤 속도로 회복했는가?</div>
</div>
""", unsafe_allow_html=True)
st.caption(md("왼쪽에서 조건을 바꾸면 모든 숫자와 차트가 함께 바뀝니다. 금액은 예매에서 취소를 뺀 순예매액이며 "
              f"예매일자 기준입니다. 자료 조회일 {DATA_DATE}."))

months = [m for m in sorted(df["월"].unique())]
with st.sidebar:
    st.header("조건 선택")
    unit = st.radio("집계 단위", ["분기별", "월별"], index=0,
                    help="분기별은 3개월 합계(완결된 분기만)입니다. 월별은 취소와 예매 시점의 영향으로 크게 흔들립니다.")
    metric_label = st.selectbox("지표", list(METRICS), index=0)
    genres = st.multiselect("장르 (추이·2019년 대비 차트에 표시)", MUSIC + COMPARE, default=MUSIC,
                            format_func=name,
                            help="핵심 지표와 결론은 항상 음악 3장르(대중음악·클래식·국악) 기준입니다. 뮤지컬·연극·전체 공연은 비교용입니다.")
    start = st.selectbox("시작 월", months, index=0)
    end = st.selectbox("종료 월", months, index=len(months) - 1)

if start >= end:
    st.error("시작 월이 종료 월보다 앞서야 합니다. 기간을 다시 선택해 주세요.")
    st.stop()

metric, unit_name = METRICS[metric_label]
start_ts, end_ts = pd.Timestamp(start + "-01"), pd.Timestamp(end + "-01")

w = monthly_wide(df, metric)
q = to_quarterly(w)
s_full = q if unit == "분기별" else w
idx_full = index_vs_base(s_full, unit)
qidx = index_vs_base(q, "분기별")

s_view = in_range(s_full, unit, start_ts, end_ts)
idx_view = in_range(idx_full, unit, start_ts, end_ts)
qidx_view = in_range(qidx, "분기별", start_ts, end_ts)

ttm = ttm_index(w, end_ts)
rec = {g: recovery_info(qidx_view, g) for g in MUSIC}
music_sum_ttm = w[MUSIC][(w.index <= end_ts) & (w.index > end_ts - pd.DateOffset(months=12))]
share_ttm = (float(music_sum_ttm["대중음악"].sum() / music_sum_ttm.sum().sum() * 100)
             if len(music_sum_ttm) == 12 and music_sum_ttm.sum().sum() > 0 else np.nan)

# ---------------------------------------------------------------- 핵심 지표
def kpi(col, title: str, value: str, sub: str, help_text: str):
    with col:
        with st.container(border=True):
            st.metric(title, value, help=help_text)
            st.caption(md(sub))


st.caption(md(f"{unit} · {metric_label} · {start} ~ {end}"))
k1, k2, k3, k4 = st.columns(4)
ttm_m = ttm[MUSIC].dropna()
if ttm_m.empty:
    kpi(k1, "2019년 대비 1위", "-", "종료 월을 2019-12 이후로 선택하세요", "최근 12개월 합계가 2019년 연간 합계의 몇 %인지로 비교합니다.")
else:
    top = ttm_m.idxmax()
    kpi(k1, "2019년 대비 1위", name(top), f"최근 12개월이 2019년의 {ttm_m[top]:,.0f}%",
        "음악 3장르 중, 종료 월까지 최근 12개월 합계가 2019년 연간 합계의 몇 %인지 가장 높은 장르입니다.")
dips = {g: r for g, r in rec.items() if r and r["dip"]}
if dips:
    deepest = min(dips, key=lambda g: dips[g]["low"])
    kpi(k2, "저점이 가장 깊은 장르", name(deepest),
        f"{qlabel(dips[deepest]['low_ts'])} · 2019년 같은 분기의 {dips[deepest]['low']:.0f}%",
        "선택 기간 안에서 분기 지수(2019년 같은 분기 = 100)가 가장 낮았던 분기와 그 값입니다.")
    done = {g: r for g, r in dips.items() if r["rec_ts"] is not None}
    if done:
        first = min(done, key=lambda g: done[g]["rec_ts"])
        kpi(k3, "가장 먼저 회복한 장르", name(first), f"{qlabel(done[first]['rec_ts'])} (저점 이후 {RECOVER_QUARTERS}개 분기 연속 100% 이상)",
            f"저점 이후 {RECOVER_QUARTERS}개 분기 연속으로 2019년 같은 분기 수준 이상이 된 첫 분기가 가장 빠른 장르입니다.")
    else:
        kpi(k3, "가장 먼저 회복한 장르", "없음", "선택 기간에서 회복이 확인되지 않음",
            f"저점 이후 {RECOVER_QUARTERS}개 분기 연속으로 2019년 같은 분기 수준 이상이 된 장르가 없습니다.")
else:
    kpi(k2, "저점이 가장 깊은 장르", "-", "선택 기간에 2019년 수준 아래로 떨어진 분기가 없음", "코로나 시기를 포함하도록 기간을 넓혀 보세요.")
    kpi(k3, "가장 먼저 회복한 장르", "-", "선택 기간에 하락 구간이 없음", "코로나 시기를 포함하도록 기간을 넓혀 보세요.")
kpi(k4, "대중음악 비중 (음악 3장르)", "-" if pd.isna(share_ttm) else f"{share_ttm:.0f}%", "최근 12개월 순예매액 기준",
    "대중음악 + 클래식 + 국악 합계 중 대중음악의 몫입니다. 규모가 한쪽으로 쏠려 있어 장르 비교에 주의가 필요합니다.")

with st.expander("용어 풀이: 순예매액 · 2019년 대비 지수 · 최근 12개월 · 저점 · 회복"):
    st.markdown(md(
        "- **순예매액(순예매수):** 예매에서 취소를 뺀 금액(장수)입니다. 티켓을 산 날짜로 집계하며, 공연이 열린 날짜가 아닙니다.\n"
        f"- **2019년 대비 지수:** 같은 분기(월)의 {BASE_YEAR}년 값을 100으로 한 값입니다. 200이면 {BASE_YEAR}년 같은 분기의 두 배입니다. 계절성이 같은 시기끼리 비교해 제거됩니다.\n"
        f"- **최근 12개월 대비:** 종료 월까지 최근 12개월 합계가 {BASE_YEAR}년 연간 합계의 몇 %인지입니다.\n"
        f"- **저점:** 선택 기간 안에서 분기 지수가 가장 낮았던 분기입니다.\n"
        f"- **회복:** 저점 이후 {RECOVER_QUARTERS}개 분기 연속으로 지수가 100 이상이 된 첫 분기입니다. 한 분기만 넘은 것은 회복으로 보지 않습니다.\n"
        "- **비중:** 대중음악·클래식·국악 합계 중 각 장르의 몫입니다."))

tab_idx, tab_trend, tab_share, tab_data = st.tabs(["2019년 대비", "추이", "장르 비중", "자료"])

with tab_idx:
    st.markdown(md(f"**2019년 대비 지수**는 같은 {'분기' if unit == '분기별' else '월'}의 {BASE_YEAR}년 값을 100으로 놓은 값입니다. "
                   "100보다 높으면 2019년 같은 시기보다 컸다는 뜻이고, 점선(100)이 2019년 수준입니다."))
    if not genres:
        st.info("왼쪽에서 장르를 하나 이상 선택해 주세요.")
    elif idx_view.dropna(how="all").empty:
        st.warning("선택한 기간에 계산할 지수가 없습니다. 기간을 넓혀 주세요.")
    else:
        st.plotly_chart(make_line_figure(idx_view, genres, unit, "2019년 대비 지수", ref100=True))
        rows = []
        for g in genres:
            r = recovery_info(qidx_view, g) if g in qidx_view.columns else None
            rows.append({"장르": name(g),
                         "최근 12개월 (2019년=100)": None if pd.isna(ttm.get(g, np.nan)) else round(float(ttm[g]), 1),
                         "저점 분기": "-" if not r else qlabel(r["low_ts"]),
                         "저점 지수": None if not r else round(r["low"], 1),
                         "회복 분기": "-" if (not r or not r["dip"]) else (qlabel(r["rec_ts"]) if r["rec_ts"] is not None else
                                                                           ("확인 불가" if not r["confirmed"] else "미회복"))})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.caption(md("규모가 작은 장르는 지수가 크게 흔들립니다. 월별 화면에서는 취소가 몰린 달의 값이 음수일 수 있어 지수도 음수로 나옵니다. "
                      "회복 분기는 분기 기준으로 계산합니다."))

with tab_trend:
    st.markdown(md(f"장르별 **{metric_label}** 추이입니다. 값의 크기는 장르마다 매우 달라서, 작은 장르의 흐름은 "
                   "‘2019년 대비’ 탭에서 함께 보세요."))
    if not genres:
        st.info("왼쪽에서 장르를 하나 이상 선택해 주세요.")
    elif s_view.empty:
        st.warning("선택한 기간에 해당하는 자료가 없습니다. 기간을 넓혀 주세요.")
    else:
        st.plotly_chart(make_line_figure(s_view, genres, unit, unit_name, ref100=False))
        st.caption(md("음영은 설명용 참고 구간(국내 첫 확진~거리두기 전면 해제)이며 분석 기준이 아닙니다."))

with tab_share:
    st.markdown(md("**음악 3장르 중 각 장르의 비중**입니다(분기별, 순예매액 기준). 규모가 한쪽으로 쏠려 있는지 확인하는 용도입니다."))
    qa = to_quarterly(monthly_wide(df, "순예매액"))[MUSIC]
    qa = in_range(qa, "분기별", start_ts, end_ts)
    qa = qa[(qa > 0).all(axis=1)]            # 취소가 몰려 음수가 된 분기는 비중을 계산하지 않음
    if qa.empty:
        st.warning("선택한 기간에 비중을 계산할 분기가 없습니다. 기간을 넓혀 주세요.")
    else:
        share = qa.div(qa.sum(axis=1), axis=0) * 100
        st.plotly_chart(make_share_figure(share))
        st.caption(md("순예매액이 음수인 분기는 제외했습니다. 클래식과 국악이 차지하는 비중이 매우 작아 막대에서 잘 보이지 않을 수 있습니다."))

with tab_data:
    st.markdown(md(f"선택한 조건의 값입니다({unit}, {metric_label}). 내려받아 직접 확인할 수 있습니다."))
    if s_view.empty or not genres:
        st.info("표시할 자료가 없습니다. 장르와 기간을 확인해 주세요.")
    else:
        tbl = s_view[[g for g in genres if g in s_view.columns]].copy()
        tbl.index = [qlabel(t) if unit == "분기별" else t.strftime("%Y-%m") for t in tbl.index]
        tbl.columns = [name(c) for c in tbl.columns]
        tbl.index.name = "기간"
        st.dataframe(tbl.round(1))
        st.download_button("표를 CSV로 내려받기", tbl.round(2).to_csv().encode("utf-8-sig"),
                           file_name="kopis_genre_selected.csv", mime="text/csv")

# ---------------------------------------------------------------- 결론 (탭 아래)
neg_view = int((in_range(monthly_wide(df, "순예매액"), "월별", start_ts, end_ts)[genres] < 0).sum().sum()) if genres else 0
sents = result_sentences(ttm, rec, share_ttm, "순예매액" if metric == "순예매액" else "순예매수", end)
with st.container(border=True):
    st.subheader("결론")
    for t in sents:
        st.markdown("- " + md(t))
    if not sents:
        st.markdown("- 선택한 조건에서는 문장으로 요약할 결과가 없습니다. 종료 월과 기간을 확인해 주세요.")
with st.expander("이 결론의 한계", expanded=True):
    for t in limit_lines(neg_view, share_ttm, unit):
        st.markdown("- " + md(t))

with st.expander("출처와 집계 기준"):
    st.markdown(md(
        "- **출처:** 예술경영지원센터, 공연예술통합전산망(KOPIS) 예매통계 ‘장르별통계’. 월 단위로 조회해 합쳤고, "
        f"조회일은 {DATA_DATE}(최종집계 2026.11.26)입니다. 통계는 갱신되므로 조회일에 따라 값이 달라질 수 있습니다.\n"
        f"- **기간:** 2019.01~{LAST_MONTH.replace('-', '.')} (2026.11은 11/25까지의 부분 자료라 제외). "
        "KOPIS의 본격적인 자료 수집은 2019년 공연법 시행 이후입니다.\n"
        "- **지표:** 순예매액 = 총 티켓예매액(예매 − 취소, 예매일자 기준, 천원 단위를 억 원으로 환산). "
        "순예매수 = 총 티켓예매수(예매수 − 취소수, 만 장).\n"
        f"- **집계:** 분기 = 3개월 합(완결된 분기만). 2019년 대비 지수 = 같은 월(분기)의 {BASE_YEAR}년 값을 100으로 한 값. "
        f"회복 = 저점 이후 {RECOVER_QUARTERS}개 분기 연속 100 이상. 공연건수는 월별 합산 시 중복이 생겨 사용하지 않았습니다.\n"
        "- **장르 분류:** KOPIS는 2023년 1월 장르를 개편했습니다. 이 자료에는 2019년부터 대중음악이 집계되어 있으나, "
        "개편 전후의 정의 차이는 확인하지 못했습니다.\n"
        "- **음영 구간:** 국내 첫 확진(2020.01)과 사회적 거리두기 전면 해제(2022.04)는 설명용 참고일이며, 분석 기준이 아닙니다."))
