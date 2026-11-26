"""공연예술 장르별 성장 양상 대시보드

질문: 2019년 이후 공연예술(전체와 장르별)은 얼마나 성장했고, 그 성장의 주된 요인은 관객 수와 가격 중 무엇인가?
자료: 예술경영지원센터 공연예술통합전산망(KOPIS) 예매통계 장르별통계 (월 단위로 조회해 합침)
      국가데이터처(통계청) 소비자물가지수 총지수(2020=100) 월별 — 물가 비교·물가 반영 계산용
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="공연예술 장르별 성장 양상", layout="wide")

# ---------------------------------------------------------------- 설정
DATA_PATH = Path(__file__).parent / "data" / "kopis_genre_monthly.csv"
CPI_PATH = Path(__file__).parent / "data" / "cpi_monthly.csv"
DATA_DATE = "2026-11-26"       # 자료를 내려받은 날(KOPIS 최종집계 시점)
def window_label(end_month):
    """종료 월을 포함해 거꾸로 12개월인 기간을 '2025.11~2026.10'처럼 돌려준다."""
    e = pd.Period(end_month, freq="M")
    return f"{(e - 11).strftime('%Y.%m')}~{e.strftime('%Y.%m')}"


LAST_MONTH = "2026-10"         # 마지막으로 끝난 달 (2026-11은 11/25까지의 부분 자료라 제외)
BASE_YEAR = 2019               # 비교 기준 연도 (물가 반영 금액은 이 해의 평균 물가 기준 가격)

# 개별 표시 장르 7개 + 전체 합계. 복합·대중무용은 규모가 매우 작아(합계의 0.3% 안팎) 개별 표시에서 빼고 합계에만 포함된다.
GENRES = ["대중음악", "뮤지컬", "연극", "서양음악(클래식)", "한국음악(국악)", "무용(서양/한국무용)", "서커스/마술"]
TOTAL = ["합계"]
LABEL = {"서양음악(클래식)": "클래식", "한국음악(국악)": "국악", "무용(서양/한국무용)": "무용", "합계": "전체 공연"}
# 색: 검증된 범주형 팔레트의 앞 7색을 장르에 고정해서 배정(선택이 바뀌어도 같은 장르는 같은 색). 합계는 무채색.
COLORS = {"대중음악": "#2a78d6", "서양음악(클래식)": "#eb6834", "한국음악(국악)": "#1baf7a",
          "뮤지컬": "#eda100", "연극": "#e87ba4", "무용(서양/한국무용)": "#008300", "서커스/마술": "#6250d6", "합계": "#9aa0a8"}
CPI_COLOR = "#6b7380"
NAVY = "#14284b"
INK = "#3a4150"
GRID = "#e8ebf0"
FONT = "Pretendard, 'Noto Sans KR', 'Apple SD Gothic Neo', 'Malgun Gothic', sans-serif"
# 참고 구간(국내 첫 확진 ~ 사회적 거리두기 전면 해제). 설명용 음영이며 분석 기준이 아니다.
COVID_SPAN = ("2020-01-20", "2022-04-18")
GROWTH_VIEW = "성장 (2019년 = 100)"
LEVEL_VIEW = "실제 값 (월별)"


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
def load_data():
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df = df[df["월"] <= LAST_MONTH].copy()
    df["날짜"] = pd.to_datetime(df["월"] + "-01")
    df["순예매액"] = df["총예매액_천원"] / 1e5     # 천원 → 억 원
    df["순예매수"] = df["총예매수"] / 1e4          # 건(장) → 만 장
    # 물가 반영: 월별 소비자물가지수를 기준 연도 평균으로 나눈 값(물가)으로 금액을 나눠 '기준 연도 가격'으로 환산
    cpi = pd.read_csv(CPI_PATH, encoding="utf-8-sig")
    cpi["날짜"] = pd.to_datetime(cpi["월"] + "-01")
    base = cpi.loc[cpi["월"].str[:4] == str(BASE_YEAR), "소비자물가지수"].mean()
    cpi["물가"] = cpi["소비자물가지수"] / base
    df = df.merge(cpi[["월", "물가"]], on="월", how="left")
    df["실질예매액"] = df["순예매액"] / df["물가"]
    price = cpi[cpi["월"] <= LAST_MONTH].set_index("날짜")["물가"].sort_index()
    return df, price


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


def ratio(a: pd.DataFrame, c: pd.DataFrame) -> pd.DataFrame:
    """장당 평균 가격 = 금액 ÷ 수량. 금액이나 수량이 0 이하인 달(분기)은 계산하지 않는다."""
    return (a / c.where(c > 0)).where(a > 0)


def last12(w, end_ts: pd.Timestamp):
    return w[(w.index <= end_ts) & (w.index > end_ts - pd.DateOffset(months=12))]


def rolling_index(w: pd.DataFrame, denom: pd.DataFrame = None) -> pd.DataFrame:
    """각 달까지 '최근 12개월 합계'가 BASE_YEAR 연간 합계의 몇 %인지(2019-12 이후).
    월 하나가 아니라 12개월 합계끼리 비교하므로 계절성과 기준 달 하나의 영향이 없다.
    denom(수량)을 주면 금액 ÷ 수량, 즉 장당 평균 가격의 지수가 된다."""
    base = w[w.index.year == BASE_YEAR]
    start = pd.Timestamp(f"{BASE_YEAR}-12-01")
    if len(base) < 12:
        return pd.DataFrame(np.nan, index=w.index[w.index >= start], columns=w.columns)
    roll = w.rolling(12, min_periods=12).sum()
    if denom is None:
        b = base.sum()
        out = roll / b.where(b > 0) * 100
    else:
        bd = denom[denom.index.year == BASE_YEAR].sum()
        rd = denom.rolling(12, min_periods=12).sum()
        bp = base.sum() / bd.where(bd > 0)
        out = (roll / rd.where(rd > 0)) / bp.where(bp > 0) * 100
    return out[out.index >= start]


def rolling_price_level(price: pd.Series) -> pd.Series:
    """소비자물가의 최근 12개월 평균이 기준 연도 평균의 몇 %인가(2019-12 이후)."""
    out = price.rolling(12, min_periods=12).mean() * 100
    return out[out.index >= pd.Timestamp(f"{BASE_YEAR}-12-01")]


def decompose(wa_n: pd.DataFrame, wa_r: pd.DataFrame, wc: pd.DataFrame, end_ts: pd.Timestamp):
    """종료 월까지 최근 12개월을 BASE_YEAR 12개월과 비교한 배수(기준 연도 = 1)와 그 구성.
    명목 예매액 = 예매수 × 장당 가격(물가 반영) × 물가 요인  (곱셈으로 정확히 성립)
    물가 요인 = 명목 예매액 배수 ÷ 물가 반영 예매액 배수 (금액 가중 물가 상승분)."""
    base_n = wa_n[wa_n.index.year == BASE_YEAR]
    if len(base_n) < 12 or end_ts <= pd.Timestamp(f"{BASE_YEAR}-12-01"):
        return None
    cur_n, cur_r, cur_c = last12(wa_n, end_ts), last12(wa_r, end_ts), last12(wc, end_ts)
    if len(cur_n) < 12:
        return None
    b_n, b_r = base_n.sum(), wa_r[wa_r.index.year == BASE_YEAR].sum()
    b_c = wc[wc.index.year == BASE_YEAR].sum()
    c_n, c_r, c_c = cur_n.sum(), cur_r.sum(), cur_c.sum()
    ok = ((b_n > 0) & (b_r > 0) & (b_c > 0) & (c_n > 0) & (c_r > 0) & (c_c > 0)).reindex(wa_n.columns)
    A, R, Q = c_n / b_n, c_r / b_r, c_c / b_c
    out = pd.DataFrame({"명목 예매액": A, "예매액(물가 반영)": R, "예매수": Q,
                        "장당 가격(명목)": A / Q, "장당 가격(물가 반영)": R / Q, "물가 요인": A / R})
    out.loc[~ok.to_numpy()] = np.nan
    return out


def contributions(row: pd.Series) -> dict:
    """명목 예매액 성장(로그)을 세 요인으로 나눈 몫(%). 성장하지 않았거나 계산할 수 없으면 빈 dict."""
    vals = row[["예매수", "장당 가격(물가 반영)", "물가 요인", "명목 예매액"]]
    if vals.isna().any() or (vals <= 0).any() or row["명목 예매액"] <= 1:
        return {}
    la = float(np.log(row["명목 예매액"]))
    return {"관객 수": float(np.log(row["예매수"])) / la * 100,
            "가격(물가 반영)": float(np.log(row["장당 가격(물가 반영)"])) / la * 100,
            "물가 상승": float(np.log(row["물가 요인"])) / la * 100}


def in_range(s: pd.DataFrame, unit: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    if unit == "월별":
        return s[(s.index >= start_ts) & (s.index <= end_ts)]
    return s[(s.index >= start_ts) & (s.index + pd.DateOffset(months=2) <= end_ts)]


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


def make_line_figure(s: pd.DataFrame, genres: list, ytitle: str, ref100: bool, quarterly: bool,
                     cpi: pd.Series = None, hover_suffix: str = "") -> go.Figure:
    """장르별 선 그래프. cpi를 주면 소비자물가 선을 점선으로 함께 그린다."""
    fig = go.Figure()
    for g in genres:
        if g not in s.columns:
            continue
        labels = [qlabel(t) if quarterly else t.strftime("%Y-%m") for t in s.index]
        fig.add_trace(go.Scatter(
            x=s.index, y=s[g], mode="lines", name=name(g), customdata=labels,
            line=dict(color=COLORS[g], width=2.5, dash="dot" if g == "합계" else "solid"),
            hovertemplate="<b>" + name(g) + "</b><br>%{customdata}" + hover_suffix + "<br>%{y:,.1f}<extra></extra>"))
    if cpi is not None and len(cpi):
        fig.add_trace(go.Scatter(
            x=cpi.index, y=cpi.to_numpy(), mode="lines", name="소비자물가", customdata=[t.strftime("%Y-%m") for t in cpi.index],
            line=dict(color=CPI_COLOR, width=2, dash="dash"),
            hovertemplate="<b>소비자물가</b><br>%{customdata}" + hover_suffix + "<br>%{y:,.1f}<extra></extra>"))
    if ref100:
        fig.add_hline(y=100, line_dash="dot", line_color="#b4bac4", line_width=1)
    fig.update_yaxes(title=ytitle)
    fig.update_xaxes(tickformat="%Y", dtick="M12", hoverformat="%Y-%m")
    xs = [s.index.min(), s.index.max()] if len(s) else []
    if xs:
        add_covid_band(fig, xs[0], xs[1])
    return style(fig)


def make_share_figure(share: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    xs = [qlabel(t) for t in share.index]
    for g in GENRES:
        fig.add_trace(go.Bar(
            x=xs, y=share[g], name=name(g), marker=dict(color=COLORS[g], line=dict(color="white", width=2)),
            hovertemplate="<b>" + name(g) + "</b><br>%{x}<br>%{y:.1f}%<extra></extra>"))
    fig.update_layout(barmode="stack")
    fig.update_yaxes(title="7개 장르 중 비중 (%)", range=[0, 100])
    fig.update_xaxes(type="category", tickangle=-45)
    return style(fig)


def make_quadrant_figure(dec_all: pd.DataFrame, inc: pd.Series) -> go.Figure:
    """관객 수 배수(가로) vs 물가 반영 가격 배수(세로). 점 크기 = 예매액 증가분. 전체 공연은 회색 마름모."""
    fig = go.Figure()
    top = float(inc[GENRES].clip(lower=0).max()) or 1.0
    for g in GENRES + TOTAL:
        if g not in dec_all.index:
            continue
        q, p = dec_all.loc[g, "예매수"], dec_all.loc[g, "장당 가격(물가 반영)"]
        if pd.isna(q) or pd.isna(p):
            continue
        if g == "합계":
            marker = dict(symbol="diamond", size=16, color=COLORS[g], line=dict(color="white", width=1.5))
        else:
            size = 14 + 56 * (max(float(inc.get(g, 0)), 0) / top) ** 0.5
            marker = dict(size=size, color=COLORS[g], opacity=0.85, line=dict(color="white", width=1.5))
        fig.add_trace(go.Scatter(
            x=[q], y=[p], mode="markers+text", name=name(g), text=[name(g)], textposition="top center", marker=marker,
            hovertemplate=f"<b>{name(g)}</b><br>예매수 %{{x:.2f}}배<br>가격(물가 반영) %{{y:.2f}}배<br>증가분 {float(inc.get(g, 0)):,.0f}억 원<extra></extra>"))
    fig.add_hline(y=1, line_dash="dot", line_color="#b4bac4", line_width=1)
    fig.add_vline(x=1, line_dash="dot", line_color="#b4bac4", line_width=1)
    fig.update_xaxes(title="예매수 (2019년 연간 = 1배)")
    fig.update_yaxes(title="장당 가격, 물가 반영 (2019년 연간 = 1배)")
    fig.update_layout(showlegend=False)
    return style(fig)


# ---------------------------------------------------------------- 문장
def fmt_pct(x: float) -> str:
    return "-" if pd.isna(x) else f"{x:,.0f}%"


def result_table(dec, ttm_amount, base_amount):
    """결론 표: 행 = 전체 공연 + 7개 장르, 열 = 예매액·예매수·장당 가격(2019년 연간 = 100%) + 비중(7개 장르 중)."""
    if dec is None:
        return None
    idx = [g for g in TOTAL + GENRES if g in dec.index]
    tot7 = float(ttm_amount[GENRES].sum().sum()) if ttm_amount is not None and len(ttm_amount) == 12 else np.nan
    share = {g: (np.nan if pd.isna(tot7) or tot7 <= 0 else float(ttm_amount[g].sum()) / tot7 * 100) for g in GENRES}
    share["합계"] = np.nan if pd.isna(tot7) or tot7 <= 0 else 100.0
    t19 = float(base_amount["합계"]) if base_amount is not None and "합계" in base_amount.index else np.nan
    contrib = {g: (np.nan if pd.isna(t19) or t19 <= 0 else (float(ttm_amount[g].sum()) - float(base_amount[g])) / t19 * 100)
               for g in GENRES + TOTAL}
    df = pd.DataFrame({"장르": [name(g) for g in idx],
                       "예매액": [dec.loc[g, "명목 예매액"] * 100 for g in idx],
                       "예매수": [dec.loc[g, "예매수"] * 100 for g in idx],
                       "장당 가격": [dec.loc[g, "장당 가격(명목)"] * 100 for g in idx],
                       "비중": [share[g] for g in idx],
                       "성장 기여도": [contrib[g] for g in idx]})
    return df.set_index("장르")


def style_extremes(df):
    """7개 장르 중 열별 최고는 굵은 파란색, 최저는 굵은 주황색 (전체 공연 행은 제외하고 비교)."""
    genre_rows = [name(g) for g in GENRES if name(g) in df.index]

    def paint(col):
        sub = col.loc[genre_rows].dropna()
        out = pd.Series("", index=col.index)
        if len(sub) >= 2:
            out[sub.idxmax()] = "font-weight:700; color:#1a5fb4; background-color:#e3edfb"
            out[sub.idxmin()] = "font-weight:700; color:#b34a14; background-color:#fdeadf"
        if name("합계") in out.index:
            out[name("합계")] = out[name("합계")] or "font-weight:700"
        return out

    return df.style.apply(paint, axis=0).format("{:,.0f}%", na_rep="-").format("{:,.1f}%", subset=["비중"], na_rep="-").format("{:+,.0f}%p", subset=["성장 기여도"], na_rep="-")


def result_sentences(dec, top_inc: str = None) -> list:
    out = []
    if dec is not None:
        for g, lead in [("합계", "전체 공연"), (top_inc, f"{name(top_inc) if top_inc else ''}(증가분 1위)")]:
            if g and g in dec.index:
                c = contributions(dec.loc[g])
                if c:
                    out.append(f"**{lead} 성장의 구성** (로그 기준): 관객 수 {c['관객 수']:.0f}%, "
                               f"가격(물가 반영) {c['가격(물가 반영)']:.0f}%, 물가 {c['물가 상승']:.0f}%")
    return out


def limit_lines(neg_months: int, share: float) -> list:
    return [
        "**예매액:** 예매일자 기준(공연일 아님), 주최자 수익 아님."
        + (f" 선택 기간에 월 값이 음수인 경우 {neg_months}건(취소 집중)." if neg_months else " 취소가 몰린 달은 음수 가능."),
        "**코로나 시기 하락:** 수요 감소 + 대규모 취소가 섞인 값.",
        "**2019년 기준이 낮음:** 2019.6.25 공연법 개정으로 전송이 의무화됨. 그 전 집계는 불완전해 2019년 상반기가 하반기보다 크게 낮고, 2019년 대비 성장에 집계 확대가 섞임.",
        "**물가 반영:** 소비자물가지수(총지수) 기준이라 공연 고유의 물가와 다를 수 있음.",
        "**장당 가격 상승:** 좌석 등급·할인·공연 구성 변화를 포함하므로 가격 인상만을 뜻하지 않음.",
        "**월별 변동:** 월별 장당 가격은 그 달 판매 공연 구성에 따라 크게 흔들려 12개월 합계로 비교.",
        "**규모 차이:**" + (f" 대중음악이 7개 장르의 {share:.0f}%." if not pd.isna(share) else "")
        + " 서커스/마술·국악·무용은 소수 공연에도 지수가 크게 흔들려 장르 간 동일 비교는 부적절.",
        "**집계 범위:** 전산 발권 공연만 포함. 수기 발권·무료 공연은 제외되어 과소 집계 가능.",
        "**대상:** 공연 티켓 시장. 음원·스트리밍 등 다른 문화 소비는 제외. 복합·대중무용은 규모가 작아 개별 표시 없이 전체에만 포함.",
    ]


# ================================================================ 화면
df, price_s = load_data()
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
<div class="title">공연예술 장르별 성장 양상 대시보드</div>
<div class="lead">2019년 이후 공연예술(전체와 장르별)은 얼마나 성장했고, 그 성장의 주된 요인은 관객 수와 가격 중 무엇인가?</div>
</div>
""", unsafe_allow_html=True)
st.caption(md(f"왼쪽에서 장르·기간을 바꾸면 차트가 함께 바뀜. 금액은 예매에서 취소를 뺀 예매일자 기준 예매액. 자료 조회일 {DATA_DATE}."))

months = [m for m in sorted(df["월"].unique())]
with st.sidebar:
    st.header("조건 선택")
    genres = st.multiselect("장르 (차트에 표시)", TOTAL + GENRES, default=["합계", "대중음악", "뮤지컬", "연극", "서양음악(클래식)"], format_func=name,
                            help="핵심 지표·결론은 항상 전체 공연과 7개 장르 기준. 선택은 차트 선만 바꿈.")
    start = st.selectbox("시작 월", months, index=0)
    end = st.selectbox("종료 월", months, index=len(months) - 1)

if start >= end:
    st.error("시작 월이 종료 월보다 앞서야 함. 기간을 다시 선택.")
    st.stop()

start_ts, end_ts = pd.Timestamp(start + "-01"), pd.Timestamp(end + "-01")
wa_n, wa_r, wc = monthly_wide(df, "순예매액"), monthly_wide(df, "실질예매액"), monthly_wide(df, "순예매수")
dec = decompose(wa_n, wa_r, wc, end_ts)
mrow = dec.loc[[g for g in GENRES if g in dec.index]] if dec is not None else None


def share_of(frame: pd.DataFrame):
    """7개 장르 중 대중음악의 몫(%, 예매액 기준). 12개월이 모두 있고 합계가 양수일 때만."""
    f = frame[GENRES]
    return float(f["대중음악"].sum() / f.sum().sum() * 100) if len(f) == 12 and f.sum().sum() > 0 else np.nan


share_ttm = share_of(last12(wa_n, end_ts))
# 예매액이 2019년 연간보다 가장 많이 늘어난 장르(규모 차이를 보완하려고 증가율이 아니라 증가분 기준)
_inc = (last12(wa_n, end_ts)[GENRES].sum() - wa_n[wa_n.index.year == BASE_YEAR][GENRES].sum()) if len(last12(wa_n, end_ts)) == 12 else None
top_inc = None if _inc is None or _inc.isna().all() or end_ts <= pd.Timestamp(f"{BASE_YEAR}-12-01") else _inc.idxmax()


_l12 = last12(price_s, end_ts)
cpi_end = float(_l12.mean() * 100) if len(_l12) == 12 and end_ts > pd.Timestamp(f"{BASE_YEAR}-12-01") else np.nan


def window(s):
    """선택한 시작~종료 월로 자른다(월 단위 인덱스)."""
    return s[(s.index >= start_ts) & (s.index <= end_ts)]


# ---------------------------------------------------------------- 핵심 지표
def kpi(col, title: str, value: str, sub: str, help_text: str):
    with col:
        with st.container(border=True):
            st.metric(title, value, help=help_text)
            st.caption(md(sub))


st.caption(md(f"{start} ~ {end} · 핵심 지표: 종료 월까지 최근 12개월 ÷ 2019년 연간"))
k0, k1, k2, k3 = st.columns(4)
need = "종료 월을 2019-12 이후로 선택"
_tot = dec.loc["합계", "명목 예매액"] if dec is not None and "합계" in dec.index else np.nan
if pd.isna(_tot):
    kpi(k0, "전체 공연 예매액", "-", need, "전체 공연 최근 12개월 예매액 ÷ 2019년 연간.")
else:
    kpi(k0, "전체 공연 예매액", f"{_tot * 100:,.0f}%", "최근 12개월 ÷ 2019년 연간",
        "복합·대중무용 포함 전체 공연의 예매액.")

_a19 = wa_n[wa_n.index.year == BASE_YEAR][GENRES].sum()
ser = mrow["명목 예매액"].dropna() if mrow is not None else pd.Series(dtype=float)
if ser.empty:
    kpi(k1, "예매액 성장 1위 장르", "-", need, "7개 장르 중 최근 12개월 예매액 ÷ 2019년 연간이 가장 높은 장르.")
else:
    top = ser.idxmax()
    kpi(k1, "예매액 성장 1위 장르", name(top), f"2019년의 {ser[top] * 100:,.0f}% (2019년 예매액 {_a19[top]:,.0f}억 원)",
        "7개 장르 중 최근 12개월 예매액 ÷ 2019년 연간이 가장 높은 장르. 소규모 장르가 1위일 수 있어 2019년 값 병기.")

_t19 = float(wa_n[wa_n.index.year == BASE_YEAR]["합계"].sum())
if top_inc is None or _inc is None or _t19 <= 0:
    kpi(k2, "성장 기여도 1위 장르", "-", need, "7개 장르 중 성장 기여도(%p)가 가장 큰 장르.")
else:
    _all_inc = float(last12(wa_n, end_ts)["합계"].sum()) - _t19
    kpi(k2, "성장 기여도 1위 장르", name(top_inc),
        f"{_inc[top_inc] / _t19 * 100:+,.0f}%p (전체 {_all_inc / _t19 * 100:+,.0f}%p 중, 증가분 {_inc[top_inc]:,.0f}억 원)",
        "증가분 = 최근 12개월 예매액 − 2019년 연간 예매액. 성장 기여도 = 증가분 ÷ 2019년 전체 예매액. 7개 장르 중 가장 큰 장르.")


def share_by_genre(frame: pd.DataFrame):
    """7개 장르 중 각 장르의 몫(%, 예매액 기준). 12개월이 모두 있고 합계가 양수일 때만."""
    f = frame[GENRES]
    return f.sum() / f.sum().sum() * 100 if len(f) == 12 and f.sum().sum() > 0 else None


_sh_now = share_by_genre(last12(wa_n, end_ts))
_sh_base = None if end_ts <= pd.Timestamp(f"{BASE_YEAR}-12-01") else share_by_genre(wa_n[wa_n.index.year == BASE_YEAR])
if _sh_now is None:
    kpi(k3, "비중 1위 장르", "-", "선택한 종료 월 기준 12개월 필요", "7개 장르 합계 중 예매액 몫이 가장 큰 장르.")
else:
    _top_sh = _sh_now.idxmax()
    kpi(k3, "비중 1위 장르", name(_top_sh),
        f"{_sh_now[_top_sh]:.0f}% (7개 장르 중" + ("" if _sh_base is None else f", 2019년 {_sh_base[_top_sh]:.0f}%") + ")",
        "최근 12개월 예매액 기준, 7개 장르 합계 중 가장 큰 몫을 차지한 장르.")

with st.expander("용어 풀이"):
    st.markdown(md(
        "- **예매액(예매수):** 예매에서 취소를 뺀 티켓 금액(장수). 티켓을 산 날짜로 집계.\n"
        f"- **성장 지수({BASE_YEAR}년 = 100):** 그 달까지 **최근 12개월 합계**가 {BASE_YEAR}년 연간 합계의 몇 %인가. ex) 200이면 {BASE_YEAR}년의 두 배.\n"
        "- **장당 평균 가격:** 예매액 ÷ 예매수 (만 원).\n"
        f"- **물가 반영 가격:** (월별 예매액 ÷ 그 달 물가)를 12개월 합산해 총 예매수로 나눈 값. 그 달 물가 = 월별 소비자물가지수 ÷ {BASE_YEAR}년 평균.\n"
        "- **물가 상승(물가 요인):** 명목 예매액 배수 ÷ 물가 반영 예매액 배수. 예매액이 많은 달의 물가에 더 큰 비중.\n"
        "- **성장의 구성:** 명목 예매액 = 예매수 × 장당 가격(물가 반영) × 물가 요인. 몫은 로그 기준, 합 100%.\n"
        "- **비중:** 7개 장르 합계 중 각 장르의 몫."))

tab_rev, tab_cnt, tab_price, tab_share = st.tabs(["예매액", "예매수", "장당 가격과 물가", "장르 비중"])

growth_note = ("각 점: 그 달까지 최근 12개월 합계 ÷ 2019년 연간(2019-12 = 100). "
               "음영: 참고용 코로나 구간(첫 확진~거리두기 전면 해제), 분석 기준 아님.")
level_note = "월별 값. 계절성이 있고, 취소가 몰린 달은 작아지거나 음수. 소규모 장르는 성장 보기에서 비교."


def growth_tab(w_all: pd.DataFrame, level_title: str, key: str, label: str):
    view = st.radio("보기", [GROWTH_VIEW, LEVEL_VIEW], horizontal=True, key=key)
    if not genres:
        st.info("왼쪽에서 장르를 하나 이상 선택.")
    elif view == GROWTH_VIEW:
        g = window(rolling_index(w_all))
        if g.dropna(how="all").empty:
            st.warning("선택 기간에 계산할 값 없음. 종료 월을 2019-12 이후로, 기간을 넓힐 것.")
        else:
            st.plotly_chart(make_line_figure(g, genres, "2019년 연간 = 100 (최근 12개월 합계 기준)", ref100=True, quarterly=False,
                                             hover_suffix=" (최근 12개월)"))
            st.caption(md(growth_note + (" 명목 금액(물가 미반영)." if label == "예매액" else "")))
    else:
        mv = window(w_all)
        if mv.empty:
            st.warning("선택 기간에 해당 값 없음. 기간을 넓힐 것.")
        else:
            st.plotly_chart(make_line_figure(mv, genres, level_title, ref100=False, quarterly=False))
            st.caption(md(level_note))


with tab_rev:
    st.markdown(md("**예매액 추이**(티켓 판매 금액 = 총 예매액 − 취소액)"))
    growth_tab(wa_n, "억 원", "rev_view", "예매액")

with tab_cnt:
    st.markdown(md("**예매수 추이**(티켓 판매 장수 = 총 예매수 − 취소수)"))
    growth_tab(wc, "만 장", "cnt_view", "예매수")

with tab_price:
    ALL = TOTAL + GENRES
    last_ts = pd.Timestamp(LAST_MONTH + "-01")
    dec_all = decompose(wa_n, wa_r, wc, last_ts)
    st.markdown(md("**장당 평균 가격**(예매액 ÷ 예매수): 물가보다 더 올랐나? 소비자물가지수는 이 탭에서만 사용. "
                   f"장르·기간 선택과 무관하게 **전체 장르·전체 기간(2019~{LAST_MONTH.replace('-', '.')})** 표시."))
    g = rolling_index(wa_n, denom=wc)
    st.plotly_chart(make_line_figure(g, ALL, "2019년 연간 = 100 (최근 12개월 기준)", ref100=True, quarterly=False,
                                     cpi=rolling_price_level(price_s), hover_suffix=" (최근 12개월)"))
    st.caption(md("장르 선이 점선(소비자물가)보다 위면 물가 이상으로 오른 것. 명목 가격 지수이며 물가는 최근 12개월 평균 지수. "
                  "공연별 가격 차가 커서 실제 가격은 생략하고 2019년 대비 변화만 비교. " + growth_note))
    if dec_all is not None:
        st.markdown(f"##### 장당 가격과 물가 (최근 12개월 {window_label(LAST_MONTH)}, 2019년 연간 = 100)")
        rows = [{"장르": name(gname),
                 "장당 가격·명목": round(dec_all.loc[gname, "장당 가격(명목)"] * 100, 1),
                 "장당 가격·물가 반영": round(dec_all.loc[gname, "장당 가격(물가 반영)"] * 100, 1),
                 "물가 상승": round(dec_all.loc[gname, "물가 요인"] * 100, 1)}
                for gname in ALL if gname in dec_all.index and not dec_all.loc[gname].isna().all()]
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.caption(md("명목 = 물가 반영 × 물가 상승. 물가 반영: 월별 예매액을 그 달 물가로 나눈 뒤 12개월 합산. "
                      "물가 상승: 예매액이 많은 달의 물가에 더 큰 비중."))

    if dec_all is not None:
        rows = []
        for gname in ALL:
            if gname not in dec_all.index or dec_all.loc[gname].isna().all():
                continue
            r = dec_all.loc[gname]
            c = contributions(r)
            rows.append({"장르": name(gname),
                         "명목 예매액 (2019=100)": round(r["명목 예매액"] * 100, 1),
                         "예매수 (2019=100)": round(r["예매수"] * 100, 1),
                         "장당 가격·물가 반영 (2019=100)": round(r["장당 가격(물가 반영)"] * 100, 1),
                         "물가 요인 (2019=100)": round(r["물가 요인"] * 100, 1),
                         "성장 몫: 관객 수 (%)": None if not c else round(c["관객 수"], 1),
                         "성장 몫: 가격 (%)": None if not c else round(c["가격(물가 반영)"], 1),
                         "성장 몫: 물가 (%)": None if not c else round(c["물가 상승"], 1)})
        if rows:
            st.markdown(f"##### 예매액 성장은 무엇에서 왔나 (최근 12개월 {window_label(LAST_MONTH)}, 2019년 대비)")
            st.dataframe(pd.DataFrame(rows), hide_index=True)
            st.caption(md("명목 예매액 = 예매수 × 장당 가격(물가 반영) × 물가 요인. 몫은 명목 예매액이 늘어난 장르만 계산, 세 요인 합 100%. 음수는 성장을 깎은 요인."))

            st.markdown("##### 관객 수 vs 가격: 성장은 어느 쪽에서 왔나")
            _inc_all = last12(wa_n, last_ts).sum() - wa_n[wa_n.index.year == BASE_YEAR].sum()
            st.plotly_chart(make_quadrant_figure(dec_all, _inc_all))
            st.caption(md("최근 12개월 ÷ 2019년 연간. 점 크기: 예매액 증가분(억 원). 오른쪽일수록 관객 증가, 위쪽일수록 물가 이상의 가격 상승. "
                          "회색 마름모: 전체 공연. 규모가 작은 장르는 기준에 따라 크게 흔들림."))

with tab_share:
    st.markdown(md("**7개 장르 비중**(분기별, 명목 예매액 기준). 장르 구성 변화 확인용."))
    qa = to_quarterly(wa_n)[GENRES]
    qa = in_range(qa, "분기별", start_ts, end_ts)
    qa = qa[(qa > 0).all(axis=1)]            # 취소가 몰려 음수가 된 분기는 비중을 계산하지 않음
    if qa.empty:
        st.warning("선택 기간에 비중을 계산할 분기 없음. 기간을 넓힐 것.")
    else:
        share = qa.div(qa.sum(axis=1), axis=0) * 100
        st.plotly_chart(make_share_figure(share))
        st.caption(md("예매액이 음수인 분기는 제외. 비중이 작은 장르는 막대에서 안 보일 수 있음."))

# ---------------------------------------------------------------- 결론 (탭 아래)
neg_view = int((in_range(wa_n, "월별", start_ts, end_ts)[genres] < 0).sum().sum()) if genres else 0
sents = result_sentences(dec, top_inc)
with st.container(border=True):
    st.subheader("결론")
    rt = result_table(dec, last12(wa_n, end_ts), wa_n[wa_n.index.year == BASE_YEAR].sum())
    if rt is not None:
        st.dataframe(style_extremes(rt))
        cpi_txt = "" if pd.isna(cpi_end) else f"같은 기간 소비자물가 {cpi_end:.0f}%.  \n"
        st.caption(md("예매액: 티켓 판매 금액(총 예매액 − 취소액).  \n"
                      "예매수: 판매 장수(총 예매수 − 취소수).  \n"
                      "장당 가격: 예매액 ÷ 예매수(명목).   \n"
                      "비중: 7개 장르 예매액 합계 중 몫.  \n"
                      "성장 기여도: (최근 12개월 − 2019년) 예매액 ÷ 2019년 전체 예매액. 장르별 합이 전체 성장분(%p). 복합·대중무용 분은 표에 없음.  \n"
                      + cpi_txt))
                      
    for t in sents:
        st.markdown("- " + md(t))
    if not sents:
        st.markdown("- 선택 조건에서 요약할 결과 없음. 종료 월과 기간을 확인.")
with st.expander("이 결론의 한계", expanded=True):
    for t in limit_lines(neg_view, share_ttm):
        st.markdown("- " + md(t))

with st.expander("출처와 집계 기준"):
    st.markdown(md(
        "- **출처:** 예술경영지원센터 공연예술통합전산망(KOPIS) 예매통계 ‘장르별통계’, 월 단위 조회 후 병합.\n"
        f"- **조회일:** {DATA_DATE} (최종집계 2026.11.26). 통계 갱신에 따라 값이 달라질 수 있음.\n"
        "- **물가:** 국가데이터처(통계청) 소비자물가지수 총지수(2020=100), 월별.\n"
        f"- **기간:** 2019.01~{LAST_MONTH.replace('-', '.')}. 2026.11은 11/25까지의 부분 자료라 제외. 2018년 이전은 집계가 불완전해 제외(2018년 연간 공연건수 5,387건).\n"
        "- **2019년 집계:** 2019.6.25 개정 공연법 시행으로 예매처·제작사의 전송이 의무화됨. 그 전에는 동의한 제작사만 집계(포괄율 약 38%, 2017년 기준).\n"
        "- **단위:** 예매액은 천원을 억 원으로, 예매수는 만 장으로 환산. 예매액·예매수 모두 예매에서 취소를 뺀 값(예매일자 기준).\n"
        "- **공연건수:** 월별 합산 시 중복이 생겨 사용하지 않음.\n"
        "- **장르 분류:** KOPIS가 2023.01 장르를 개편함. 개편 전후의 정의 차이는 미확인.\n"
        "- **음영 구간:** 국내 첫 확진(2020.01)~거리두기 전면 해제(2022.04). 설명용 참고일이며 분석 기준 아님."))
