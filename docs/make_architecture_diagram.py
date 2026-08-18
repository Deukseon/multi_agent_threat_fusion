"""
docs/architecture.png 생성 스크립트.

[2026-08-18] README가 참조하는 docs/architecture.png가 이 세션(압축 이후)에는
실제 파일로 남아있지 않아서(예전 세션에서 만든 이미지는 대화가 압축되며 유실),
Phase 2.5/2.6까지 반영된 현재 구조 기준으로 새로 그렸다. matplotlib만 사용해서
외부 graphviz 설치 없이 재현 가능하게 함 — 이 스크립트 자체도 저장소에 같이
커밋해두면, 나중에 구조가 또 바뀌었을 때 다시 실행해서 갱신할 수 있다.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.font_manager as fm

# 한글 렌더링용 폰트 지정. matplotlib은 Noto의 TTC(여러 언어 묶음) 파일에서 이름을
# 하나만("Noto Sans CJK JP") 등록하지만 실제 폰트 파일엔 한글 글리프도 포함돼 있어
# 정상 렌더링된다 (fc-list엔 "Noto Sans CJK KR"로도 보이지만 matplotlib 폰트매니저는
# 그 별칭을 등록하지 않음 — 등록된 이름 기준으로 지정해야 함).
_cjk_candidates = ["Noto Sans CJK JP", "Noto Sans CJK KR", "Noto Sans CJK SC"]
_registered = {f.name for f in fm.fontManager.ttflist}
for cand in _cjk_candidates:
    if cand in _registered:
        plt.rcParams["font.family"] = cand
        break
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(13, 9.5))
ax.set_xlim(0, 13)
ax.set_ylim(0, 9.5)
ax.axis("off")

COLOR_START = "#4a4a4a"
COLOR_SPECIALIST = "#2f6fed"
COLOR_COORD = "#c0392b"
COLOR_OUT = "#1e8449"
COLOR_NOTE = "#6b6b6b"


def box(x, y, w, h, text, fc, ec="#222222", fontsize=10.5, fontcolor="white", weight="bold"):
    b = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.06,rounding_size=0.08",
        linewidth=1.3, edgecolor=ec, facecolor=fc, zorder=3,
    )
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
             fontsize=fontsize, color=fontcolor, weight=weight, zorder=4,
             linespacing=1.4)
    return (x + w / 2, y, x + w / 2, y + h, x, y + h / 2, x + w, y + h / 2)


def arrow(p1, p2, color="#333333", lw=1.6, style="-|>", ls="solid"):
    a = FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=14,
                         linewidth=lw, color=color, zorder=2, linestyle=ls)
    ax.add_patch(a)


# 제목
ax.text(6.5, 9.15, "multi_agent_threat_fusion — 멀티에이전트 위협 융합 파이프라인",
         ha="center", va="center", fontsize=15, weight="bold")
ax.text(6.5, 8.75, "LangGraph Send API 기반 병렬 fan-out → coordinator 종합 → LLM SITREP",
         ha="center", va="center", fontsize=10.5, color=COLOR_NOTE)

# START
sx, sy, sw, sh = 5.5, 7.55, 2.0, 0.55
_, top, *_ = box(sx, sy, sw, sh, "START", COLOR_START, fontsize=11)
start_bottom = (sx + sw / 2, sy)

# dispatch_specialists 라벨 (화살표들이 모이는 지점이라 흰 배경을 깔아 겹침 방지)
ax.text(6.5, 7.15, "dispatch_specialists()\n(이번 사이클에 있는 센서만 골라 Send)",
        ha="center", va="center", fontsize=9, color=COLOR_NOTE, linespacing=1.3,
        zorder=5, bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none"))

# 전문 에이전트 4개
agent_y = 5.55
agent_h = 1.35
agent_w = 2.55
gap = 0.35
n = 4
total_w = n * agent_w + (n - 1) * gap
start_x = 6.5 - total_w / 2

agents = [
    ("radar_agent\n(레이더/항적)", "실연동: OpenSky API\n(collect_radar_observation)"),
    ("cv_agent\n(EO/IR 표적탐지)", "실연동: YOLO26-OBB\n(DOTA-v1.0 파인튜닝)"),
    ("sigint_agent\n(신호정보)", "규칙 기반\n(note 플래그 중심 채점)"),
    ("ir_anomaly_agent\n(열이상 조기경보)", "실연동: Chronos-2\n(forecast-then-compare)"),
]

agent_tops = []
for i, (title, sub) in enumerate(agents):
    x = start_x + i * (agent_w + gap)
    cx, top_y, *_ = box(x, agent_y, agent_w, agent_h, title, COLOR_SPECIALIST, fontsize=10)
    ax.text(x + agent_w / 2, agent_y - 0.32, sub, ha="center", va="top",
            fontsize=8, color=COLOR_NOTE, linespacing=1.3)
    agent_tops.append((x + agent_w / 2, agent_y + agent_h))
    arrow(start_bottom, (x + agent_w / 2, agent_y + agent_h), color="#555555")

# coordinator
cx, cy, cw, ch = 5.3, 3.35, 2.4, 0.85
coord_top = (cx + cw / 2, cy + ch)
box(cx, cy, cw, ch, "coordinator\n(신뢰도 가중평균 +\nCRITICAL 오버라이드 2종)", COLOR_COORD, fontsize=9.3)
for (ax_, ay_) in agent_tops:
    arrow((ax_, agent_y), (cx + cw / 2, cy + ch), color="#555555")

# generate_sitrep
gx, gy, gw, gh = 5.1, 2.05, 2.8, 0.75
box(gx, gy, gw, gh, "generate_sitrep\n(coordinator 요약 → Claude API 자연어 브리핑,\n실패 시 규칙기반 요약 폴백)", COLOR_OUT, fontsize=8.6)
arrow((cx + cw / 2, cy), (gx + gw / 2, gy + gh), color="#555555")

# END
ex, ey, ew, eh = 5.7, 1.05, 1.6, 0.55
box(ex, ey, ew, eh, "END", COLOR_START, fontsize=11)
arrow((gx + gw / 2, gy), (ex + ew / 2, ey + eh), color="#555555")

# 범례
legend_items = [
    (COLOR_START, "제어 노드"),
    (COLOR_SPECIALIST, "전문 에이전트 (병렬 fan-out)"),
    (COLOR_COORD, "coordinator (join)"),
    (COLOR_OUT, "출력 노드"),
]
lx, ly = 0.4, 0.55
for i, (c, label) in enumerate(legend_items):
    ax.add_patch(mpatches.Rectangle((lx + i * 3.1, ly), 0.25, 0.25, facecolor=c, edgecolor="#222"))
    ax.text(lx + i * 3.1 + 0.35, ly + 0.125, label, ha="left", va="center", fontsize=8.3)

ax.text(0.4, 0.15,
        "2026-08-18 기준: radar/cv/ir_anomaly는 실제 API·모델 연동 검증 완료(사용자 PC), "
        "sigint는 아직 규칙 기반. generate_sitrep은 코드 완료·실제 API 키 미보유로 폴백 경로만 검증.",
        ha="left", va="center", fontsize=7.6, color=COLOR_NOTE)

plt.tight_layout()
plt.savefig("/root/work/multi_agent_threat_fusion/docs/architecture.png", dpi=170, bbox_inches="tight")
print("saved")