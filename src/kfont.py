"""한글 폰트 고정 (Phase 00.5, C). 모든 final figure 는 import 시 setup_korean_font() 를 호출한다.
시스템 설치 폰트만 사용 (repo 에 폰트 파일 복사 금지). 실패 시 RuntimeError — 깨진 glyph figure 를 만들지 않는다."""
import warnings
import matplotlib
from matplotlib import font_manager

CANDIDATES = ["Noto Sans CJK KR", "NanumGothic", "NanumBarunGothic", "Noto Sans KR", "Malgun Gothic"]


def setup_korean_font(strict=True):
    names = {f.name for f in font_manager.fontManager.ttflist}
    for fam in CANDIDATES:
        if fam in names:
            path = font_manager.findfont(font_manager.FontProperties(family=fam), fallback_to_default=False)
            matplotlib.rcParams["font.family"] = fam
            matplotlib.rcParams["axes.unicode_minus"] = False
            return fam, path
    if strict:
        raise RuntimeError(f"Korean font not found among {CANDIDATES}")
    matplotlib.rcParams["axes.unicode_minus"] = False
    return None, None


def smoke_test(out_path):
    """한글 glyph 렌더 smoke test. missing-glyph 경고 수를 반환 (0 이어야 PASS)."""
    import matplotlib.pyplot as plt
    fam, path = setup_korean_font()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fig, ax = plt.subplots(figsize=(6, 2.4))
        ax.plot([0, 1, 2], [0, -1, 2])
        ax.set_title("한글 폰트 점검: KODEX 200 · 인버스 · 국고채10년 · 골드선물(H) −3%")
        ax.set_xlabel("거래일"); ax.set_ylabel("로그수익률 (−/+)")
        fig.savefig(out_path, dpi=110, bbox_inches="tight"); plt.close(fig)
    missing = [str(x.message) for x in w if "missing" in str(x.message).lower() or "glyph" in str(x.message).lower()]
    return {"family": fam, "path": path, "missing_glyph_warnings": len(missing), "examples": missing[:3]}
