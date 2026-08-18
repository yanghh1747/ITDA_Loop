import numpy as np
import matplotlib.pyplot as plt


# =========================
# 1. 모델 성능 데이터
# =========================

models = [
    "XGBoost",
    "LightGBM",
    "7-day Average"
]

# R2
r2_train = [0.982, 0.918, 0.523]
r2_test  = [0.523, 0.565, 0.321]

# MAE
mae_train = [0.956, 2.161, 5.180]
mae_test  = [4.414, 4.288, 5.210]

# RMSE
rmse_train = [1.292, 2.781, 6.702]
rmse_test  = [5.798, 5.536, 6.915]


# =========================
# 2. 논문 스타일 색상
# =========================

train_color = "#72C9AD"   # 연한 민트/초록
test_color = "#F28E6B"    # 연한 주황


# =========================
# 3. Radar chart 함수
# =========================

def draw_radar(
    ax,
    title,
    train_values,
    test_values,
    max_value,
    radial_ticks
):

    # 논문 그림처럼
    # XGBoost = 왼쪽 위
    # LightGBM = 오른쪽 위
    # 7-day Average = 아래
    angles = np.deg2rad([150, 30, 270])

    # 도형을 닫기 위해 첫 값 반복
    angles_closed = np.append(
        angles,
        angles[0]
    )

    train_closed = np.append(
        train_values,
        train_values[0]
    )

    test_closed = np.append(
        test_values,
        test_values[0]
    )

    # =========================
    # Train
    # =========================

    ax.plot(
        angles_closed,
        train_closed,
        color=train_color,
        linewidth=2
    )

    ax.fill(
        angles_closed,
        train_closed,
        color=train_color,
        alpha=0.18
    )

    # =========================
    # Test
    # =========================

    ax.plot(
        angles_closed,
        test_closed,
        color=test_color,
        linewidth=2
    )

    ax.fill(
        angles_closed,
        test_closed,
        color=test_color,
        alpha=0.18
    )

    # =========================
    # 모델명
    # =========================

    ax.set_xticks(angles)

    ax.set_xticklabels(
        models,
        fontsize=10,
        fontweight="bold"
    )
    ax.tick_params(
    axis="x",
    pad=15
)

    # =========================
    # 반지름 범위
    # =========================

    ax.set_ylim(
        0,
        max_value
    )

    ax.set_yticks(radial_ticks)

    ax.set_yticklabels(
        [str(x) for x in radial_ticks],
        fontsize=10
    )

    # 값 숫자는 아래쪽에 위치
    ax.set_rlabel_position(270)

    # =========================
    # 격자 스타일
    # =========================

    ax.grid(
        True,
        linestyle="--",
        linewidth=0.8,
        alpha=0.55
    )

    # 바깥 원
    ax.spines["polar"].set_linewidth(1.2)
    ax.spines["polar"].set_color("black")

    # =========================
    # 제목
    # =========================

    ax.set_title(
        title,
        fontsize=14,
        fontweight="bold",
        pad=16
    )


# =========================
# 4. 전체 Figure
# =========================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 5),
    subplot_kw={"polar": True}
)

fig.patch.set_facecolor("white")


# =========================
# 5. R2
# =========================

draw_radar(
    axes[0],
    "R²",
    r2_train,
    r2_test,
    max_value=1.0,
    radial_ticks=[0.2, 0.4, 0.6, 0.8, 1.0]
)


# =========================
# 6. MAE
# =========================

draw_radar(
    axes[1],
    "MAE",
    mae_train,
    mae_test,
    max_value=6.0,
    radial_ticks=[1, 2, 3, 4, 5, 6]
)


# =========================
# 7. RMSE
# =========================

draw_radar(
    axes[2],
    "RMSE",
    rmse_train,
    rmse_test,
    max_value=8.0,
    radial_ticks=[2, 4, 6, 8]
)


# =========================
# 8. 범례
# =========================

legend_lines = [
    plt.Line2D(
        [0],
        [0],
        color=train_color,
        linewidth=2
    ),

    plt.Line2D(
        [0],
        [0],
        color=test_color,
        linewidth=2
    )
]

axes[2].legend(
    legend_lines,
    ["Training", "Testing"],
    loc="upper right",
    bbox_to_anchor=(1.25, 1.10),
    frameon=True,
    fontsize=9
)


# =========================
# 9. 전체 배치
# =========================

plt.subplots_adjust(
    left=0.05,
    right=0.95,
    top=0.88,
    bottom=0.10,
    wspace=0.35
)


# =========================
# 10. 저장
# =========================

plt.savefig(
    "data/model_performance_radar_paper_style.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()