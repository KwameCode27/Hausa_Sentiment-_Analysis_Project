import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    models = ["Logistic Regression", "MNB"]
    count_vectorizer = [77.30, 73.45]
    tfidf = [77.13, 74.66]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7.2, 5.1))
    ax.set_facecolor("#f2f2f2")
    fig.patch.set_facecolor("#f2f2f2")

    ax.bar(x - width / 2, count_vectorizer, width=width,
           label="CountVectorizer", color="#7c9ecf", alpha=0.8)
    ax.bar(x + width / 2, tfidf, width=width,
           label="TF-IDF", color="#4d7fb4", alpha=0.85)

    ax.set_title(
        "Performance Comparison of LR and MNB Techniques in\nterms of Accuracy",
        fontsize=15,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylim(70, 80)
    ax.set_yticks(np.arange(70, 81, 2))
    ax.tick_params(axis='y', labelsize=10)
    ax.grid(axis="y", linestyle="-", linewidth=0.8, alpha=0.6)
    ax.set_axisbelow(True)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")

    leg = ax.legend(frameon=True, edgecolor="0.7")
    for text in leg.get_texts():
        text.set_fontsize(10)

    plt.tight_layout()
    output_path = "reports/model_performance_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved chart to {output_path}")


if __name__ == "__main__":
    main()
