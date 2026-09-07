from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path

from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.shapes import Drawing, Line, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_DIRECTORY = PROJECT_ROOT / "outputs/cifar10-resnet18-pretrained-baseline"
OUTPUT_PATH = PROJECT_ROOT / "output/pdf/semantic_drift_baseline_report.pdf"

NAVY = colors.HexColor("#102A43")
INK = colors.HexColor("#243B53")
MUTED = colors.HexColor("#627D98")
TEAL = colors.HexColor("#0E8F95")
ORANGE = colors.HexColor("#F28C45")
PALE_BLUE = colors.HexColor("#EAF4F6")
PALE_ORANGE = colors.HexColor("#FFF1E6")
PALE_GRAY = colors.HexColor("#F4F7FA")
RULE = colors.HexColor("#D9E2EC")
WHITE = colors.white


def load_run() -> tuple[dict, dict, list[dict[str, float]]]:
    summary = json.loads((RUN_DIRECTORY / "summary.json").read_text(encoding="utf-8"))
    config = json.loads((RUN_DIRECTORY / "config.json").read_text(encoding="utf-8"))
    with (RUN_DIRECTORY / "metrics.csv").open(encoding="utf-8") as stream:
        raw_metrics = list(csv.DictReader(stream))
    metrics = [
        {
            "epoch": int(row["epoch"]),
            "learning_rate": float(row["learning_rate"]),
            "train_loss": float(row["train_loss"]),
            "train_accuracy": float(row["train_accuracy"]),
            "validation_loss": float(row["validation_loss"]),
            "validation_accuracy": float(row["validation_accuracy"]),
        }
        for row in raw_metrics
    ]
    return summary, config, metrics


def styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "Title",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=24,
            leading=28,
            textColor=WHITE,
            alignment=TA_LEFT,
            spaceAfter=4,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=10.5,
            leading=14,
            textColor=colors.HexColor("#D9EAF0"),
        ),
        "section": ParagraphStyle(
            "Section",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=17,
            textColor=NAVY,
            spaceBefore=5,
            spaceAfter=7,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.4,
            leading=14,
            textColor=INK,
            spaceAfter=7,
        ),
        "small": ParagraphStyle(
            "Small",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.8,
            leading=10.5,
            textColor=MUTED,
        ),
        "table_header": ParagraphStyle(
            "TableHeader",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=8.2,
            leading=10,
            textColor=WHITE,
            alignment=TA_LEFT,
        ),
        "table_cell": ParagraphStyle(
            "TableCell",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8.2,
            leading=10.5,
            textColor=INK,
        ),
        "kpi_value": ParagraphStyle(
            "KpiValue",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=24,
            textColor=NAVY,
            alignment=TA_CENTER,
        ),
        "kpi_label": ParagraphStyle(
            "KpiLabel",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=7.5,
            leading=9,
            textColor=MUTED,
            alignment=TA_CENTER,
        ),
        "callout": ParagraphStyle(
            "Callout",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=14,
            textColor=NAVY,
            alignment=TA_CENTER,
        ),
    }


def page_chrome(canvas, doc) -> None:
    width, height = A4
    canvas.saveState()
    canvas.setFillColor(NAVY)
    canvas.rect(0, height - 14 * mm, width, 14 * mm, fill=1, stroke=0)
    canvas.setFont("Helvetica-Bold", 8)
    canvas.setFillColor(WHITE)
    canvas.drawString(18 * mm, height - 9 * mm, "SEMANTIC DRIFT RESEARCH PROJECT")
    canvas.setFont("Helvetica", 7.5)
    canvas.drawRightString(width - 18 * mm, height - 9 * mm, "PHASE 1 BASELINE REPORT")

    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.5)
    canvas.line(18 * mm, 13 * mm, width - 18 * mm, 13 * mm)
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 7.5)
    canvas.drawString(18 * mm, 8.5 * mm, "Nafas Mohebi | Reproducible CIFAR-10 baseline")
    canvas.drawRightString(width - 18 * mm, 8.5 * mm, f"Page {doc.page}")
    canvas.restoreState()


def title_panel(style: dict[str, ParagraphStyle]) -> Table:
    report_date = date.today().strftime("%d %B %Y")
    content = [
        Paragraph("Clean CIFAR-10 Baseline", style["title"]),
        Paragraph(
            "Transfer-learning benchmark for temporal representation drift experiments",
            style["subtitle"],
        ),
        Spacer(1, 7 * mm),
        Paragraph(
            f"Student: <b>Nafas Mohebi</b> &nbsp;&nbsp; | &nbsp;&nbsp; Date: {report_date}",
            style["subtitle"],
        ),
    ]
    table = Table([[content]], colWidths=[174 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), NAVY),
                ("LEFTPADDING", (0, 0), (-1, -1), 12 * mm),
                ("RIGHTPADDING", (0, 0), (-1, -1), 12 * mm),
                ("TOPPADDING", (0, 0), (-1, -1), 11 * mm),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10 * mm),
            ]
        )
    )
    return table


def kpi_cards(summary: dict, style: dict[str, ParagraphStyle]) -> Table:
    cards = [
        (
            f"{summary['test_accuracy'] * 100:.2f}%",
            "FINAL TEST ACCURACY",
            PALE_BLUE,
        ),
        (
            f"{summary['best_validation_accuracy'] * 100:.2f}%",
            "BEST VALIDATION ACCURACY",
            PALE_ORANGE,
        ),
        (str(summary["best_epoch"]), "SELECTED EPOCH", PALE_GRAY),
    ]
    data = [
        [
            [
                Paragraph(value, style["kpi_value"]),
                Spacer(1, 1.5 * mm),
                Paragraph(label, style["kpi_label"]),
            ]
            for value, label, _ in cards
        ]
    ]
    table = Table(data, colWidths=[55.3 * mm] * 3, hAlign="CENTER")
    commands: list[tuple] = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4 * mm),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4 * mm),
        ("TOPPADDING", (0, 0), (-1, -1), 5 * mm),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5 * mm),
        ("BOX", (0, 0), (-1, -1), 0.7, RULE),
        ("INNERGRID", (0, 0), (-1, -1), 0.7, WHITE),
    ]
    for column, (_, _, background) in enumerate(cards):
        commands.append(("BACKGROUND", (column, 0), (column, 0), background))
    table.setStyle(TableStyle(commands))
    return table


def definition_table(style: dict[str, ParagraphStyle]) -> Table:
    rows = [
        ["Component", "Canonical baseline choice"],
        ["Dataset", "CIFAR-10: 45,000 train / 5,000 validation / 10,000 test"],
        ["Backbone", "torchvision ResNet-18 with official ImageNet weights"],
        ["Trainable head", "512 -> 128 projection + ReLU -> 10-class linear classifier"],
        ["Transfer policy", "Frozen backbone and frozen BatchNorm running statistics"],
        ["Input pipeline", "224 x 224; ImageNet normalization; train-only crop and flip"],
    ]
    formatted = [
        [Paragraph(cell, style["table_header"]) for cell in rows[0]],
        *[
            [Paragraph(cell, style["table_cell"]) for cell in row]
            for row in rows[1:]
        ],
    ]
    table = Table(formatted, colWidths=[42 * mm, 124 * mm], repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, PALE_GRAY]),
                ("GRID", (0, 0), (-1, -1), 0.45, RULE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3 * mm),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3 * mm),
                ("TOPPADDING", (0, 0), (-1, -1), 2.5 * mm),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5 * mm),
            ]
        )
    )
    return table


def architecture_diagram() -> Drawing:
    drawing = Drawing(470, 120)
    boxes = [
        (5, 40, 80, 44, "CIFAR-10", "224 x 224"),
        (105, 40, 100, 44, "ResNet-18", "ImageNet, frozen"),
        (225, 40, 85, 44, "Projection", "512 -> 128"),
        (330, 40, 75, 44, "Classifier", "128 -> 10"),
        (425, 40, 40, 44, "Class", "label"),
    ]
    for index, (x, y, width, height, title, subtitle) in enumerate(boxes):
        fill = PALE_BLUE if index in {0, 1} else PALE_ORANGE
        drawing.add(Rect(x, y, width, height, 7, 7, fillColor=fill, strokeColor=RULE))
        drawing.add(
            String(
                x + width / 2,
                y + 26,
                title,
                fontName="Helvetica-Bold",
                fontSize=9,
                fillColor=NAVY,
                textAnchor="middle",
            )
        )
        drawing.add(
            String(
                x + width / 2,
                y + 13,
                subtitle,
                fontName="Helvetica",
                fontSize=6.8,
                fillColor=MUTED,
                textAnchor="middle",
            )
        )
        if index < len(boxes) - 1:
            next_x = boxes[index + 1][0]
            y_mid = y + height / 2
            drawing.add(Line(x + width + 3, y_mid, next_x - 7, y_mid, strokeColor=TEAL))
            drawing.add(Line(next_x - 11, y_mid + 3, next_x - 7, y_mid, strokeColor=TEAL))
            drawing.add(Line(next_x - 11, y_mid - 3, next_x - 7, y_mid, strokeColor=TEAL))
    drawing.add(
        String(
            155,
            17,
            "frozen feature extraction",
            fontName="Helvetica",
            fontSize=7,
            fillColor=MUTED,
            textAnchor="middle",
        )
    )
    drawing.add(
        String(
            315,
            17,
            "trainable parameters",
            fontName="Helvetica-Bold",
            fontSize=7,
            fillColor=ORANGE,
            textAnchor="middle",
        )
    )
    return drawing


def accuracy_chart(metrics: list[dict[str, float]]) -> Drawing:
    drawing = Drawing(470, 220)
    chart = LinePlot()
    chart.x = 45
    chart.y = 35
    chart.width = 400
    chart.height = 155
    train = [(row["epoch"], row["train_accuracy"] * 100) for row in metrics]
    validation = [(row["epoch"], row["validation_accuracy"] * 100) for row in metrics]
    chart.data = [train, validation]
    chart.joinedLines = 1
    chart.lines[0].strokeColor = TEAL
    chart.lines[0].strokeWidth = 2.2
    chart.lines[1].strokeColor = ORANGE
    chart.lines[1].strokeWidth = 2.2
    chart.xValueAxis.valueMin = 1
    chart.xValueAxis.valueMax = 10
    chart.xValueAxis.valueSteps = list(range(1, 11))
    chart.xValueAxis.labelTextFormat = "%d"
    chart.xValueAxis.labels.fontName = "Helvetica"
    chart.xValueAxis.labels.fontSize = 7
    chart.xValueAxis.strokeColor = RULE
    chart.yValueAxis.valueMin = 78
    chart.yValueAxis.valueMax = 91
    chart.yValueAxis.valueStep = 2
    chart.yValueAxis.labels.fontName = "Helvetica"
    chart.yValueAxis.labels.fontSize = 7
    chart.yValueAxis.labelTextFormat = "%d%%"
    chart.yValueAxis.strokeColor = RULE
    chart.yValueAxis.visibleGrid = 1
    chart.yValueAxis.gridStrokeColor = RULE
    chart.yValueAxis.gridStrokeDashArray = (2, 3)
    drawing.add(chart)

    drawing.add(String(245, 5, "Epoch", fontName="Helvetica-Bold", fontSize=8, fillColor=MUTED))
    drawing.add(Rect(145, 203, 10, 3, fillColor=TEAL, strokeColor=TEAL))
    drawing.add(String(160, 199, "Train accuracy", fontName="Helvetica", fontSize=8, fillColor=INK))
    drawing.add(Rect(275, 203, 10, 3, fillColor=ORANGE, strokeColor=ORANGE))
    drawing.add(
        String(290, 199, "Validation accuracy", fontName="Helvetica", fontSize=8, fillColor=INK)
    )
    return drawing


def hyperparameter_table(config: dict, style: dict[str, ParagraphStyle]) -> Table:
    training = config["training"]
    rows = [
        ["Parameter", "Value", "Parameter", "Value"],
        ["Optimizer", "AdamW", "Epochs", str(training["epochs"])],
        ["Learning rate", f"{training['learning_rate']:.4f}", "Batch size", "128"],
        ["Weight decay", f"{training['weight_decay']:.4f}", "Label smoothing", "0.1"],
        ["Scheduler", "Cosine annealing", "Seed", str(config["seed"])],
        ["Checkpoint rule", "Best validation accuracy", "Device", "Apple MPS"],
    ]
    formatted = [
        [Paragraph(cell, style["table_header"]) for cell in rows[0]],
        *[
            [Paragraph(cell, style["table_cell"]) for cell in row]
            for row in rows[1:]
        ],
    ]
    table = Table(formatted, colWidths=[35 * mm, 48 * mm, 35 * mm, 48 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, PALE_GRAY]),
                ("GRID", (0, 0), (-1, -1), 0.45, RULE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 2.5 * mm),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2.5 * mm),
                ("TOPPADDING", (0, 0), (-1, -1), 2.3 * mm),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.3 * mm),
            ]
        )
    )
    return table


def epoch_table(metrics: list[dict[str, float]], style: dict[str, ParagraphStyle]) -> Table:
    rows = [["Epoch", "Learning rate", "Train loss", "Train accuracy", "Val loss", "Val accuracy"]]
    for row in metrics:
        rows.append(
            [
                str(row["epoch"]),
                f"{row['learning_rate']:.6f}",
                f"{row['train_loss']:.4f}",
                f"{row['train_accuracy'] * 100:.2f}%",
                f"{row['validation_loss']:.4f}",
                f"{row['validation_accuracy'] * 100:.2f}%",
            ]
        )
    formatted = [
        [Paragraph(cell, style["table_header"]) for cell in rows[0]],
        *[
            [Paragraph(cell, style["table_cell"]) for cell in row]
            for row in rows[1:]
        ],
    ]
    table = Table(
        formatted,
        colWidths=[18 * mm, 31 * mm, 28 * mm, 31 * mm, 28 * mm, 30 * mm],
        repeatRows=1,
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, PALE_GRAY]),
                ("GRID", (0, 0), (-1, -1), 0.4, RULE),
                ("ALIGN", (0, 1), (-1, -1), "RIGHT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 2 * mm),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2 * mm),
                ("TOPPADDING", (0, 0), (-1, -1), 1.8 * mm),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 1.8 * mm),
                ("BACKGROUND", (0, 9), (-1, 9), PALE_ORANGE),
                ("TEXTCOLOR", (0, 9), (-1, 9), NAVY),
                ("FONTNAME", (0, 9), (-1, 9), "Helvetica-Bold"),
            ]
        )
    )
    return table


def build_report() -> None:
    summary, config, metrics = load_run()
    style = styles()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    document = BaseDocTemplate(
        str(OUTPUT_PATH),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=20 * mm,
        bottomMargin=18 * mm,
        title="Clean CIFAR-10 Baseline Report",
        author="Nafas Mohebi",
        subject="Phase 1 baseline for semantic representation drift experiments",
    )
    frame = Frame(
        document.leftMargin,
        document.bottomMargin,
        document.width,
        document.height,
        id="content",
        leftPadding=0,
        rightPadding=0,
        topPadding=0,
        bottomPadding=0,
    )
    document.addPageTemplates([PageTemplate(id="report", frames=[frame], onPage=page_chrome)])

    story = [
        title_panel(style),
        Spacer(1, 8 * mm),
        kpi_cards(summary, style),
        Spacer(1, 8 * mm),
        Paragraph("Executive summary", style["section"]),
        Paragraph(
            "A clean transfer-learning baseline has been established for the semantic "
            "representation drift project. An ImageNet-pretrained ResNet-18 was used as "
            "a frozen feature extractor, while a compact 10-class head was trained on "
            "CIFAR-10. The selected checkpoint achieved <b>87.07% test accuracy</b> and "
            "is now the canonical no-drift reference for all later degradation and "
            "realignment comparisons.",
            style["body"],
        ),
        Paragraph("Completed baseline definition", style["section"]),
        definition_table(style),
        Spacer(1, 7 * mm),
        Table(
            [[Paragraph("Baseline status: COMPLETE AND REPRODUCIBLE", style["callout"])]],
            colWidths=[166 * mm],
            style=TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), PALE_BLUE),
                    ("BOX", (0, 0), (-1, -1), 0.8, TEAL),
                    ("TOPPADDING", (0, 0), (-1, -1), 4 * mm),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4 * mm),
                ]
            ),
        ),
        PageBreak(),
        Paragraph("Model and data flow", style["section"]),
        Paragraph(
            "The pretrained convolutional backbone converts each image into a 512-D "
            "feature vector. Only the 128-D projection and the final 10-class classifier "
            "are optimized. This clean separation is useful later: the trained baseline "
            "can be wrapped as aligned transmitter and receiver endpoints without "
            "changing the reference task.",
            style["body"],
        ),
        architecture_diagram(),
        Spacer(1, 2 * mm),
        Paragraph("Training trajectory", style["section"]),
        accuracy_chart(metrics),
        Spacer(1, 2 * mm),
        KeepTogether(
            [
                Paragraph("Observed behavior", style["section"]),
                Paragraph(
                    "Validation accuracy improved from 85.54% to a peak of 88.04% at "
                    "epoch 9. The final train-validation gap at the selected checkpoint "
                    "was 1.35 percentage points, indicating stable head training with no "
                    "large overfitting gap. Epoch 9 was selected before one-time test "
                    "evaluation.",
                    style["body"],
                ),
            ]
        ),
        PageBreak(),
        Paragraph("Exact training configuration", style["section"]),
        hyperparameter_table(config, style),
        Spacer(1, 7 * mm),
        Paragraph("Per-epoch evidence", style["section"]),
        epoch_table(metrics, style),
        Spacer(1, 7 * mm),
        Paragraph("Reproducibility and artifacts", style["section"]),
        Paragraph(
            "The run is defined by <b>configs/baseline/cifar10.yaml</b>. Raw metrics, the "
            "resolved configuration, the best checkpoint, and the machine-readable "
            "summary are stored under <b>outputs/cifar10-resnet18-pretrained-baseline/</b>. "
            "The full run used seed 42 and completed in 1,066.14 seconds on Apple MPS. "
            "All five software tests and the code quality check passed after training.",
            style["body"],
        ),
        Table(
            [
                [
                    Paragraph(
                        "NEXT MILESTONE",
                        ParagraphStyle(
                            "NextLabel",
                            parent=style["kpi_label"],
                            textColor=TEAL,
                            alignment=TA_LEFT,
                        ),
                    ),
                    Paragraph(
                        "Wrap the baseline as an initially aligned transmitter-receiver "
                        "pair and add a class-prototype semantic codebook.",
                        style["body"],
                    ),
                ]
            ],
            colWidths=[35 * mm, 131 * mm],
            style=TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), PALE_BLUE),
                    ("BOX", (0, 0), (-1, -1), 0.8, TEAL),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4 * mm),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4 * mm),
                    ("TOPPADDING", (0, 0), (-1, -1), 4 * mm),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 2 * mm),
                ]
            ),
        ),
    ]
    document.build(story)


if __name__ == "__main__":
    build_report()
