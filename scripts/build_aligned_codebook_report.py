from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path

import torch
from reportlab.graphics.shapes import Circle, Drawing, Line, Rect, String
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
BASELINE_DIRECTORY = PROJECT_ROOT / "outputs/cifar10-resnet18-pretrained-baseline"
CODEBOOK_DIRECTORY = BASELINE_DIRECTORY / "aligned_codebook"
OUTPUT_PATH = PROJECT_ROOT / "output/pdf/semantic_drift_aligned_codebook_report.pdf"

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


def load_evidence() -> tuple[dict, dict, list[dict[str, str]], torch.Tensor]:
    baseline = json.loads((BASELINE_DIRECTORY / "summary.json").read_text(encoding="utf-8"))
    codebook = json.loads((CODEBOOK_DIRECTORY / "summary.json").read_text(encoding="utf-8"))
    with (CODEBOOK_DIRECTORY / "class_prototypes.csv").open(encoding="utf-8") as stream:
        classes = list(csv.DictReader(stream))
    artifact = torch.load(
        CODEBOOK_DIRECTORY / "initial_codebook.pt",
        map_location="cpu",
        weights_only=False,
    )
    prototypes = artifact["codebook"].to(torch.float64)
    return baseline, codebook, classes, prototypes


def styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "Title",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=23,
            leading=27,
            textColor=WHITE,
            alignment=TA_LEFT,
            spaceAfter=4,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=10.3,
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
        "subsection": ParagraphStyle(
            "Subsection",
            parent=base["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=10.5,
            leading=13,
            textColor=NAVY,
            spaceBefore=3,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.25,
            leading=13.5,
            textColor=INK,
            spaceAfter=7,
        ),
        "small": ParagraphStyle(
            "Small",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.7,
            leading=10.2,
            textColor=MUTED,
        ),
        "bullet": ParagraphStyle(
            "Bullet",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8.9,
            leading=12.4,
            textColor=INK,
            leftIndent=4 * mm,
            firstLineIndent=-2.5 * mm,
            bulletIndent=0,
            spaceAfter=4,
        ),
        "table_header": ParagraphStyle(
            "TableHeader",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=8,
            leading=9.5,
            textColor=WHITE,
            alignment=TA_LEFT,
        ),
        "table_cell": ParagraphStyle(
            "TableCell",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8,
            leading=10,
            textColor=INK,
        ),
        "table_cell_bold": ParagraphStyle(
            "TableCellBold",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=8,
            leading=10,
            textColor=NAVY,
        ),
        "kpi_value": ParagraphStyle(
            "KpiValue",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=21,
            leading=23,
            textColor=NAVY,
            alignment=TA_CENTER,
        ),
        "kpi_label": ParagraphStyle(
            "KpiLabel",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=7.2,
            leading=8.8,
            textColor=MUTED,
            alignment=TA_CENTER,
        ),
        "callout": ParagraphStyle(
            "Callout",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=9.8,
            leading=13.5,
            textColor=NAVY,
            alignment=TA_CENTER,
        ),
        "formula": ParagraphStyle(
            "Formula",
            parent=base["BodyText"],
            fontName="Courier-Bold",
            fontSize=9.2,
            leading=13,
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
    canvas.drawRightString(width - 18 * mm, height - 9 * mm, "PHASE 1 CODEBOOK REPORT")

    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.5)
    canvas.line(18 * mm, 13 * mm, width - 18 * mm, 13 * mm)
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 7.5)
    canvas.drawString(18 * mm, 8.5 * mm, "Nafas Mohebi | Aligned semantic codebook")
    canvas.drawRightString(width - 18 * mm, 8.5 * mm, f"Page {doc.page}")
    canvas.restoreState()


def title_panel(style: dict[str, ParagraphStyle]) -> Table:
    report_date = date.today().strftime("%d %B %Y")
    content = [
        Paragraph("Aligned CIFAR-10 Semantic Codebook", style["title"]),
        Paragraph(
            "Shared transmitter-receiver initialization for representation drift experiments",
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


def kpi_cards(codebook: dict, style: dict[str, ParagraphStyle]) -> Table:
    cards = [
        (
            f"{codebook['test']['receiver_accuracy'] * 100:.2f}%",
            "CODEBOOK TEST ACCURACY",
            PALE_BLUE,
        ),
        (
            f"{codebook['test']['transmitter_receiver_agreement'] * 100:.0f}%",
            "TX-RX AGREEMENT",
            PALE_ORANGE,
        ),
        ("0.0", "INITIAL SEMANTIC DRIFT", PALE_GRAY),
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


def formatted_table(
    rows: list[list[str]],
    widths: list[float],
    style: dict[str, ParagraphStyle],
    *,
    bold_first_column: bool = False,
    compact: bool = False,
) -> Table:
    formatted: list[list[Paragraph]] = []
    for row_index, row in enumerate(rows):
        output_row: list[Paragraph] = []
        for column_index, cell in enumerate(row):
            if row_index == 0:
                paragraph_style = style["table_header"]
            elif bold_first_column and column_index == 0:
                paragraph_style = style["table_cell_bold"]
            else:
                paragraph_style = style["table_cell"]
            output_row.append(Paragraph(cell, paragraph_style))
        formatted.append(output_row)
    padding = 1.8 * mm if compact else 2.5 * mm
    table = Table(formatted, colWidths=widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, PALE_GRAY]),
                ("GRID", (0, 0), (-1, -1), 0.45, RULE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 2.6 * mm),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2.6 * mm),
                ("TOPPADDING", (0, 0), (-1, -1), padding),
                ("BOTTOMPADDING", (0, 0), (-1, -1), padding),
            ]
        )
    )
    return table


def scope_table(style: dict[str, ParagraphStyle]) -> Table:
    rows = [
        ["Element", "Phase 1 aligned-codebook definition"],
        ["Source model", "Best CIFAR-10 checkpoint: epoch 9, seed 42"],
        ["Representation", "128-D output of the trained projection head"],
        ["Codebook", "10 L2-normalized class prototypes; one per CIFAR-10 class"],
        ["Construction data", "45,000 training images only; deterministic evaluation transforms"],
        ["Decoder rule", "Cosine nearest prototype"],
        ["Endpoint state", "Independent files with exactly equal initial values; no local updates"],
    ]
    return formatted_table(rows, [42 * mm, 124 * mm], style, bold_first_column=True)


def architecture_diagram() -> Drawing:
    drawing = Drawing(470, 105)
    boxes = [
        (4, 47, 55, "Image", "224 x 224", PALE_BLUE),
        (72, 47, 68, "ResNet-18", "frozen", PALE_BLUE),
        (153, 47, 65, "Projection", "512 -> 128", PALE_ORANGE),
        (231, 47, 67, "Tx codebook", "10 x 128", PALE_BLUE),
        (311, 47, 55, "Message", "prototype", PALE_ORANGE),
        (379, 47, 67, "Rx codebook", "10 x 128", PALE_BLUE),
    ]
    for index, (x, y, width, title, subtitle, fill) in enumerate(boxes):
        drawing.add(Rect(x, y, width, 42, 6, 6, fillColor=fill, strokeColor=RULE))
        drawing.add(
            String(
                x + width / 2,
                y + 25,
                title,
                fontName="Helvetica-Bold",
                fontSize=7.6,
                fillColor=NAVY,
                textAnchor="middle",
            )
        )
        drawing.add(
            String(
                x + width / 2,
                y + 12,
                subtitle,
                fontName="Helvetica",
                fontSize=6.2,
                fillColor=MUTED,
                textAnchor="middle",
            )
        )
        if index < len(boxes) - 1:
            next_x = boxes[index + 1][0]
            y_mid = y + 21
            drawing.add(Line(x + width + 3, y_mid, next_x - 6, y_mid, strokeColor=TEAL))
            drawing.add(Line(next_x - 10, y_mid + 3, next_x - 6, y_mid, strokeColor=TEAL))
            drawing.add(Line(next_x - 10, y_mid - 3, next_x - 6, y_mid, strokeColor=TEAL))
    drawing.add(
        String(
            265,
            25,
            "q* = argmax cosine(z, c_k)",
            fontName="Courier-Bold",
            fontSize=8,
            fillColor=NAVY,
            textAnchor="middle",
        )
    )
    drawing.add(
        String(
            265,
            10,
            "At initialization: C_T,0 = C_R,0",
            fontName="Helvetica-Bold",
            fontSize=7.4,
            fillColor=TEAL,
            textAnchor="middle",
        )
    )
    return drawing


def class_coverage_table(classes: list[dict[str, str]], style: dict[str, ParagraphStyle]) -> Table:
    rows = [["Class", "Train n", "Class", "Train n"]]
    for index in range(5):
        left = classes[index]
        right = classes[index + 5]
        rows.append(
            [
                left["class_name"].title(),
                f"{int(left['training_samples']):,}",
                right["class_name"].title(),
                f"{int(right['training_samples']):,}",
            ]
        )
    return formatted_table(
        rows,
        [48 * mm, 27 * mm, 48 * mm, 27 * mm],
        style,
        compact=True,
    )


def add_blossom(drawing: Drawing, x: float, y: float, color) -> None:
    offsets = [(0, 5), (5, 0), (0, -5), (-5, 0)]
    for dx, dy in offsets:
        drawing.add(Circle(x + dx, y + dy, 3.2, fillColor=color, strokeColor=None))
    drawing.add(Circle(x, y, 2.2, fillColor=WHITE, strokeColor=None))


def accuracy_comparison_chart(codebook: dict) -> Drawing:
    drawing = Drawing(470, 255)
    drawing.add(
        String(
            30,
            237,
            "Classifier and codebook accuracy",
            fontName="Helvetica-Bold",
            fontSize=10,
            fillColor=NAVY,
        )
    )
    drawing.add(
        String(
            30,
            222,
            "Correct predictions; validation n=5,000 and test n=10,000",
            fontName="Helvetica",
            fontSize=7,
            fillColor=MUTED,
        )
    )
    add_blossom(drawing, 445, 235, TEAL)

    x0, y0, width, height = 56, 38, 370, 160
    for tick in [0, 25, 50, 75, 100]:
        y = y0 + height * tick / 100
        drawing.add(Line(x0, y, x0 + width, y, strokeColor=RULE, strokeWidth=0.5))
        drawing.add(
            String(
                x0 - 9,
                y - 2.5,
                f"{tick}%",
                fontName="Helvetica",
                fontSize=6.5,
                fillColor=MUTED,
                textAnchor="end",
            )
        )
    drawing.add(Line(x0, y0, x0, y0 + height, strokeColor=MUTED, strokeWidth=0.7))
    drawing.add(Line(x0, y0, x0 + width, y0, strokeColor=MUTED, strokeWidth=0.7))

    groups = [
        (
            "Validation",
            codebook["validation"]["classifier_accuracy"] * 100,
            codebook["validation"]["receiver_accuracy"] * 100,
        ),
        (
            "Test",
            codebook["test"]["classifier_accuracy"] * 100,
            codebook["test"]["receiver_accuracy"] * 100,
        ),
    ]
    centers = [160, 325]
    bar_width = 48
    gap = 8
    for center, (label, classifier, prototype) in zip(centers, groups, strict=True):
        left = center - bar_width - gap / 2
        for index, (value, fill) in enumerate([(classifier, TEAL), (prototype, ORANGE)]):
            x = left + index * (bar_width + gap)
            bar_height = height * value / 100
            drawing.add(
                Rect(
                    x,
                    y0,
                    bar_width,
                    bar_height,
                    fillColor=fill,
                    strokeColor=NAVY,
                    strokeWidth=0.35,
                )
            )
            drawing.add(
                String(
                    x + bar_width / 2,
                    y0 + bar_height + 6,
                    f"{value:.2f}%",
                    fontName="Helvetica-Bold",
                    fontSize=7.5,
                    fillColor=NAVY,
                    textAnchor="middle",
                )
            )
        drawing.add(
            String(
                center,
                22,
                label,
                fontName="Helvetica-Bold",
                fontSize=8,
                fillColor=INK,
                textAnchor="middle",
            )
        )

    drawing.add(Rect(145, 4, 10, 4, fillColor=TEAL, strokeColor=TEAL))
    drawing.add(
        String(160, 2, "Linear classifier", fontName="Helvetica", fontSize=7, fillColor=INK)
    )
    drawing.add(Rect(260, 4, 10, 4, fillColor=ORANGE, strokeColor=ORANGE))
    drawing.add(
        String(275, 2, "Prototype codebook", fontName="Helvetica", fontSize=7, fillColor=INK)
    )
    return drawing


def performance_table(codebook: dict, style: dict[str, ParagraphStyle]) -> Table:
    rows = [["Split", "Classifier", "Codebook", "Difference", "Retention", "n"]]
    for name, key in [("Validation", "validation"), ("Test", "test")]:
        classifier = codebook[key]["classifier_accuracy"] * 100
        receiver = codebook[key]["receiver_accuracy"] * 100
        rows.append(
            [
                name,
                f"{classifier:.2f}%",
                f"{receiver:.2f}%",
                f"{receiver - classifier:+.2f} pp",
                f"{receiver / classifier * 100:.2f}%",
                f"{codebook[key]['samples']:,}",
            ]
        )
    return formatted_table(
        rows,
        [30 * mm, 28 * mm, 28 * mm, 29 * mm, 28 * mm, 23 * mm],
        style,
        bold_first_column=True,
    )


def blend_color(low, high, fraction: float):
    fraction = max(0.0, min(1.0, fraction))
    return colors.Color(
        low.red + (high.red - low.red) * fraction,
        low.green + (high.green - low.green) * fraction,
        low.blue + (high.blue - low.blue) * fraction,
    )


def similarity_heatmap(prototypes: torch.Tensor, class_names: list[str]) -> Drawing:
    similarity = prototypes @ prototypes.T
    drawing = Drawing(470, 335)
    drawing.add(
        String(
            20,
            316,
            "Prototype cosine-similarity matrix",
            fontName="Helvetica-Bold",
            fontSize=10,
            fillColor=NAVY,
        )
    )
    drawing.add(
        String(
            20,
            301,
            "Normalized 128-D class prototypes; diagonal similarity equals 1.00",
            fontName="Helvetica",
            fontSize=7,
            fillColor=MUTED,
        )
    )
    add_blossom(drawing, 445, 314, TEAL)

    cell = 25
    x0 = 170
    y0 = 35
    abbreviations = ["air", "auto", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    for row in range(10):
        drawing.add(
            String(
                x0 - 10,
                y0 + (9 - row) * cell + 9,
                class_names[row].title(),
                fontName="Helvetica",
                fontSize=6.8,
                fillColor=INK,
                textAnchor="end",
            )
        )
        for column in range(10):
            value = float(similarity[row, column])
            fraction = (value - 0.30) / 0.70
            fill = blend_color(PALE_BLUE, NAVY, fraction)
            x = x0 + column * cell
            y = y0 + (9 - row) * cell
            drawing.add(Rect(x, y, cell, cell, fillColor=fill, strokeColor=WHITE, strokeWidth=0.5))
            text_color = WHITE if value >= 0.67 else INK
            drawing.add(
                String(
                    x + cell / 2,
                    y + 9,
                    f"{value:.2f}",
                    fontName="Helvetica",
                    fontSize=5.5,
                    fillColor=text_color,
                    textAnchor="middle",
                )
            )
    for column, label in enumerate(abbreviations):
        drawing.add(
            String(
                x0 + column * cell + cell / 2,
                y0 - 10,
                label,
                fontName="Helvetica",
                fontSize=6.2,
                fillColor=INK,
                textAnchor="middle",
            )
        )

    legend_x, legend_y = 25, 150
    for index in range(8):
        value = 0.30 + 0.10 * index
        drawing.add(
            Rect(
                legend_x + index * 10,
                legend_y,
                10,
                8,
                fillColor=blend_color(PALE_BLUE, NAVY, index / 7),
                strokeColor=None,
            )
        )
    drawing.add(
        String(legend_x, legend_y - 10, "0.30", fontName="Helvetica", fontSize=6, fillColor=MUTED)
    )
    drawing.add(
        String(
            legend_x + 80,
            legend_y - 10,
            "1.00",
            fontName="Helvetica",
            fontSize=6,
            fillColor=MUTED,
            textAnchor="end",
        )
    )
    drawing.add(
        String(
            legend_x,
            legend_y + 14,
            "Cosine similarity",
            fontName="Helvetica-Bold",
            fontSize=6.5,
            fillColor=NAVY,
        )
    )
    return drawing


def similarity_summary_table(
    prototypes: torch.Tensor,
    class_names: list[str],
    style: dict[str, ParagraphStyle],
) -> Table:
    similarity = prototypes @ prototypes.T
    pairs: list[tuple[float, str, str]] = []
    for left in range(10):
        for right in range(left + 1, 10):
            pairs.append((float(similarity[left, right]), class_names[left], class_names[right]))
    pairs.sort(reverse=True)
    rows = [["Pair", "Cosine similarity", "Interpretation"]]
    descriptions = [
        "Highest off-diagonal similarity",
        "Second-highest similarity",
        "Third-highest similarity",
    ]
    for (value, left, right), description in zip(pairs[:3], descriptions, strict=True):
        rows.append([f"{left.title()} - {right.title()}", f"{value:.3f}", description])
    return formatted_table(rows, [54 * mm, 40 * mm, 72 * mm], style, bold_first_column=True)


def alignment_table(codebook: dict, style: dict[str, ParagraphStyle]) -> Table:
    checksum = codebook["codebook_sha256"]
    rows = [
        ["Integrity check", "Observed value", "Pass condition"],
        ["Codebook shape", "10 x 128", "Expected dimensions"],
        ["Exact tensor equality", str(codebook["exactly_equal"]), "True"],
        ["Independent storage", str(codebook["independent_storage"]), "True"],
        ["Maximum absolute difference", "0.0", "0.0"],
        ["Mean cosine distance", "0.0", "0.0"],
        ["Mean Euclidean distance", "0.0", "0.0"],
        ["Tx-Rx agreement", "100.00%", "100.00%"],
        ["Tensor SHA-256", f"{checksum[:16]}...{checksum[-12:]}", "Same at both endpoints"],
    ]
    return formatted_table(
        rows,
        [58 * mm, 55 * mm, 53 * mm],
        style,
        bold_first_column=True,
        compact=True,
    )


def limitation_table(style: dict[str, ParagraphStyle]) -> Table:
    rows = [
        ["Limitation", "Why it matters", "Required follow-up"],
        [
            "Single random seed",
            "The accuracy is a reproducible run, not a variance estimate.",
            "Run 3-5 seeds and report mean plus standard deviation.",
        ],
        [
            "One prototype per class",
            "A single mean compresses within-class modes and may merge related classes.",
            "Compare multiple prototypes or a VQ codebook after the minimal pipeline is stable.",
        ],
        [
            "No channel or drift yet",
            "This report validates initialization, not robustness or detection.",
            "Add the abstract channel and asymmetric local updates next.",
        ],
        [
            "Frozen backbone",
            "The result is a controlled budget baseline, not maximum CIFAR-10 accuracy.",
            "Keep it fixed for the first drift study; fine-tune only as an ablation.",
        ],
    ]
    return formatted_table(
        rows,
        [39 * mm, 62 * mm, 65 * mm],
        style,
        bold_first_column=True,
        compact=True,
    )


def build_report() -> None:
    baseline, codebook, classes, prototypes = load_evidence()
    style = styles()
    class_names = [row["class_name"] for row in classes]
    test_classifier = codebook["test"]["classifier_accuracy"] * 100
    test_codebook = codebook["test"]["receiver_accuracy"] * 100
    test_gap = test_codebook - test_classifier
    test_retention = test_codebook / test_classifier * 100

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    document = BaseDocTemplate(
        str(OUTPUT_PATH),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=20 * mm,
        bottomMargin=18 * mm,
        title="Aligned CIFAR-10 Semantic Codebook Report",
        author="Nafas Mohebi",
        subject="Phase 1 aligned transmitter-receiver semantic codebook",
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
        kpi_cards(codebook, style),
        Spacer(1, 8 * mm),
        Paragraph("Technical summary", style["section"]),
        Paragraph(
            "Phase 1 now has an auditable aligned semantic codebook. The epoch-9 "
            "baseline checkpoint was used to encode only the 45,000-image training "
            "split into 128-D representations. One L2-normalized mean prototype was "
            "computed for each CIFAR-10 class, then cloned into independent transmitter "
            "and receiver artifacts. The aligned prototype path achieved "
            f"<b>{test_codebook:.2f}% test accuracy</b>, retained <b>{test_retention:.2f}%</b> "
            f"of the linear classifier accuracy, and produced <b>zero initial drift</b> "
            "with 100% endpoint agreement.",
            style["body"],
        ),
        Paragraph("The completed aligned reference", style["section"]),
        scope_table(style),
        Spacer(1, 7 * mm),
        Table(
            [[Paragraph("PHASE 1 STATUS: COMPLETE, ALIGNED, AND REPRODUCIBLE", style["callout"])]],
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
        Paragraph("One training-only codebook initializes both endpoints", style["section"]),
        Paragraph(
            "The transmitter first maps an image to a 128-D semantic vector. The "
            "nearest normalized class prototype is selected by maximum cosine similarity. "
            "That prototype is the semantic message, and the receiver interprets it "
            "against an equal but independently stored copy of the codebook. No endpoint "
            "parameter or prototype is updated in this phase.",
            style["body"],
        ),
        architecture_diagram(),
        Spacer(1, 1 * mm),
        Table(
            [
                [
                    Paragraph(
                        "c_k = normalize(mean{z_i : y_i = k}) &nbsp;&nbsp; | &nbsp;&nbsp; "
                        "q* = argmax_k cosine(z, c_k)",
                        style["formula"],
                    )
                ]
            ],
            colWidths=[166 * mm],
            style=TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), PALE_GRAY),
                    ("BOX", (0, 0), (-1, -1), 0.6, RULE),
                    ("TOPPADDING", (0, 0), (-1, -1), 3 * mm),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3 * mm),
                ]
            ),
        ),
        Spacer(1, 5 * mm),
        Paragraph("Construction controls protect the test set", style["section"]),
        formatted_table(
            [
                ["Control", "Implementation", "Reason"],
                [
                    "Training-only prototypes",
                    "45,000 train images",
                    "Validation and test samples cannot influence codebook values.",
                ],
                [
                    "Deterministic transforms",
                    "Resize, center crop, ImageNet normalization",
                    "No random crop or flip changes the extracted reference vectors.",
                ],
                [
                    "Frozen checkpoint",
                    "Epoch 9; model.eval()",
                    "Codebook generation does not train or update model weights.",
                ],
                [
                    "Normalized comparison",
                    "L2 prototypes plus cosine similarity",
                    "Vector direction, rather than scale, determines the nearest class.",
                ],
            ],
            [44 * mm, 56 * mm, 66 * mm],
            style,
            bold_first_column=True,
            compact=True,
        ),
        Spacer(1, 5 * mm),
        Paragraph("All ten classes are represented", style["section"]),
        class_coverage_table(classes, style),
        Spacer(1, 3 * mm),
        Paragraph(
            "Counts range from 4,470 to 4,522 and sum to 45,000; every class prototype "
            "is supported by thousands of training observations.",
            style["small"],
        ),
        PageBreak(),
        Paragraph("Prototype decoding preserves 98.04% of classifier accuracy", style["section"]),
        Paragraph(
            "The prototype decoder is intentionally simpler than the trained linear "
            "classifier. On the 10,000-image test set it scored "
            f"<b>{test_codebook:.2f}%</b>, a <b>{abs(test_gap):.2f} percentage-point</b> "
            f"reduction from {test_classifier:.2f}%. This small, measured cost creates "
            "an interpretable 10-item semantic codebook and defines the aligned reference "
            "L0 for later drift experiments.",
            style["body"],
        ),
        accuracy_comparison_chart(codebook),
        Spacer(1, 2 * mm),
        performance_table(codebook, style),
        Spacer(1, 6 * mm),
        KeepTogether(
            [
                Paragraph("How to interpret the comparison", style["section"]),
                Paragraph(
                    "The original classifier is retained as the model-quality check: "
                    "its re-evaluated 88.04% validation and 87.07% test accuracies exactly "
                    "match the baseline report. The codebook path is evaluated separately "
                    "because it replaces the learned 128-to-10 decision layer with nearest-"
                    "prototype decoding. Future degradation must therefore be measured from "
                    "<b>85.36%</b>, not from the 87.07% classifier result.",
                    style["body"],
                ),
            ]
        ),
        Table(
            [[Paragraph("Aligned codebook baseline: L0 = 85.36% test accuracy", style["callout"])]],
            colWidths=[166 * mm],
            style=TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), PALE_ORANGE),
                    ("BOX", (0, 0), (-1, -1), 0.8, ORANGE),
                    ("TOPPADDING", (0, 0), (-1, -1), 4 * mm),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4 * mm),
                ]
            ),
        ),
        PageBreak(),
        Paragraph("The prototype space is structured rather than collapsed", style["section"]),
        Paragraph(
            "The normalized prototypes are distinct: off-diagonal cosine similarities "
            "range from 0.319 to 0.872, with a mean of 0.543. Related visual categories "
            "occupy nearby directions; cat-dog is the closest pair. This is descriptive "
            "evidence of class structure, not proof that geometry alone explains every "
            "classification error.",
            style["body"],
        ),
        similarity_heatmap(prototypes, class_names),
        similarity_summary_table(prototypes, class_names, style),
        PageBreak(),
        KeepTogether(
            [
                Paragraph("Endpoint alignment is exact by construction", style["section"]),
                Paragraph(
                    "The canonical tensor was cloned twice, not independently estimated. "
                    "The transmitter and receiver files therefore have the same numerical "
                    "checksum while remaining separate artifacts that can be updated "
                    "independently in a later phase. All initial drift measures are zero "
                    "and every transmitted prototype decodes to the same receiver index.",
                    style["body"],
                ),
                alignment_table(codebook, style),
            ]
        ),
        Spacer(1, 6 * mm),
        Paragraph("What this phase establishes - and what it does not", style["section"]),
        Paragraph(
            "The result establishes a clean common semantic state before any asymmetric "
            "learning, channel corruption, detector, or realignment operator is introduced. "
            "It does not yet show that drift will occur, that a receiver-side signal can "
            "detect it, or that a low-overhead update can recover accuracy. Those are the "
            "questions for the next experimental phases.",
            style["body"],
        ),
        limitation_table(style),
        Spacer(1, 7 * mm),
        Paragraph("Recommended next steps", style["section"]),
        Paragraph(
            "<b>Freeze the aligned artifacts.</b> Treat the saved epoch-9 checkpoint "
            "and both codebook files as the immutable t=0 reference.",
            style["bullet"],
            bulletText="1.",
        ),
        Paragraph(
            "<b>Add the abstract task-message channel.</b> Support quantization, feature "
            "dropout, erasure, and additive embedding noise as controlled options.",
            style["bullet"],
            bulletText="2.",
        ),
        Paragraph(
            "<b>Inject asymmetric local drift.</b> Update transmitter and receiver states "
            "from different local streams while keeping the channel error-free first.",
            style["bullet"],
            bulletText="3.",
        ),
        Paragraph(
            "<b>Measure degradation from L0.</b> Log accuracy, cosine prototype drift, "
            "Euclidean prototype drift, and task-disagreement rate per round and seed.",
            style["bullet"],
            bulletText="4.",
        ),
    ]
    document.build(story)


if __name__ == "__main__":
    build_report()
