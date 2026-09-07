from __future__ import annotations

import csv
import json
from pathlib import Path

import arabic_reshaper
from bidi.algorithm import get_display
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.shapes import Drawing, Line, Rect, String
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    PageBreak,
    PageTemplate,
    Spacer,
    Table,
    TableStyle,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_DIRECTORY = PROJECT_ROOT / "outputs/cifar10-resnet18-pretrained-baseline"
OUTPUT_PATH = PROJECT_ROOT / "output/pdf/semantic_drift_baseline_explanation_fa.pdf"

FONT_REGULAR_PATH = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"
FONT_BOLD_PATH = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
FONT_REGULAR = "PersianRegular"
FONT_BOLD = "PersianBold"

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


def register_fonts() -> None:
    pdfmetrics.registerFont(TTFont(FONT_REGULAR, FONT_REGULAR_PATH))
    pdfmetrics.registerFont(TTFont(FONT_BOLD, FONT_BOLD_PATH))


def visual_rtl(text: str) -> str:
    reshaped = arabic_reshaper.reshape(text)
    return get_display(reshaped, base_dir="R")


def _line_width(text: str, font_name: str, font_size: float) -> float:
    return pdfmetrics.stringWidth(visual_rtl(text), font_name, font_size)


def wrap_logical_rtl(
    text: str,
    max_width: float,
    font_name: str,
    font_size: float,
) -> list[str]:
    paragraphs = text.split("\n")
    lines: list[str] = []
    for paragraph in paragraphs:
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            if _line_width(candidate, font_name, font_size) <= max_width:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines


class BidiText(Flowable):
    def __init__(
        self,
        text: str,
        *,
        font_name: str = FONT_REGULAR,
        font_size: float = 10,
        leading: float = 15,
        color: colors.Color = INK,
        align: str = "right",
        direction: str = "rtl",
        top_padding: float = 0,
        bottom_padding: float = 0,
    ) -> None:
        super().__init__()
        self.text = text
        self.font_name = font_name
        self.font_size = font_size
        self.leading = leading
        self.color = color
        self.align = align
        self.direction = direction
        self.top_padding = top_padding
        self.bottom_padding = bottom_padding
        self.lines: list[str] = []

    def wrap(self, available_width: float, available_height: float) -> tuple[float, float]:
        del available_height
        self.width = available_width
        if self.direction == "rtl":
            self.lines = wrap_logical_rtl(
                self.text,
                available_width,
                self.font_name,
                self.font_size,
            )
        else:
            self.lines = self.text.split("\n")
        self.height = (
            self.top_padding
            + len(self.lines) * self.leading
            + self.bottom_padding
        )
        return self.width, self.height

    def draw(self) -> None:
        canvas = self.canv
        canvas.saveState()
        canvas.setFont(self.font_name, self.font_size)
        canvas.setFillColor(self.color)
        y = self.height - self.top_padding - self.font_size
        for logical_line in self.lines:
            line = visual_rtl(logical_line) if self.direction == "rtl" else logical_line
            if self.align == "center":
                canvas.drawCentredString(self.width / 2, y, line)
            elif self.align == "left":
                canvas.drawString(0, y, line)
            else:
                canvas.drawRightString(self.width, y, line)
            y -= self.leading
        canvas.restoreState()


class RTLBullet(Flowable):
    def __init__(
        self,
        text: str,
        *,
        marker: str = "•",
        font_size: float = 9.6,
        leading: float = 15,
    ) -> None:
        super().__init__()
        self.text = text
        self.marker = marker
        self.font_size = font_size
        self.leading = leading
        self.lines: list[str] = []

    def wrap(self, available_width: float, available_height: float) -> tuple[float, float]:
        del available_height
        self.width = available_width
        text_width = available_width - 11 * mm
        self.lines = wrap_logical_rtl(
            self.text,
            text_width,
            FONT_REGULAR,
            self.font_size,
        )
        self.height = len(self.lines) * self.leading + 2
        return self.width, self.height

    def draw(self) -> None:
        canvas = self.canv
        canvas.saveState()
        canvas.setFillColor(INK)
        y = self.height - self.font_size
        canvas.setFont(FONT_BOLD, self.font_size)
        canvas.drawRightString(self.width, y, visual_rtl(self.marker))
        canvas.setFont(FONT_REGULAR, self.font_size)
        text_right = self.width - 8 * mm
        for logical_line in self.lines:
            canvas.drawRightString(text_right, y, visual_rtl(logical_line))
            y -= self.leading
        canvas.restoreState()


def heading(text: str, *, level: int = 1) -> BidiText:
    if level == 1:
        return BidiText(
            text,
            font_name=FONT_BOLD,
            font_size=16,
            leading=22,
            color=NAVY,
            top_padding=2,
            bottom_padding=5,
        )
    return BidiText(
        text,
        font_name=FONT_BOLD,
        font_size=12,
        leading=18,
        color=TEAL,
        top_padding=2,
        bottom_padding=3,
    )


def body(text: str) -> BidiText:
    return BidiText(
        text,
        font_size=9.8,
        leading=16,
        color=INK,
        bottom_padding=5,
    )


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


def page_chrome(canvas, document) -> None:
    page_width, page_height = A4
    canvas.saveState()
    canvas.setFillColor(NAVY)
    canvas.rect(0, page_height - 14 * mm, page_width, 14 * mm, fill=1, stroke=0)
    canvas.setFont(FONT_BOLD, 8)
    canvas.setFillColor(WHITE)
    canvas.drawRightString(
        page_width - 18 * mm,
        page_height - 9 * mm,
        visual_rtl("پروژه پژوهشی رانش معنایی"),
    )
    canvas.setFont("Helvetica", 7.5)
    canvas.drawString(18 * mm, page_height - 9 * mm, "BASELINE EXPLANATION - FA")

    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.5)
    canvas.line(18 * mm, 13 * mm, page_width - 18 * mm, 13 * mm)
    canvas.setFillColor(MUTED)
    canvas.setFont(FONT_REGULAR, 7.5)
    canvas.drawRightString(
        page_width - 18 * mm,
        8.5 * mm,
        visual_rtl("نفس محبی - گزارش توضیحی خط پایه"),
    )
    canvas.setFont(FONT_REGULAR, 7.5)
    canvas.drawString(
        18 * mm,
        8.5 * mm,
        visual_rtl(f"صفحه {document.page}"),
    )
    canvas.restoreState()


def title_panel() -> Table:
    title = BidiText(
        "توضیح کامل خط پایه تمیز CIFAR-10",
        font_name=FONT_BOLD,
        font_size=23,
        leading=31,
        color=WHITE,
        bottom_padding=4,
    )
    subtitle = BidiText(
        "گزارش فارسی و راست چین برای ارائه مرحله اول پروژه رانش معنایی",
        font_size=10.5,
        leading=16,
        color=colors.HexColor("#D9EAF0"),
        bottom_padding=5,
    )
    author = BidiText(
        "دانشجو: نفس محبی | تاریخ: 23 اوت 2026",
        font_size=9.5,
        leading=15,
        color=colors.HexColor("#D9EAF0"),
    )
    table = Table([[[title, subtitle, Spacer(1, 4 * mm), author]]], colWidths=[166 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), NAVY),
                ("LEFTPADDING", (0, 0), (-1, -1), 10 * mm),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10 * mm),
                ("TOPPADDING", (0, 0), (-1, -1), 9 * mm),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 9 * mm),
            ]
        )
    )
    return table


def kpi_cards(summary: dict) -> Table:
    cards = [
        (
            f"{summary['test_accuracy'] * 100:.2f}%",
            "دقت نهایی آزمایش",
            PALE_BLUE,
        ),
        (
            f"{summary['best_validation_accuracy'] * 100:.2f}%",
            "بهترین دقت اعتبارسنجی",
            PALE_ORANGE,
        ),
        (str(summary["best_epoch"]), "دوره انتخاب شده", PALE_GRAY),
    ]
    cells = []
    for value, label, _ in cards:
        cells.append(
            [
                BidiText(
                    value,
                    font_name="Helvetica-Bold",
                    font_size=22,
                    leading=26,
                    color=NAVY,
                    align="center",
                    direction="ltr",
                    bottom_padding=3,
                ),
                BidiText(
                    label,
                    font_name=FONT_BOLD,
                    font_size=8,
                    leading=12,
                    color=MUTED,
                    align="center",
                ),
            ]
        )
    table = Table([[cells[0], cells[1], cells[2]]], colWidths=[55.3 * mm] * 3)
    commands: list[tuple] = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4 * mm),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4 * mm),
        ("TOPPADDING", (0, 0), (-1, -1), 4 * mm),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4 * mm),
        ("BOX", (0, 0), (-1, -1), 0.7, RULE),
        ("INNERGRID", (0, 0), (-1, -1), 0.7, WHITE),
    ]
    for index, (_, _, background) in enumerate(cards):
        commands.append(("BACKGROUND", (index, 0), (index, 0), background))
    table.setStyle(TableStyle(commands))
    return table


def architecture_diagram() -> Drawing:
    drawing = Drawing(470, 128)
    boxes = [
        (5, 43, 80, 44, "CIFAR-10", "224 x 224"),
        (105, 43, 100, 44, "ResNet-18", "ImageNet - frozen"),
        (225, 43, 85, 44, "Projection", "512 -> 128"),
        (330, 43, 75, 44, "Classifier", "128 -> 10"),
        (425, 43, 40, 44, "Class", "label"),
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
            18,
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
            18,
            "trainable parameters",
            fontName="Helvetica-Bold",
            fontSize=7,
            fillColor=ORANGE,
            textAnchor="middle",
        )
    )
    return drawing


def accuracy_chart(metrics: list[dict[str, float]]) -> Drawing:
    drawing = Drawing(470, 225)
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
    drawing.add(Rect(145, 205, 10, 3, fillColor=TEAL, strokeColor=TEAL))
    drawing.add(String(160, 201, "Train accuracy", fontName="Helvetica", fontSize=8, fillColor=INK))
    drawing.add(Rect(275, 205, 10, 3, fillColor=ORANGE, strokeColor=ORANGE))
    drawing.add(
        String(290, 201, "Validation accuracy", fontName="Helvetica", fontSize=8, fillColor=INK)
    )
    return drawing


def config_table(config: dict) -> Table:
    training = config["training"]
    raw_rows = [
        ("بهینه ساز", "AdamW"),
        ("تعداد دوره ها", str(training["epochs"])),
        ("نرخ یادگیری اولیه", f"{training['learning_rate']:.4f}"),
        ("اندازه دسته", "128"),
        ("کاهش وزن", f"{training['weight_decay']:.4f}"),
        ("نرم سازی برچسب", "0.1"),
        ("زمان بندی", "Cosine annealing"),
        ("بذر تصادفی", str(config["seed"])),
        ("قانون انتخاب", "Best validation accuracy"),
        ("دستگاه", "Apple MPS"),
    ]
    rows = []
    for index in range(0, len(raw_rows), 2):
        first = raw_rows[index]
        second = raw_rows[index + 1]
        rows.append(
            [
                BidiText(first[1], font_size=8.3, leading=11, align="center"),
                BidiText(first[0], font_name=FONT_BOLD, font_size=8.3, leading=11),
                BidiText(second[1], font_size=8.3, leading=11, align="center"),
                BidiText(second[0], font_name=FONT_BOLD, font_size=8.3, leading=11),
            ]
        )
    header = [
        BidiText("مقدار", font_name=FONT_BOLD, font_size=8, leading=11, color=WHITE),
        BidiText("پارامتر", font_name=FONT_BOLD, font_size=8, leading=11, color=WHITE),
        BidiText("مقدار", font_name=FONT_BOLD, font_size=8, leading=11, color=WHITE),
        BidiText("پارامتر", font_name=FONT_BOLD, font_size=8, leading=11, color=WHITE),
    ]
    table = Table([header, *rows], colWidths=[45 * mm, 38 * mm, 45 * mm, 38 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, PALE_GRAY]),
                ("GRID", (0, 0), (-1, -1), 0.45, RULE),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 2.5 * mm),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2.5 * mm),
                ("TOPPADDING", (0, 0), (-1, -1), 2 * mm),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2 * mm),
            ]
        )
    )
    return table


def epoch_table(metrics: list[dict[str, float]]) -> Table:
    header_labels = [
        "Val accuracy",
        "Val loss",
        "Train accuracy",
        "Train loss",
        "Learning rate",
        "Epoch",
    ]
    header = [
        BidiText(
            label,
            font_name="Helvetica-Bold",
            font_size=7.2,
            leading=10,
            color=WHITE,
            align="center",
            direction="ltr",
        )
        for label in header_labels
    ]
    rows = []
    for row in metrics:
        values = [
            f"{row['validation_accuracy'] * 100:.2f}%",
            f"{row['validation_loss']:.4f}",
            f"{row['train_accuracy'] * 100:.2f}%",
            f"{row['train_loss']:.4f}",
            f"{row['learning_rate']:.6f}",
            str(row["epoch"]),
        ]
        rows.append(
            [
                BidiText(
                    value,
                    font_name="Helvetica",
                    font_size=7.6,
                    leading=10,
                    align="center",
                    direction="ltr",
                )
                for value in values
            ]
        )
    table = Table(
        [header, *rows],
        colWidths=[29 * mm, 25 * mm, 31 * mm, 25 * mm, 35 * mm, 21 * mm],
        repeatRows=1,
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, PALE_GRAY]),
                ("BACKGROUND", (0, 9), (-1, 9), PALE_ORANGE),
                ("GRID", (0, 0), (-1, -1), 0.4, RULE),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 1.5 * mm),
                ("RIGHTPADDING", (0, 0), (-1, -1), 1.5 * mm),
                ("TOPPADDING", (0, 0), (-1, -1), 1.4 * mm),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 1.4 * mm),
            ]
        )
    )
    return table


def callout(text: str, *, color: colors.Color = PALE_BLUE) -> Table:
    table = Table(
        [[BidiText(text, font_name=FONT_BOLD, font_size=10, leading=16, color=NAVY)]],
        colWidths=[166 * mm],
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), color),
                ("BOX", (0, 0), (-1, -1), 0.8, TEAL),
                ("LEFTPADDING", (0, 0), (-1, -1), 5 * mm),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5 * mm),
                ("TOPPADDING", (0, 0), (-1, -1), 4 * mm),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4 * mm),
            ]
        )
    )
    return table


def build_report() -> None:
    register_fonts()
    summary, config, metrics = load_run()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    document = BaseDocTemplate(
        str(OUTPUT_PATH),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=20 * mm,
        bottomMargin=18 * mm,
        title="Persian explanation of the clean CIFAR-10 baseline",
        author="Nafas Mohebi",
        subject="Right-to-left baseline explanation for supervisor review",
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
    document.addPageTemplates(
        [PageTemplate(id="persian-report", frames=[frame], onPage=page_chrome)]
    )

    story = [
        title_panel(),
        Spacer(1, 7 * mm),
        kpi_cards(summary),
        Spacer(1, 7 * mm),
        heading("هدف این گزارش"),
        body(
            "این فایل نتیجه مرحله اول پروژه، یعنی ساخت یک خط پایه تمیز و بدون "
            "رانش را توضیح می دهد. پیش از آن که فرستنده و گیرنده به صورت مستقل "
            "آموزش ببینند، باید بدانیم مدل در حالت سالم و کاملا هم تراز چه دقتی "
            "دارد. این عملکرد سالم، نقطه مرجع همه آزمایش های بعدی است."
        ),
        callout(
            "عدد مرجع رسمی پروژه برابر با دقت 87.07 درصد روی مجموعه آزمایش است. "
            "یعنی مدل از 10,000 تصویر آزمایش، تعداد 8,707 تصویر را درست طبقه بندی کرده است."
        ),
        Spacer(1, 6 * mm),
        heading("سه عدد اصلی صفحه اول", level=2),
        RTLBullet(
            "دقت نهایی آزمایش 87.07 درصد است. این عدد روی داده هایی محاسبه شده "
            "که برای آموزش یا انتخاب مدل استفاده نشده اند."
        ),
        RTLBullet(
            "بهترین دقت اعتبارسنجی 88.04 درصد است. از داده های اعتبارسنجی فقط "
            "برای انتخاب بهترین دوره آموزش استفاده شده است."
        ),
        RTLBullet(
            "بهترین وزن ها مربوط به دوره نهم هستند. دقت اعتبارسنجی در دوره دهم "
            "کاهش یافت، بنابراین وزن های دوره نهم حفظ شدند."
        ),
        PageBreak(),
        heading("داده ها و معماری مدل"),
        heading("تقسیم بندی دیتاست CIFAR-10", level=2),
        body(
            "داده ها به سه بخش مستقل تقسیم شدند: 45,000 تصویر برای آموزش، "
            "5,000 تصویر برای اعتبارسنجی و 10,000 تصویر برای آزمایش نهایی. "
            "این جداسازی مانع می شود که انتخاب مدل تحت تاثیر داده های آزمایش قرار گیرد."
        ),
        RTLBullet("45,000 تصویر آموزشی برای یادگیری وزن های سر مدل"),
        RTLBullet("5,000 تصویر اعتبارسنجی برای انتخاب بهترین دوره"),
        RTLBullet("10,000 تصویر آزمایش برای گزارش نتیجه نهایی"),
        Spacer(1, 4 * mm),
        heading("مدل از دو بخش اصلی تشکیل شده است", level=2),
        body(
            "بخش اول یک ResNet-18 با وزن های رسمی ImageNet است. این شبکه قبلا "
            "روی مجموعه بزرگی از تصاویر آموزش دیده و ویژگی های عمومی مانند لبه، "
            "بافت و شکل را یاد گرفته است. بخش دوم یک سر کوچک مخصوص ده کلاس CIFAR-10 است."
        ),
        architecture_diagram(),
        heading("ساختار سر قابل آموزش", level=2),
        callout("بردار 512 بعدی به 128 بعد تبدیل می شود و سپس طبقه بند 10 کلاسه قرار می گیرد."),
        Spacer(1, 4 * mm),
        heading("چرا بخش اصلی مدل فریز شد؟", level=2),
        RTLBullet("آموزش سریع تر و کم هزینه تر می شود."),
        RTLBullet("ویژگی های از پیش آموخته شده ImageNet تخریب نمی شوند."),
        RTLBullet("مدل اولیه برای فرستنده و گیرنده پایدار و یکسان باقی می ماند."),
        RTLBullet("فقط لایه تبدیل 512 به 128 و طبقه بند نهایی آموزش می بینند."),
        PageBreak(),
        heading("آماده سازی ورودی و روند آموزش"),
        heading("پردازش تصاویر", level=2),
        body(
            "تصاویر اصلی CIFAR-10 فقط 32 در 32 پیکسل هستند، اما ResNet-18 از پیش "
            "آموزش دیده برای ورودی ImageNet طراحی شده است. تصاویر به اندازه 224 در 224 "
            "تبدیل و با میانگین و انحراف معیار ImageNet نرمال سازی شدند."
        ),
        RTLBullet("روی داده های آموزش، برش تصادفی و چرخش افقی اعمال شد."),
        RTLBullet(
            "روی داده های اعتبارسنجی و آزمایش تغییر تصادفی انجام نشد تا ارزیابی ثابت باشد."
        ),
        Spacer(1, 4 * mm),
        heading("نمودار روند دقت", level=2),
        accuracy_chart(metrics),
        body(
            "خط سبز دقت آموزش و خط نارنجی دقت اعتبارسنجی را نشان می دهد. دقت "
            "اعتبارسنجی از 85.54 درصد در دوره اول به بیشترین مقدار 88.04 درصد در "
            "دوره نهم رسید. در دوره دهم دقت آموزش کمی بهتر شد، اما دقت اعتبارسنجی "
            "به 87.76 درصد کاهش یافت."
        ),
        callout(
            "در دوره نهم فاصله دقت آموزش و اعتبارسنجی فقط 1.35 واحد درصد بود؛ "
            "بنابراین نشانه ای از بیش برازش شدید دیده نمی شود."
        ),
        PageBreak(),
        heading("تنظیمات دقیق آموزش"),
        config_table(config),
        Spacer(1, 7 * mm),
        heading("معنی پارامترهای اصلی", level=2),
        RTLBullet(
            "AdamW بهینه سازی است که وزن های سر مدل را برای کاهش خطا به روز می کند."
        ),
        RTLBullet(
            "نرخ یادگیری اولیه 0.001 تعیین می کند وزن ها در هر مرحله چه اندازه تغییر کنند."
        ),
        RTLBullet(
            "اندازه دسته 128 یعنی در هر مرحله 128 تصویر پردازش و سپس گرادیان محاسبه شده است."
        ),
        RTLBullet(
            "کاهش وزن 0.0005 نوعی منظم سازی است و احتمال بیش برازش را کم می کند."
        ),
        RTLBullet(
            "نرم سازی برچسب 0.1 از اعتماد بیش از حد مدل جلوگیری می کند و تعمیم را بهتر می سازد."
        ),
        RTLBullet(
            "زمان بندی کسینوسی نرخ یادگیری را از 0.001 به حدود 0.000024 کاهش داد."
        ),
        Spacer(1, 5 * mm),
        heading("بازتولیدپذیری و زمان اجرا", level=2),
        body(
            "بذر تصادفی روی 42 ثابت شد تا تقسیم داده ها و عملیات تصادفی تا حد "
            "ممکن قابل تکرار باشند. اجرای کامل روی Apple MPS انجام شد و 1,066.14 "
            "ثانیه، معادل تقریبا 17 دقیقه و 46 ثانیه، طول کشید."
        ),
        PageBreak(),
        heading("شواهد دوره به دوره"),
        epoch_table(metrics),
        Spacer(1, 6 * mm),
        heading("تفسیر جدول", level=2),
        body(
            "خطای آموزش از 0.9858 به 0.7861 کاهش یافته است. خطای اعتبارسنجی نیز "
            "در دوره نهم به کمترین مقدار 0.8139 رسید. در همان دوره بیشترین دقت "
            "اعتبارسنجی، یعنی 88.04 درصد، ثبت شد. بنابراین انتخاب دوره نهم هم بر "
            "اساس دقت و هم بر اساس خطا منطقی است."
        ),
        heading("تفاوت دقت و خطا", level=2),
        body(
            "دقت فقط بررسی می کند پاسخ نهایی درست بوده یا غلط. خطا علاوه بر درستی "
            "پاسخ، میزان اطمینان مدل را نیز در نظر می گیرد. کاهش خطا همراه با افزایش "
            "دقت نشان می دهد فرآیند یادگیری به صورت پایدار پیش رفته است."
        ),
        callout(
            "اختلاف بهترین دقت اعتبارسنجی و دقت آزمایش فقط 0.97 واحد درصد است و "
            "برای دو مجموعه مستقل اختلافی طبیعی محسوب می شود.",
            color=PALE_ORANGE,
        ),
        PageBreak(),
        heading("کاربرد خط پایه در ادامه پروژه"),
        body(
            "این وزن ها نقطه شروع مشترک فرستنده و گیرنده خواهند بود. هر دو سمت در "
            "ابتدا از یک مدل یکسان استفاده می کنند و نمایش های آن ها هم تراز هستند. "
            "سپس دو سمت روی جریان های داده محلی متفاوت به صورت مستقل آموزش می بینند."
        ),
        RTLBullet("هر دو سمت از checkpoint دوره نهم شروع می کنند.", marker="1"),
        RTLBullet("آموزش مستقل باعث فاصله گرفتن نمایش های معنایی می شود.", marker="2"),
        RTLBullet("افت دقت نسبت به مقدار مرجع 87.07 درصد اندازه گیری می شود.", marker="3"),
        RTLBullet("پس از هم ترازی مجدد، میزان بازیابی دقت محاسبه می شود.", marker="4"),
        Spacer(1, 5 * mm),
        heading("مثال عددی", level=2),
        body(
            "اگر رانش دقت را از 87.07 درصد به 70 درصد کاهش دهد و هم ترازی مجدد "
            "آن را به 83 درصد برساند، افت اولیه 17.07 واحد درصد و بازیابی 13 واحد "
            "درصد خواهد بود. در این مثال حدود 76 درصد از دقت از دست رفته بازیابی شده است."
        ),
        heading("محدودیت فعلی", level=2),
        RTLBullet(
            "نتیجه فعلی فقط با یک بذر تصادفی اجرا شده و برای ادعای علمی نهایی کافی نیست."
        ),
        RTLBullet(
            "در نسخه نهایی باید آزمایش با حداقل سه تا پنج بذر تکرار و میانگین و "
            "انحراف معیار گزارش شود."
        ),
        RTLBullet(
            "بخش اصلی ResNet-18 فریز بوده است؛ بنابراین این نتیجه بیشترین دقت ممکن "
            "با آموزش کامل شبکه نیست."
        ),
        Spacer(1, 5 * mm),
        heading("خلاصه پیشنهادی برای ارائه به استاد", level=2),
        callout(
            "در مرحله اول یک مدل مرجع تمیز برای CIFAR-10 ساختم. از ResNet-18 با "
            "وزن های ImageNet به عنوان بخش اصلی فریز شده استفاده کردم و یک سر کوچک "
            "ده کلاسه روی آن آموزش دادم. بهترین دقت اعتبارسنجی در دوره نهم 88.04 "
            "درصد و دقت نهایی آزمایش 87.07 درصد شد. این مدل از این پس نقطه شروع "
            "هم تراز فرستنده و گیرنده و مرجع اندازه گیری رانش و بازیابی خواهد بود."
        ),
        Spacer(1, 5 * mm),
        body(
            "فایل تنظیم اجرا در configs/baseline/cifar10.yaml و نتایج خام، بهترین "
            "وزن ها و خلاصه ماشینی در outputs/cifar10-resnet18-pretrained-baseline "
            "ذخیره شده اند. همه پنج آزمون نرم افزاری و بررسی کیفیت کد پس از آموزش پاس شدند."
        ),
    ]
    document.build(story)


if __name__ == "__main__":
    build_report()
