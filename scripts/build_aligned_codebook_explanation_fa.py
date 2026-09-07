from __future__ import annotations

import json
from pathlib import Path

from build_baseline_explanation_fa import (
    FONT_BOLD,
    FONT_REGULAR,
    INK,
    MUTED,
    NAVY,
    ORANGE,
    PALE_BLUE,
    PALE_GRAY,
    PALE_ORANGE,
    RULE,
    TEAL,
    WHITE,
    BidiText,
    RTLBullet,
    body,
    callout,
    heading,
    register_fonts,
    visual_rtl,
)
from reportlab.graphics.shapes import Drawing, Line, Rect, String
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    PageBreak,
    PageTemplate,
    Spacer,
    Table,
    TableStyle,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = (
    PROJECT_ROOT / "outputs/cifar10-resnet18-pretrained-baseline/aligned_codebook/summary.json"
)
OUTPUT_PATH = PROJECT_ROOT / "output/pdf/semantic_drift_aligned_codebook_explanation_fa.pdf"


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
    canvas.drawString(18 * mm, page_height - 9 * mm, "ALIGNED CODEBOOK - FA")

    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.5)
    canvas.line(18 * mm, 13 * mm, page_width - 18 * mm, 13 * mm)
    canvas.setFillColor(MUTED)
    canvas.setFont(FONT_REGULAR, 7.5)
    canvas.drawRightString(
        page_width - 18 * mm,
        8.5 * mm,
        visual_rtl("نفس محبی - توضیح فارسی کدبوک هم تراز"),
    )
    canvas.drawString(18 * mm, 8.5 * mm, visual_rtl(f"صفحه {document.page}"))
    canvas.restoreState()


def title_panel() -> Table:
    title = BidiText(
        "توضیح فارسی کدبوک معنایی هم تراز",
        font_name=FONT_BOLD,
        font_size=22,
        leading=31,
        color=WHITE,
        bottom_padding=4,
    )
    subtitle = BidiText(
        "نسخه آموزشی و راست چین گزارش مرحله اول برای ارائه به استاد",
        font_size=10.5,
        leading=16,
        color=colors.HexColor("#D9EAF0"),
        bottom_padding=5,
    )
    author = BidiText(
        "دانشجو: نفس محبی | تاریخ: ۱ سپتامبر ۲۰۲۶",
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
        (f"{summary['test']['receiver_accuracy'] * 100:.2f}%", "دقت کدبوک روی آزمون", PALE_BLUE),
        (
            f"{summary['test']['transmitter_receiver_agreement'] * 100:.0f}%",
            "توافق فرستنده و گیرنده",
            PALE_ORANGE,
        ),
        (f"{summary['drift']['mean_cosine_distance']:.1f}", "رانش معنایی اولیه", PALE_GRAY),
    ]
    cells = []
    for value, label, _ in cards:
        cells.append(
            [
                BidiText(
                    value,
                    font_name="Helvetica-Bold",
                    font_size=21,
                    leading=25,
                    color=NAVY,
                    align="center",
                    direction="ltr",
                    bottom_padding=3,
                ),
                BidiText(
                    label,
                    font_name=FONT_BOLD,
                    font_size=7.8,
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


def rtl_table(rows: list[list[str]], widths: list[float], *, header: bool = True) -> Table:
    rendered_rows = []
    for row_index, row in enumerate(rows):
        rendered_row = []
        for value in row:
            rendered_row.append(
                BidiText(
                    value,
                    font_name=FONT_BOLD if row_index == 0 and header else FONT_REGULAR,
                    font_size=8.1,
                    leading=11.5,
                    color=WHITE if row_index == 0 and header else INK,
                    align="center" if row_index == 0 else "right",
                )
            )
        rendered_rows.append(rendered_row)
    table = Table(rendered_rows, colWidths=widths, repeatRows=1 if header else 0)
    commands = [
        ("GRID", (0, 0), (-1, -1), 0.45, RULE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 2.5 * mm),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2.5 * mm),
        ("TOPPADDING", (0, 0), (-1, -1), 2.0 * mm),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.0 * mm),
    ]
    if header:
        commands.extend(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, PALE_GRAY]),
            ]
        )
    else:
        commands.append(("ROWBACKGROUNDS", (0, 0), (-1, -1), [WHITE, PALE_GRAY]))
    table.setStyle(TableStyle(commands))
    return table


def architecture_diagram() -> Drawing:
    drawing = Drawing(470, 116)
    boxes = [
        (4, 43, 58, "Image", "224 x 224", PALE_BLUE),
        (77, 43, 69, "ResNet-18", "frozen", PALE_BLUE),
        (161, 43, 68, "Projection", "512 -> 128", PALE_ORANGE),
        (244, 43, 70, "Tx codebook", "10 x 128", PALE_BLUE),
        (329, 43, 55, "Message", "prototype", PALE_ORANGE),
        (399, 43, 67, "Rx codebook", "10 x 128", PALE_BLUE),
    ]
    for index, (x, y, width, title, subtitle, fill) in enumerate(boxes):
        drawing.add(Rect(x, y, width, 42, 6, 6, fillColor=fill, strokeColor=RULE))
        drawing.add(
            String(
                x + width / 2,
                y + 25,
                title,
                fontName="Helvetica-Bold",
                fontSize=7.5,
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
                fontSize=6.1,
                fillColor=MUTED,
                textAnchor="middle",
            )
        )
        if index < len(boxes) - 1:
            next_x = boxes[index + 1][0]
            y_mid = y + 21
            drawing.add(Line(x + width + 3, y_mid, next_x - 7, y_mid, strokeColor=TEAL))
            drawing.add(Line(next_x - 11, y_mid + 3, next_x - 7, y_mid, strokeColor=TEAL))
            drawing.add(Line(next_x - 11, y_mid - 3, next_x - 7, y_mid, strokeColor=TEAL))
    drawing.add(
        String(
            235,
            17,
            "cosine nearest-prototype decoding",
            fontName="Helvetica-Bold",
            fontSize=7.2,
            fillColor=TEAL,
            textAnchor="middle",
        )
    )
    return drawing


def accuracy_chart(summary: dict) -> Drawing:
    drawing = Drawing(470, 250)
    x0, y0 = 75, 42
    plot_height = 165
    plot_width = 350
    for value in [0, 25, 50, 75, 100]:
        y = y0 + plot_height * value / 100
        drawing.add(Line(x0, y, x0 + plot_width, y, strokeColor=RULE, strokeWidth=0.6))
        drawing.add(
            String(
                x0 - 10,
                y - 2,
                f"{value}%",
                fontName="Helvetica",
                fontSize=6.5,
                fillColor=MUTED,
                textAnchor="end",
            )
        )
    groups = [
        (
            "Validation",
            summary["validation"]["classifier_accuracy"] * 100,
            summary["validation"]["receiver_accuracy"] * 100,
        ),
        (
            "Test",
            summary["test"]["classifier_accuracy"] * 100,
            summary["test"]["receiver_accuracy"] * 100,
        ),
    ]
    for index, (label, classifier, codebook) in enumerate(groups):
        center = x0 + 105 + index * 175
        for offset, value, color in [(-28, classifier, TEAL), (18, codebook, ORANGE)]:
            height = plot_height * value / 100
            drawing.add(
                Rect(
                    center + offset,
                    y0,
                    38,
                    height,
                    fillColor=color,
                    strokeColor=NAVY,
                    strokeWidth=0.5,
                )
            )
            drawing.add(
                String(
                    center + offset + 19,
                    y0 + height + 7,
                    f"{value:.2f}%",
                    fontName="Helvetica-Bold",
                    fontSize=7,
                    fillColor=NAVY,
                    textAnchor="middle",
                )
            )
        drawing.add(
            String(
                center + 12,
                25,
                label,
                fontName="Helvetica",
                fontSize=8,
                fillColor=INK,
                textAnchor="middle",
            )
        )
    drawing.add(Rect(135, 224, 10, 5, fillColor=TEAL, strokeColor=TEAL))
    drawing.add(String(150, 221, "Classifier", fontName="Helvetica", fontSize=7.2, fillColor=INK))
    drawing.add(Rect(245, 224, 10, 5, fillColor=ORANGE, strokeColor=ORANGE))
    drawing.add(String(260, 221, "Codebook", fontName="Helvetica", fontSize=7.2, fillColor=INK))
    return drawing


def build_report() -> None:
    register_fonts()
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    document = BaseDocTemplate(
        str(OUTPUT_PATH),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=20 * mm,
        bottomMargin=18 * mm,
        title="Persian explanation of the aligned CIFAR-10 semantic codebook",
        author="Nafas Mohebi",
        subject="Right-to-left explanation for supervisor presentation",
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
        [PageTemplate(id="persian-codebook-report", frames=[frame], onPage=page_chrome)]
    )

    story = [
        title_panel(),
        Spacer(1, 7 * mm),
        kpi_cards(summary),
        Spacer(1, 7 * mm),
        heading("هدف این مرحله"),
        body(
            "در این مرحله یک فرهنگ معنایی مشترک برای فرستنده و گیرنده ساخته شده است. "
            "هدف این است که پیش از ایجاد هر نوع رانش، یک وضعیت اولیه سالم، قابل تکرار و "
            "قابل اندازه گیری داشته باشیم. این وضعیت در ادامه با نام نقطه مرجع ال صفر "
            "استفاده می شود."
        ),
        callout(
            "نتیجه اصلی: کدبوک روی مجموعه آزمون به دقت 85.36 درصد رسید، توافق فرستنده "
            "و گیرنده 100 درصد است و فاصله اولیه دو کدبوک صفر است."
        ),
        Spacer(1, 6 * mm),
        heading("کدبوک در این پروژه چیست؟", level=2),
        body(
            "کدبوک یک جدول از بردارهای معنایی است. برای هر یک از ده کلاس مجموعه "
            "سیفار ده، یک بردار 128 بعدی وجود دارد. هر بردار نماینده میانگین ویژگی های "
            "تصاویر آموزشی همان کلاس است و به آن پروتوتایپ کلاس گفته می شود."
        ),
        rtl_table(
            [
                ["مقدار", "ویژگی"],
                ["10 در 128", "ابعاد کدبوک"],
                ["10", "تعداد پروتوتایپ ها"],
                ["128", "تعداد ابعاد هر پروتوتایپ"],
                ["45,000 تصویر آموزشی", "داده ساخت کدبوک"],
                ["نزدیک ترین پروتوتایپ کسینوسی", "قانون تصمیم گیری"],
            ],
            [65 * mm, 101 * mm],
        ),
        PageBreak(),
        heading("روش ساخت کدبوک و مسیر پیام"),
        body(
            "تصویر ابتدا وارد مدل از پیش آموزش دیده رزنت هجده می شود. خروجی مدل با سر "
            "پروجکشن از 512 بعد به 128 بعد تبدیل می شود. سپس بردار تصویر با ده پروتوتایپ "
            "مقایسه می شود و نزدیک ترین پروتوتایپ به عنوان پیام معنایی انتخاب می شود."
        ),
        architecture_diagram(),
        heading("ساخت پروتوتایپ هر کلاس", level=2),
        body(
            "برای هر کلاس، تمام بردارهای 128 بعدی تصاویر آموزشی آن کلاس جمع آوری شده و "
            "میانگین آن ها محاسبه شده است. سپس بردار میانگین نرمال سازی شده تا طول آن "
            "برابر یک شود. در نتیجه جهت بردار، معیار اصلی مقایسه خواهد بود."
        ),
        BidiText(
            "c_k = normalize(mean{z_i : y_i = k})",
            font_name="Courier-Bold",
            font_size=10,
            leading=15,
            color=NAVY,
            align="center",
            direction="ltr",
            top_padding=4,
            bottom_padding=8,
        ),
        heading("کنترل هایی که از داده آزمون محافظت می کنند", level=2),
        RTLBullet("پروتوتایپ ها فقط با 45,000 تصویر آموزشی ساخته شده اند."),
        RTLBullet("داده های اعتبارسنجی و آزمون در ساخت کدبوک استفاده نشده اند."),
        RTLBullet("مدل در حالت ارزیابی بوده و وزن های آن تغییر نکرده اند."),
        RTLBullet("تبدیل تصاویر قطعی بوده و افزایش داده تصادفی اعمال نشده است."),
        Spacer(1, 5 * mm),
        heading("پوشش کلاس ها", level=2),
        body(
            "تعداد تصاویر آموزشی هر کلاس بین 4,470 و 4,522 است. بنابراین هیچ کلاسی "
            "حذف نشده و هر پروتوتایپ با استفاده از چند هزار نمونه ساخته شده است."
        ),
        PageBreak(),
        heading("مقایسه دقت طبقه بند و کدبوک"),
        body(
            "طبقه بند اصلی از یک لایه آموزش دیده برای تشخیص کلاس استفاده می کند. مسیر "
            "کدبوک به جای آن، نزدیک ترین پروتوتایپ را بر اساس شباهت کسینوسی انتخاب می کند."
        ),
        accuracy_chart(summary),
        rtl_table(
            [
                ["تعداد نمونه", "حفظ دقت", "اختلاف", "کدبوک", "طبقه بند", "مجموعه"],
                ["5,000", "98.32%", "1.48- واحد درصد", "86.56%", "88.04%", "اعتبارسنجی"],
                ["10,000", "98.04%", "1.71- واحد درصد", "85.36%", "87.07%", "آزمون"],
            ],
            [23 * mm, 25 * mm, 32 * mm, 27 * mm, 27 * mm, 32 * mm],
        ),
        Spacer(1, 6 * mm),
        heading("چرا دقت کدبوک کمی کمتر است؟", level=2),
        body(
            "طبقه بند، مرزهای تصمیم گیری را مستقیما در زمان آموزش یاد گرفته است؛ اما "
            "کدبوک از یک قانون ساده و قابل تفسیر یعنی نزدیک ترین میانگین کلاس استفاده "
            "می کند. بنابراین کاهش 1.71 واحد درصدی روی آزمون طبیعی و اندازه گیری شده است."
        ),
        callout(
            "این اختلاف به معنی رانش نیست. مقدار رسمی نقطه مرجع کدبوک برابر با دقت "
            "85.36 درصد روی مجموعه آزمون است.",
            color=PALE_ORANGE,
        ),
        PageBreak(),
        heading("ساختار معنایی پروتوتایپ ها"),
        body(
            "ماتریس شباهت گزارش انگلیسی نشان می دهد هر پروتوتایپ با پروتوتایپ های "
            "دیگر چه مقدار هم جهت است. مقدار قطر اصلی برابر یک است، زیرا هر کلاس با "
            "خودش مقایسه شده است. میانگین شباهت کلاس های متفاوت 0.543 است."
        ),
        rtl_table(
            [
                ["تفسیر", "شباهت کسینوسی", "جفت کلاس"],
                ["بیشترین شباهت بین دو کلاس متفاوت", "0.872", "گربه و سگ"],
                ["دومین شباهت", "0.835", "گوزن و اسب"],
                ["سومین شباهت", "0.805", "هواپیما و کشتی"],
            ],
            [79 * mm, 40 * mm, 47 * mm],
        ),
        Spacer(1, 7 * mm),
        heading("فاصله کسینوسی و فاصله اقلیدسی", level=2),
        body(
            "فاصله کسینوسی تفاوت جهت دو بردار را اندازه می گیرد. اگر دو بردار کاملا "
            "هم جهت باشند، فاصله کسینوسی آن ها صفر است. فاصله اقلیدسی فاصله مستقیم "
            "میان مقادیر عددی دو بردار را اندازه می گیرد."
        ),
        rtl_table(
            [
                ["در صورت برابری کامل", "پرسش اصلی", "معیار"],
                ["صفر", "آیا جهت معنایی یکسان است؟", "فاصله کسینوسی"],
                ["صفر", "مقادیر دقیق چقدر فاصله دارند؟", "فاصله اقلیدسی"],
            ],
            [38 * mm, 79 * mm, 49 * mm],
        ),
        Spacer(1, 7 * mm),
        callout(
            "شباهت زیاد گربه و سگ به معنی رانش نیست؛ این عدد فقط نزدیکی دو کلاس متفاوت "
            "در فضای معنایی را نشان می دهد. رانش با مقایسه نسخه فرستنده و گیرنده سنجیده می شود."
        ),
        PageBreak(),
        heading("چرا هنوز هیچ رانشی نداریم؟"),
        body(
            "پس از ساخت کدبوک اصلی، همان آرایه عددی دو بار کپی و در دو فایل مستقل برای "
            "فرستنده و گیرنده ذخیره شده است. این دو فایل از نظر محل ذخیره مستقل هستند، "
            "اما تمام مقادیر آن ها در لحظه شروع دقیقا یکسان است."
        ),
        rtl_table(
            [
                ["نتیجه", "مقدار مشاهده شده", "بررسی"],
                ["درست", "10 در 128", "ابعاد کدبوک"],
                ["درست", "بله", "برابری دقیق آرایه ها"],
                ["درست", "بله", "ذخیره سازی مستقل"],
                ["بدون اختلاف", "0.0", "بیشترین اختلاف مطلق"],
                ["بدون رانش زاویه ای", "0.0", "میانگین فاصله کسینوسی"],
                ["بدون رانش عددی", "0.0", "میانگین فاصله اقلیدسی"],
                ["هم تراز کامل", "100%", "توافق فرستنده و گیرنده"],
            ],
            [49 * mm, 47 * mm, 70 * mm],
        ),
        Spacer(1, 6 * mm),
        callout(
            "نتیجه: وضعیت فعلی همان حالت قبل از رانش یا ال صفر است. تا این مرحله هیچ "
            "کدبوکی به صورت مستقل به روز نشده و هیچ نویز یا کانال مخابراتی اضافه نشده است."
        ),
        Spacer(1, 6 * mm),
        heading("این مرحله چه چیزی را اثبات نمی کند؟", level=2),
        RTLBullet("هنوز نشان نداده ایم که رانش در طول زمان ایجاد می شود."),
        RTLBullet("هنوز روش تشخیص رانش را آزمایش نکرده ایم."),
        RTLBullet("هنوز روش هم تراز کردن دوباره کدبوک ها را بررسی نکرده ایم."),
        RTLBullet("هنوز اثر نویز، حذف ویژگی یا کوانتیزه سازی پیام را اندازه نگرفته ایم."),
        PageBreak(),
        heading("مرحله بعدی پروژه"),
        RTLBullet(
            "فایل های فعلی مدل و دو کدبوک به عنوان وضعیت ثابت زمان صفر نگهداری شوند.",
            marker="۱",
        ),
        RTLBullet(
            "کانال پیام انتزاعی با گزینه هایی مانند نویز، حذف ویژگی و کوانتیزه سازی اضافه شود.",
            marker="۲",
        ),
        RTLBullet(
            "فرستنده و گیرنده با جریان های داده متفاوت به صورت مستقل به روز شوند "
            "تا رانش ایجاد شود.",
            marker="۳",
        ),
        RTLBullet(
            "در هر دور، دقت، فاصله کسینوسی، فاصله اقلیدسی و عدم توافق دو طرف ثبت شود.",
            marker="۴",
        ),
        Spacer(1, 7 * mm),
        heading("پرسش های پژوهشی مرحله بعد", level=2),
        RTLBullet("آیا فاصله کدبوک ها پیش از کاهش دقت افزایش پیدا می کند؟"),
        RTLBullet("کدام کلاس ها زودتر دچار رانش می شوند؟"),
        RTLBullet("آیا جفت کلاس های مشابه زودتر با یکدیگر اشتباه می شوند؟"),
        RTLBullet("روش هم ترازی مجدد چه مقدار از دقت از دست رفته را بازیابی می کند؟"),
        Spacer(1, 7 * mm),
        heading("متن کوتاه مناسب ارائه به استاد", level=2),
        callout(
            "در این مرحله برای هر کلاس سیفار ده یک پروتوتایپ 128 بعدی ساختم. کدبوک "
            "فقط با داده های آموزشی تولید و سپس به دو نسخه مستقل اما کاملا یکسان برای "
            "فرستنده و گیرنده تبدیل شد. دقت کدبوک روی آزمون 85.36 درصد، توافق دو طرف "
            "100 درصد و هر دو فاصله اولیه صفر است. بنابراین هنوز رانش نداریم و این "
            "وضعیت به عنوان نقطه مرجع ال صفر ثبت شده است."
        ),
        Spacer(1, 7 * mm),
        heading("اطلاعات بازتولید", level=2),
        body(
            f"کدبوک از وزن های دوره {summary['source_epoch']} و بذر تصادفی "
            f"{summary['seed']} ساخته شده است. شش آزمون خودکار و بررسی کیفیت کد پس از "
            "پیاده سازی با موفقیت اجرا شده اند."
        ),
        BidiText(
            f"SHA-256: {summary['codebook_sha256']}",
            font_name="Courier",
            font_size=6.7,
            leading=10,
            color=MUTED,
            align="center",
            direction="ltr",
        ),
    ]
    document.build(story)


if __name__ == "__main__":
    build_report()
