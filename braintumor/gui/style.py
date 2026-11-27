"""Visual theme: calm clinical-research palette, consistent typography."""

NAVY = "#0f2a43"
NAVY_LIGHT = "#173a5a"
TEAL = "#1b8a8f"
BG = "#f3f6f9"
CARD = "#ffffff"
BORDER = "#d5dde5"
TEXT = "#1b2733"
MUTED = "#5b6b78"

EVIDENCE_COLORS = {   # (foreground, background)
    "Not validated": ("#9b1c1c", "#fde8e8"),
    "Experimental / preliminary": ("#8a5a00", "#fff3cd"),
    "Validated on an internal held-out dataset": ("#0b5d4b", "#d6f1e8"),
    "Independently evaluated on external data": ("#0b5d1a", "#d6f1da"),
}


def evidence_style(level: str) -> str:
    fg, bg = EVIDENCE_COLORS.get(level, ("#37474f", "#e8eef2"))
    return f"background:{bg};color:{fg};border:1px solid {fg};border-radius:10px;padding:2px 10px;font-weight:600;"


QSS = f"""
* {{ font-family: 'Segoe UI', 'Helvetica Neue', Arial; font-size: 10pt; color: {TEXT}; }}
QMainWindow, QWidget#page {{ background: {BG}; }}
QWidget#sidebar {{ background: {NAVY}; }}
QLabel#brand {{ color: white; font-size: 13pt; font-weight: 700; padding: 18px 16px 2px 16px; }}
QLabel#brandSub {{ color: #9fb6cc; font-size: 8.5pt; padding: 0 16px 14px 16px; }}
QListWidget#nav {{ background: transparent; border: none; outline: none; }}
QListWidget#nav::item {{ color: #c9d8e6; padding: 11px 18px; border-left: 3px solid transparent; }}
QListWidget#nav::item:selected {{ background: {NAVY_LIGHT}; color: white; border-left: 3px solid {TEAL}; }}
QListWidget#nav::item:disabled {{ color: #5f7992; }}
QListWidget#nav::item:hover:!selected:!disabled {{ background: #14354f; }}
QFrame#card {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 8px; }}
QLabel#cardTitle {{ color: {MUTED}; font-size: 8.5pt; font-weight: 600; letter-spacing: 0.5px; }}
QLabel#h1 {{ font-size: 17pt; font-weight: 700; color: {NAVY}; }}
QLabel#muted {{ color: {MUTED}; }}
QLabel#disclaimer {{ background: #fdecea; color: #8e1c1c; border: 1px solid #e8b4b4; border-radius: 6px; padding: 6px 10px; font-size: 8.5pt; }}
QPushButton {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 6px; padding: 6px 14px; }}
QPushButton:hover {{ border-color: {TEAL}; }}
QPushButton:disabled {{ color: #9aa8b4; background: #eef2f5; }}
QPushButton#primary {{ background: {TEAL}; color: white; border: 1px solid {TEAL}; font-weight: 600; }}
QPushButton#primary:hover {{ background: #167479; }}
QPushButton#primary:disabled {{ background: #9fc9cb; border-color: #9fc9cb; color: white; }}
QPushButton#danger {{ color: #9b1c1c; }}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{ background: white; border: 1px solid {BORDER}; border-radius: 5px; padding: 4px 6px; }}
QGroupBox {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 8px; margin-top: 14px; padding: 12px 10px 8px 10px; font-weight: 600; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 6px; color: {NAVY}; }}
QTabWidget::pane {{ border: 1px solid {BORDER}; border-radius: 6px; background: {CARD}; top: -1px; }}
QTabBar::tab {{ background: #e6ecf1; padding: 7px 16px; border: 1px solid {BORDER}; border-bottom: none; border-top-left-radius: 6px; border-top-right-radius: 6px; }}
QTabBar::tab:selected {{ background: {CARD}; font-weight: 600; }}
QTableWidget, QTextBrowser, QListWidget {{ background: white; border: 1px solid {BORDER}; border-radius: 6px; gridline-color: #e5ebf0; }}
QHeaderView::section {{ background: #eef3f7; border: none; border-bottom: 1px solid {BORDER}; padding: 5px 8px; font-weight: 600; }}
QProgressBar {{ border: 1px solid {BORDER}; border-radius: 6px; background: white; text-align: center; height: 18px; }}
QProgressBar::chunk {{ background: {TEAL}; border-radius: 5px; }}
QSlider::groove:horizontal {{ height: 4px; background: #cfd9e2; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {TEAL}; width: 14px; margin: -6px 0; border-radius: 7px; }}
QStatusBar {{ background: white; border-top: 1px solid {BORDER}; color: {MUTED}; }}
QToolTip {{ background: {NAVY}; color: white; border: none; padding: 4px 8px; }}
"""
