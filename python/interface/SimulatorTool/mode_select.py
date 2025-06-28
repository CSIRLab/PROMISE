# mode_select.py 
# Author:      Emilie Ye
# Date:        2025-06-27
# Version:     0.1
# Description: Presents clickable cards for choosing between Memory, Compute in Memory, and Demo modes.
#              Enables the “Next” button once a selection is made, so the simulator can launch in the chosen mode.
# Copyright (c) 2025
from PySide6.QtWidgets import (
    QDialog, QHBoxLayout, QVBoxLayout, QLabel,
    QPushButton, QFrame, QSizePolicy
)
from PySide6.QtCore import Qt, Signal


class ModeCard(QFrame):
    clicked = Signal(str) #allow the card to emit a signal when clicked

    def __init__(self, mode_name: str, parent=None):
        super().__init__(parent)
        self.mode_name = mode_name
        self.setObjectName("modeCard")
        self.setFrameShape(QFrame.StyledPanel)
        self.setLineWidth(2)
        self.setStyleSheet("""
        QFrame#modeCard {
            border: 2px solid #DDD;
            border-radius: 8px;
            background-color: #FAFAFA;
        }
        QFrame#modeCard[selected="true"] {
            border: 2px solid #0078D7;
            background-color: #EFF7FF;
        }
        """)
        self.setProperty("selected", False)

        v = QVBoxLayout(self)
        v.setContentsMargins(20, 20, 20, 20)
        v.addStretch()
        lbl = QLabel(self.mode_name, alignment=Qt.AlignCenter)
        lbl.setStyleSheet("font-size: 14pt; font-weight: bold;")
        v.addWidget(lbl)
        v.addStretch()

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.setMinimumHeight(150)

    def mousePressEvent(self, ev):  #emit signal when clicked
        self.clicked.emit(self.mode_name)

    # Sets the selected state of the card
    def setSelected(self, yes: bool):
        self.setProperty("selected", yes) #the property "selected" is defined in the stylesheet
        self.style().unpolish(self)
        self.style().polish(self)


class ModeSelectDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Mode")
        self.setModal(True)
        self.choice = None  # Store the selected mode and deliver it to the main window

        main = QVBoxLayout(self)
        row = QHBoxLayout()
        row.setSpacing(20)
        row.setContentsMargins(20,20,20,20)

        self.cards = []
        for mode in ("Memory", "Compute in Memory", "Demo"):
            card = ModeCard(mode)
            card.clicked.connect(self.onCardClicked) #keep track of which card was clicked
            row.addWidget(card)
            self.cards.append(card)

        main.addLayout(row)


        # Next Button
        btn = QPushButton("Next")
        btn.setEnabled(False)
        btn.clicked.connect(self.accept) #emit the accept signal when clicked and deliver the choice to the main window
        btn_style = """ 
        QPushButton {
            background-color: #0078D7;
            color: white;
            border: none;
            padding: 8px 20px;
            font-weight: bold;
            font-size: 12pt;
            border-radius: 4px;
        }
        QPushButton:hover {
            background-color: #2893F3;
        }
        QPushButton:pressed {
            background-color: #005EA6;
        }
        """
        btn.setStyleSheet(btn_style)

        main.addWidget(btn, alignment=Qt.AlignCenter)
        self.ok_button = btn

        self.setLayout(main)

    def onCardClicked(self, mode_name):
        for c in self.cards:
            c.setSelected(c.mode_name == mode_name)
        self.choice = mode_name
        self.ok_button.setEnabled(True)