# -*- coding: utf-8 -*-
import time
import re, string
import os, os.path as op
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QMessageBox

import requests
import random
import json

api_server_url = 'http://60.19.58.86:20000/'
dev_dir = 'squad_v2/devset'


class Ui_QADemoGui(object):
    def setupUi(self, QADemoGui):
        # GUI Name & Default size
        QADemoGui.setObjectName("QADemoGui")
        QADemoGui.resize(1060, 860)
        # Static Variables
        self.api_server_url = api_server_url
        self.qa_token = 'qa_en'
        self.dataset_dir = dev_dir
        self.data_files = [op.join(self.dataset_dir, file) for file in os.listdir(self.dataset_dir)]
        self.NO_ANSWER_TAG = 'No Answer'
        self.EXACT_MATCH_TAG = 'Exact Match'
        self.NOT_EXACT_MATCH_TAG = 'Not Exact Match'
        self.context, self.question, self.predicted_answer, self.golden_answers = '', '', '', []
        self.time_cost = '0ms'
        self.check_result = ''
        self.total_samples = 0
        self.total_correct = 0
        self.accuracy = "0%"
        # Btns & Events
        self.centralwidget = QtWidgets.QWidget(QADemoGui)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.load_data_and_predict_btn = QtWidgets.QPushButton(self.centralwidget)
        self.load_data_and_predict_btn.setObjectName("load_data_and_predict_btn")
        self.load_data_and_predict_btn.clicked.connect(self.load_sample_and_predict)
        self.verticalLayout.addWidget(self.load_data_and_predict_btn)
        spacerItem = QtWidgets.QSpacerItem(20, 18, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)
        self.horizontalLayout_1 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_1.setObjectName("horizontalLayout_1")
        self.context_label = QtWidgets.QLabel(self.centralwidget)
        self.context_label.setAlignment(QtCore.Qt.AlignCenter)
        self.context_label.setObjectName("context_label")
        self.horizontalLayout_1.addWidget(self.context_label)
        self.context_textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.context_textEdit.setObjectName("context_textEdit")
        self.horizontalLayout_1.addWidget(self.context_textEdit)
        self.horizontalLayout_1.setStretch(0, 1)
        self.horizontalLayout_1.setStretch(1, 5)
        self.verticalLayout.addLayout(self.horizontalLayout_1)
        spacerItem1 = QtWidgets.QSpacerItem(20, 18, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem1)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.question_label = QtWidgets.QLabel(self.centralwidget)
        self.question_label.setAlignment(QtCore.Qt.AlignCenter)
        self.question_label.setObjectName("question_label")
        self.horizontalLayout_2.addWidget(self.question_label)
        self.question_textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.question_textEdit.setObjectName("question_textEdit")
        self.horizontalLayout_2.addWidget(self.question_textEdit)
        self.horizontalLayout_2.setStretch(0, 1)
        self.horizontalLayout_2.setStretch(1, 5)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        spacerItem2 = QtWidgets.QSpacerItem(20, 18, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem2)
        self.predict_from_manual_input_btn = QtWidgets.QPushButton(self.centralwidget)
        self.predict_from_manual_input_btn.setObjectName("predict_from_manual_input_btn")
        self.predict_from_manual_input_btn.clicked.connect(self.predict_from_manual_input)
        self.verticalLayout.addWidget(self.predict_from_manual_input_btn)
        spacerItem3 = QtWidgets.QSpacerItem(20, 18, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem3)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.predicted_answer_label = QtWidgets.QLabel(self.centralwidget)
        self.predicted_answer_label.setAlignment(QtCore.Qt.AlignCenter)
        self.predicted_answer_label.setObjectName("predicted_answer_label")
        self.horizontalLayout_3.addWidget(self.predicted_answer_label)
        self.predicted_answer_textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.predicted_answer_textEdit.setObjectName("predicted_answer_textEdit")
        self.horizontalLayout_3.addWidget(self.predicted_answer_textEdit)
        self.horizontalLayout_3.setStretch(0, 1)
        self.horizontalLayout_3.setStretch(1, 5)
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        spacerItem4 = QtWidgets.QSpacerItem(20, 18, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem4)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.time_cost_label = QtWidgets.QLabel(self.centralwidget)
        self.time_cost_label.setAlignment(QtCore.Qt.AlignCenter)
        self.time_cost_label.setObjectName("time_cost_label")
        self.horizontalLayout_4.addWidget(self.time_cost_label)
        self.time_cost_lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.time_cost_lineEdit.setObjectName("time_cost_lineEdit")
        self.horizontalLayout_4.addWidget(self.time_cost_lineEdit)
        spacerItem5 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem5)
        self.answer_check_label = QtWidgets.QLabel(self.centralwidget)
        self.answer_check_label.setAlignment(QtCore.Qt.AlignCenter)
        self.answer_check_label.setObjectName("answer_check_label")
        self.horizontalLayout_4.addWidget(self.answer_check_label)
        self.answer_check_lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.answer_check_lineEdit.setObjectName("answer_check_lineEdit")
        self.horizontalLayout_4.addWidget(self.answer_check_lineEdit)
        spacerItem6 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem6)
        self.predicted_samples_label = QtWidgets.QLabel(self.centralwidget)
        self.predicted_samples_label.setAlignment(QtCore.Qt.AlignCenter)
        self.predicted_samples_label.setObjectName("predicted_samples_label")
        self.horizontalLayout_4.addWidget(self.predicted_samples_label)
        self.predicted_samples_lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.predicted_samples_lineEdit.setObjectName("predicted_samples_lineEdit")
        self.horizontalLayout_4.addWidget(self.predicted_samples_lineEdit)
        spacerItem7 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem7)
        self.accuracy_label = QtWidgets.QLabel(self.centralwidget)
        self.accuracy_label.setAlignment(QtCore.Qt.AlignCenter)
        self.accuracy_label.setObjectName("accuracy_label")
        self.horizontalLayout_4.addWidget(self.accuracy_label)
        self.accuracy_lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.accuracy_lineEdit.setObjectName("accuracy_lineEdit")
        self.horizontalLayout_4.addWidget(self.accuracy_lineEdit)
        self.horizontalLayout_4.setStretch(0, 4)
        self.horizontalLayout_4.setStretch(1, 2)
        # self.horizontalLayout_4.setStretch(2, 1)
        self.horizontalLayout_4.setStretch(3, 3)
        self.horizontalLayout_4.setStretch(4, 3)
        # self.horizontalLayout_4.setStretch(5, 1)
        self.horizontalLayout_4.setStretch(6, 3)
        self.horizontalLayout_4.setStretch(7, 1)
        # self.horizontalLayout_4.setStretch(8, 1)
        self.horizontalLayout_4.setStretch(9, 2)
        self.horizontalLayout_4.setStretch(10, 2)
        self.verticalLayout.addLayout(self.horizontalLayout_4)
        spacerItem8 = QtWidgets.QSpacerItem(20, 18, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem8)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.golden_answer_label = QtWidgets.QLabel(self.centralwidget)
        self.golden_answer_label.setAlignment(QtCore.Qt.AlignCenter)
        self.golden_answer_label.setObjectName("golden_answer_label")
        self.horizontalLayout_5.addWidget(self.golden_answer_label)
        self.golden_answer_textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.golden_answer_textEdit.setObjectName("golden_answer_textEdit")
        self.horizontalLayout_5.addWidget(self.golden_answer_textEdit)
        self.horizontalLayout_5.setStretch(0, 1)
        self.horizontalLayout_5.setStretch(1, 5)
        self.verticalLayout.addLayout(self.horizontalLayout_5)
        self.verticalLayout.setStretch(0, 1)
        self.verticalLayout.setStretch(2, 10)
        self.verticalLayout.setStretch(4, 2)
        self.verticalLayout.setStretch(6, 1)
        self.verticalLayout.setStretch(8, 2)
        self.verticalLayout.setStretch(10, 1)
        self.verticalLayout.setStretch(12, 2)
        self.gridLayout.addLayout(self.verticalLayout, 0, 0, 1, 1)
        QADemoGui.setCentralWidget(self.centralwidget)
        # self.statusbar = QtWidgets.QStatusBar(QADemoGui)
        # self.statusbar.setObjectName("statusbar")
        # QADemoGui.setStatusBar(self.statusbar)

        self.retranslateUi(QADemoGui)
        QtCore.QMetaObject.connectSlotsByName(QADemoGui)

    def retranslateUi(self, QADemoGui):
        _translate = QtCore.QCoreApplication.translate
        QADemoGui.setWindowTitle(_translate("QADemoGui", "QA Demo"))
        self.load_data_and_predict_btn.setText(_translate("QADemoGui", "Load Random Sample From Dataset And Predict"))
        self.context_label.setText(_translate("QADemoGui", "Context"))
        self.question_label.setText(_translate("QADemoGui", "Question"))
        self.predict_from_manual_input_btn.setText(_translate("QADemoGui", "Predict From Manual Input"))
        self.predicted_answer_label.setText(_translate("QADemoGui", "Predicted Answer"))
        self.time_cost_label.setText(_translate("QADemoGui", "Time Cost"))
        self.answer_check_label.setText(_translate("QADemoGui", "Answer Check"))
        self.predicted_samples_label.setText(_translate("QADemoGui", "Predicted Samples"))
        self.accuracy_label.setText(_translate("QADemoGui", "Accuracy"))
        self.golden_answer_label.setText(_translate("QADemoGui", "Golden Answer"))

    def load_sample_and_predict(self):
        self.cur_data_file = random.choice(self.data_files)
        self.qa_id = op.basename(self.cur_data_file).replace('.json', '')
        self.cur_sample_data = json.load(open(self.cur_data_file))
        self.cur_sample_data['qa_id'] = self.qa_id
        print(json.dumps(self.cur_sample_data, indent=4))
        self.context, self.question, self.golden_answers = (
            self.cur_sample_data['context'],
            self.cur_sample_data['question'],
            self.cur_sample_data['answer_text'],
        )
        self.context_textEdit.setText(self.context)
        self.question_textEdit.setText(self.question)
        self.golden_answers = list(set(self.golden_answers))
        text = self.NO_ANSWER_TAG if self.golden_answers == [''] else '\n'.join(self.golden_answers)
        self.golden_answer_textEdit.setText(text)
        self.predict_and_display_result()

    def predict_from_manual_input(self):
        self.context = self.context_textEdit.toPlainText()
        self.question = self.question_textEdit.toPlainText()
        text = self.predicted_answer_textEdit.toPlainText()
        self.predicted_answer = text if text != self.NO_ANSWER_TAG else ''
        text = self.golden_answer_textEdit.toPlainText()
        self.golden_answers = [line if line != self.NO_ANSWER_TAG else '' for line in text.split('\n')]
        self.predict_and_display_result()

    def predict_and_display_result(self):
        start_t = time.time()
        datas = {
            'token': self.qa_token,
            'context': self.context,
            'question': self.question,
        }
        try:
            r = requests.post(self.api_server_url, datas, timeout=5)
            r.encoding = 'utf-8'
            if r.text[:7] != 'Error: ':
                self.predicted_answer = r.text
            else:
                _, errorMsg = r.text[7:].split('\n', maxsplit=1)
                self.critical("数值异常", errorMsg)
                return
        except Exception as e:
            self.critical("API连接失败", str(e))
            return
        self.total_samples += 1
        text = self.NO_ANSWER_TAG if self.predicted_answer == '' else self.predicted_answer
        self.predicted_answer_textEdit.setText(text)
        self.time_cost = '{:.0f}ms'.format((time.time() - start_t) * 1e3)
        self.time_cost_lineEdit.setText(self.time_cost)
        if self.answer_exact_match():
            self.total_correct += 1
            self.check_result = self.EXACT_MATCH_TAG
            self.answer_check_lineEdit.setStyleSheet("")
        else:
            self.check_result = self.NOT_EXACT_MATCH_TAG
            self.answer_check_lineEdit.setStyleSheet("color: red")
        self.answer_check_lineEdit.setText(self.check_result)
        self.predicted_samples_lineEdit.setText(str(self.total_samples))
        self.accuracy = "{:.2f}%".format(self.total_correct / self.total_samples * 1e2)
        self.accuracy_lineEdit.setText(self.accuracy)

    def answer_exact_match(self):
        normed_predicted_answer = self.normalize_answer(self.predicted_answer)
        normed_golden_answers = [self.normalize_answer(text) for text in self.golden_answers]
        return normed_predicted_answer in normed_golden_answers

    def normalize_answer(self, s):
        """Lower text and remove punctuation, articles and extra whitespace."""
        def remove_articles(text):
            regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
            return re.sub(regex, " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        return white_space_fix(remove_articles(remove_punc(s.lower())))

    def critical(self, errorType, errorMsg):
        QMessageBox.critical(self.centralwidget, errorType, errorMsg, QMessageBox.Ok, QMessageBox.Ok)
