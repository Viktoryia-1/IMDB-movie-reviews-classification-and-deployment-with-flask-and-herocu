import unittest
import os
from bs4 import BeautifulSoup
from flask.testing import FlaskClient
from app import say_hello, predict
from app import app
import warnings
warnings.filterwarnings("ignore")


class FlaskTest(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_start_page(self):
        resp = self.app.get('/')
        assert resp.status_code == 200

    def test_empty_input(self):
        resp = self.app.post('/predict', data=dict(message=''))
        self.assertEqual(resp.data, second=b'Feedback can\'t be empty string!', msg='Empty test')

    def test_chars(self):
        resp = self.app.post('/predict', data=dict(message='!+ <  '))
        self.assertEqual(resp.data, second=b'Feedback contains only chars!', msg='Char test')

    def test_nums(self):
        resp = self.app.post('/predict', data=dict(message='   4000 '))
        self.assertEqual(resp.data, second=b'Feedback contains only numbers!', msg='Num test')

    def test_result(self):
        resp = self.app.post('/predict', data=dict(message='It\'s best movie ever!'))
        soup = BeautifulSoup(resp.data, 'html.parser')
        substring = soup.find('h2', string='Positive Review').get_text()
        self.assertEqual(substring, second='Positive Review', msg='Result test')


if __name__ == '__main__':
    unittest.main()
