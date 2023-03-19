import requests
import unittest


class TestDentist(unittest.TestCase):

    def test_dentist_signup(self):
        response = requests.post('http://20.127.200.67:8080/dentist/signup', json={
            'username': 'testuser',
            'password': 'testpassword',
            'confirm_password': 'testpassword'
        })
        data = response.json()['result']
        self.assertEqual(data, 'success')

    def test_dentist_signup_invalid_password(self):
        response = requests.post('http://20.127.200.67:8080/dentist/signup', json={
            'username': 'testuser',
            'password': 'testpassword',
            'confirm_password': 'invalidpassword'
        })
        data = response.json()['result']
        self.assertEqual(data, 'invalid')

    def test_dentist_signin(self):
        response = requests.post('http://20.127.200.67:8080/dentist/signin', json={
            'username': 'testuser',
            'password': 'testpassword'
        })
        data = response.json()['result']
        self.assertEqual(data, 'success')

    def test_dentist_signin_invalid_username(self):
        response = requests.post('http://20.127.200.67:8080/dentist/signin', json={
            'username': 'invaliduser',
            'password': 'testpassword'
        })
        data = response.json()['result']
        self.assertEqual(data, 'invalid')

    def test_dentist_signin_invalid_password(self):
        response = requests.post('http://20.127.200.67:8080/dentist/signin', json={
            'username': 'testuser',
            'password': 'invalidpassword'
        })
        data = response.json()['result']
        self.assertEqual(data, 'fail')

      
