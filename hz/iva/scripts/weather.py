from __future__ import print_function

import argparse
import json
import sys
from datetime import datetime

import requests

from hz.iva.task import BaseTask


class Task(BaseTask):
    def __init__(self, api):
        super(Task, self).__init__(api, False, 1, 100)

        parser = argparse.ArgumentParser()
        parser.add_argument('--current', type=str)
        parser.add_argument('--forecast', type=str)
        parser.add_argument('--search', type=str)
        parser.add_argument('--history', type=str)
        parser.add_argument('--astronomy', type=str)
        parser.add_argument('--timezone', type=str)
        parser.add_argument('--sports', type=str)

        namespace, _ = parser.parse_known_args(sys.argv)

        self.action = None
        if namespace.current:
            self.action = ['current', namespace.current]
        elif namespace.forecast:
            self.action = ['download', namespace.forecast]
        elif namespace.search:
            self.action = ['search', namespace.search]
        elif namespace.history:
            self.action = ['history', namespace.history]
        elif namespace.astronomy:
            self.action = ['astronomy', namespace.astronomy]
        elif namespace.timezone:
            self.action = ['timezone', namespace.timezone]
        elif namespace.sports:
            self.action = ['sports', namespace.sports]

    def run(self):
        api_key = json.loads(open('key.json', 'r').read())['weather']['key']
        base_url = 'http://api.weatherapi.com/v1/'

        if self.action[0] == 'current':
            resp = requests.get(base_url + 'current.json', params={'key': api_key, 'q': self.action[1]})
            print(resp.content)
        elif self.action[0] == 'forecast':
            resp = requests.get(base_url + 'forecast.json', params={'key': api_key, 'q': self.action[1], 'days': 3})
            print(resp.content)
        elif self.action[0] == 'search':
            resp = requests.get(base_url + 'search.json', params={'key': api_key, 'q': self.action[1]})
            print(resp.content)
        elif self.action[0] == 'history':
            resp = requests.get(base_url + 'history.json', params={'key': api_key, 'q': self.action[1], 'dt': datetime.now().isoformat()})
            print(resp.content)
        elif self.action[0] == 'astronomy':
            resp = requests.get(base_url + 'astronomy.json', params={'key': api_key, 'q': self.action[1], 'dt': datetime.now().isoformat()})
            print(resp.content)
        elif self.action[0] == 'timezone':
            resp = requests.get(base_url + 'timezone.json', params={'key': api_key, 'q': self.action[1]})
            print(resp.content)
        elif self.action[0] == 'sports':
            resp = requests.get(base_url + 'sports.json', params={'key': api_key, 'q': self.action[1]})
            print(resp.content)
        else:
            raise ValueError(f'arguments are not recognized')

        if not self.event.is_set():
            self.event.set()
