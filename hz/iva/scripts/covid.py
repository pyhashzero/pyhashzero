from __future__ import print_function

import argparse
import sys

import requests

from hz.iva.task import BaseTask


class Task(BaseTask):
    def __init__(self, api):
        super(Task, self).__init__(api, False, 1, 100)

        parser = argparse.ArgumentParser()
        parser.add_argument('--country', type=str)

        namespace, _ = parser.parse_known_args(sys.argv)

        self.action = None
        if namespace.country:
            self.action = ['country', namespace.country]

    def run(self):
        base_url = 'https://coronavirus-19-api.herokuapp.com'

        if self.action[0] == 'country':
            if self.action[1] == 'world':
                resp = requests.get(f'{base_url}/all')
            elif self.action[1] == 'all':
                resp = requests.get(f'{base_url}/countries')
            else:
                resp = requests.get(f'{base_url}/countries/{self.action[1]}')
            print(resp.json())

        if not self.event.is_set():
            self.event.set()
