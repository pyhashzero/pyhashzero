from __future__ import print_function

import argparse

from hz.idm.http.download import Download
from hz.iva.task import BaseTask


def download(url, file, chunks, resume):
    d = Download(url, file, options={'interface': None, 'proxies': None, 'ipv6': None}, progress_notify=False)
    d.download(chunks, resume)
    return d


class Task(BaseTask):
    def __init__(self, *args):
        super(Task, self).__init__(*args)

        if len(args) > 0:
            parser = argparse.ArgumentParser()
            parser.add_argument('--fetch', dest='fetch', action='store_true')
            parser.set_defaults(fetch=False)

            namespace, args = parser.parse_known_args(args, self.args)

            self.args = namespace

    def run(self):
        download('file', 'filename', 10, False)

        if not self.event.is_set():
            self.event.set()
