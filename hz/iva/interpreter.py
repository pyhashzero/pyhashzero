import os
import tempfile
import threading
import time
from cmd import Cmd

import colorama
from colorama import Fore

from hz.sys.core import RWLock

PROMPT_CHAR = '~>'
HISTORY_FILENAME = tempfile.TemporaryFile('w+t')


class Interpreter(Cmd):
    first_reaction_text = ""
    first_reaction_text += Fore.BLUE + 'IVA\' sound is by default disabled.' + Fore.RESET
    first_reaction_text += "\n"
    first_reaction_text += Fore.BLUE + 'In order to let IVA talk out loud type: '
    first_reaction_text += Fore.RESET + Fore.RED + 'enable sound' + Fore.RESET
    first_reaction_text += "\n"
    first_reaction_text += Fore.BLUE + "Type 'help' for a list of available actions." + Fore.RESET
    first_reaction_text += "\n"

    prompt = first_reaction_text + Fore.RED + "{} What can I do for you?: ".format(PROMPT_CHAR) + Fore.RESET

    def __init__(self):
        super(Interpreter, self).__init__()

        self.voice = False

        self.use_rawinput = False
        self.io_lock = RWLock()

        self.lock = RWLock()
        self.tasks = {}
        self.jobs = {}
        self.services = {}

        self.event = threading.Event()
        self.main_thread = threading.Thread(target=self.clean)
        self.main_thread.start()

        # start services
        # start jobs
        # start tasks

    def input(self, prompt=None):
        with self.io_lock.r_locked():
            return input(prompt)

    def output(self, text, voice=True):
        with self.io_lock.w_locked():
            if self.voice and voice:
                print(text)

    def precmd(self, line):
        HISTORY_FILENAME.write(line + '\n')

        return 'action ' + line

    def postcmd(self, stop, line):
        self.prompt = Fore.RED + "{} What can I do for you?: ".format(PROMPT_CHAR) + Fore.RESET
        return stop

    def do_action(self, line):
        try:
            action, *args = line.split(' ')

            if action == 'enable':
                return self.enable(*args)
            if action == 'disable':
                return self.disable(*args)
            if action == 'help':
                return self.help()
            if action == 'exit':
                return self.exit()

            _module = __import__(f'H0.iva.scripts', fromlist=[f'{action}'])

            action_module = getattr(_module, action)
            action_module.sys.argv = [action] + list(args)

            # module might not have function named task
            task = getattr(action_module, 'Task')

            # function might not have any argument
            t = task(self)
            if hash(t) in self.tasks:
                raise ValueError(f'{hash(t)} already is in tasks you need to wait for the task to finish')
            self.tasks[hash(t)] = t
            return t.start()
        except Exception as ex:
            print(str(ex))
            return self.exit()

    def clean(self):
        while not self.event.is_set():
            with self.lock.r_locked():
                keys = [key for key in self.tasks.keys()]

            delete_keys = []
            for key in keys:
                task = self.tasks[key]
                if not task.running():
                    delete_keys.append(key)

            for key in delete_keys:
                with self.lock.w_locked():
                    del self.tasks[key]

            time.sleep(1)

    def enable(self, *args):
        for arg in args:
            if arg == 'voice':
                self.voice = True
        return False

    def disable(self, *args):
        for arg in args:
            if arg == 'voice':
                self.voice = False
        return False

    def help(self):
        for filename in [x for x in os.listdir('H0/iva/scripts') if os.path.isdir(f'H0/iva/script/{x}')]:
            print(filename)
            # module might not have function named do
            _module = __import__(f'H0.iva.scripts.{filename}', fromlist=[f'Task'])

            task = getattr(_module, 'Task')
            task.help()

        return False

    def exit(self):
        with self.lock.r_locked():
            keys = [key for key in self.tasks.keys()]

        for key in keys:
            task = self.tasks[key]
            task.stop()

        self.event.set()
        return True


def main():
    colorama.init()
    interpreter = Interpreter()
    interpreter.cmdloop()
