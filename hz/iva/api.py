import os
import threading
import time

from hz.sys.core import RWLock


class IVAAPI:
    def __init__(self, parent=None):
        self.parent = parent

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

    def input(self, prompt=None):
        self.parent.input(prompt)

    def output(self, text, voice=True):
        self.parent.output(text, voice)

    def do_action(self, action, *args):
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
        # return task(self, *args)

    def enable(self, *args):
        for arg in args:
            if arg == 'voice':
                self.parent.voice = True
        return False

    def disable(self, *args):
        for arg in args:
            if arg == 'voice':
                self.parent.voice = False
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
