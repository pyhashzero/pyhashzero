import math

import win32com.client
from win32com.client import constants

MSSAM = 'HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\MSSam'
MSMARY = 'HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\MSMary'
MSMIKE = 'HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\MSMike'

E_REG = {
    MSSAM: (137.89, 1.11),
    MSMARY: (156.63, 1.11),
    MSMIKE: (154.37, 1.11)
}


def to_utf8(value):
    return str(value).encode('utf-8')


def from_utf8(value):
    return value.decode('utf-8')


class Voice(object):
    def __init__(self, voice_id, name=None, languages=None, gender=None, age=None):
        self.id = voice_id
        self.name = name
        self.languages = languages
        self.gender = gender
        self.age = age

    def __str__(self):
        return """<Voice id=%(id)s name=%(name)s languages=%(languages)s gender=%(gender)s age=%(age)s>""" % self.__dict__


class UserIODriver:
    def __init__(self, tts_enabled, stt_enabled, recognizer, words):
        self.tts_enabled = tts_enabled
        self.stt_enabled = stt_enabled

        self.speaker = win32com.client.Dispatch("SAPI.SpVoice")
        self.speaker.EventInterests = 33790

        class SpeakerEvents(object):
            OnAudioLevel = self.on_audio_level
            OnBookmark = self.on_bookmark
            OnEndStream = self.on_end_stream
            OnEnginePrivate = self.on_engine_private
            OnPhoneme = self.on_phoneme
            OnSentence = self.on_sentence
            OnStartStream = self.on_start_stream
            OnViseme = self.on_viseme
            OnVoiceChange = self.on_voice_change
            OnWord = self.on_word

        self.speaker_event_handler = win32com.client.WithEvents(self.speaker, SpeakerEvents)
        self.looping = False
        self.speaking = False
        self.stopping = False
        self.rate_wpm = 200
        self.set_property('voice', self.get_property('voice'))

        if recognizer == 'SAPI.SpInProcRecognizer':
            self.listener = win32com.client.Dispatch("SAPI.SpInProcRecognizer")
            self.listener.AudioInputStream = win32com.client.Dispatch("SAPI.SpMMAudioIn")
            self.listener_base = win32com.client.getevents("SAPI.SpInProcRecoContext")
        elif recognizer == 'SAPI.SpSharedRecognizer':
            self.listener = win32com.client.Dispatch("SAPI.SpSharedRecognizer")
            self.listener_base = win32com.client.getevents("SAPI.SpSharedRecoContext")
        else:
            self.say("listener is not recognized")
            raise ValueError('listener is not recognized')

        class ListenerEvents(self.listener_base):
            OnRecognition = self.on_recognition

        self.context = self.listener.CreateRecoContext()
        self.grammar = self.context.CreateGrammar()

        if words:
            self.grammar.DictationSetState(0)
            self.words = self.grammar.Rules.Add("words", constants.SRATopLevel + constants.SRADynamic, 0)
            self.words.Clear()
            for word in words:
                self.words.InitialState.AddWordTransition(None, word)
            self.grammar.Rules.Commit()
            self.grammar.CmdSetRuleState("words", 1)  # wordsRule
            self.grammar.Rules.Commit()
        else:
            self.grammar.DictationSetState(1)

        self.listener_event_handler = ListenerEvents(self.context)
        self.say("Started successfully")

    def on_audio_level(self, *args):
        pass

    def on_bookmark(self, *args):
        pass

    def on_end_stream(self, *args):
        pass

    def on_engine_private(self, *args):
        pass

    def on_phoneme(self, *args):
        pass

    def on_sentence(self, *args):
        pass

    def on_start_stream(self, *args):
        pass

    def on_viseme(self, *args):
        pass

    def on_voice_change(self, *args):
        pass

    def on_word(self, *args):
        pass

    def on_recognition(self, _1, _2, _3, result):
        new_result = win32com.client.Dispatch(result)
        print("You said: ", new_result.PhraseInfo.GetText())

    def input(self, prompt=None):
        if self.stt_enabled:
            pass
        else:
            pass

    def output(self, text):
        if self.tts_enabled:
            pass
        else:
            pass

    def start_listening_events(self):
        pass

    def stop_listening_events(self):
        pass

    def start_speaking(self):
        pass

    def stop_speaking(self):
        pass

    def start_listening(self):
        pass

    def stop_listening(self):
        pass

    def voices(self):
        pass

    def destroy(self):
        self.speaker.EventInterests = 0

    def say(self, text):
        self.speaking = True
        # speak(x, 1) # separate thread
        # speak(x) # current thread
        # speak('', 3) # stop talking
        # self._tts.Speak(from_utf8(to_utf8(text)), 19)
        self.speaker.Speak(from_utf8(to_utf8(text)), 19)

    def stop(self):
        if not self.speaking:
            return
        self.stopping = True
        self.speaker.Speak('', 3)

    def _token_from_id(self, id_):
        tokens = self.speaker.GetVoices()
        for token in tokens:
            if token.Id == id_:
                return token
        for t in self.speaker.GetVoices():
            return t
        raise ValueError('unknown voice id %s', id_)

    def get_property(self, name):
        if name == 'voices':
            return [Voice(attr.Id, attr.GetDescription()) for attr in self.speaker.GetVoices()]
        elif name == 'voice':
            return self.speaker.Voice.Id
        elif name == 'rate':
            return self.rate_wpm
        elif name == 'volume':
            return self.speaker.Volume / 100.0
        else:
            raise KeyError('unknown property %s' % name)

    def set_property(self, name, value):
        if name == 'voice':
            token = self._token_from_id(value)
            self.speaker.Voice = token
            a, b = E_REG.get(value, E_REG[MSMARY])
            self.speaker.Rate = int(math.log(self.rate_wpm / a, b))
        elif name == 'rate':
            id_ = self.speaker.Voice.Id
            a, b = E_REG.get(id_, E_REG[MSMARY])
            try:
                self.speaker.Rate = int(math.log(value / a, b))
            except TypeError as e:
                raise ValueError(str(e))
            self.rate_wpm = value
        elif name == 'volume':
            try:
                self.speaker.Volume = int(round(value * 100, 2))
            except TypeError as e:
                raise ValueError(str(e))
        else:
            raise KeyError('unknown property %s' % name)
