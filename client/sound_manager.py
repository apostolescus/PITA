from time import time, sleep
from threading import Thread
from playsound import playsound

# minimum inteval to play
# custom voice alert

ALERT_SEC_INTERVAL = 4


class SoundManager:
    """
    Class used as an interface for the sound management.
    Plays alert sound at frontal collision
    and custom voices for other types.
    """

    def __init__(self):
        self._last_time = 0
        self._alert_list = [
            "priority",
            "keep-right",
            "stop",
            "curve-right",
            "parking",
            "no-entry",
            "pedestrians",
            "red",
            "curve-left",
            "give-way",
        ]

    def _play_sound(self):
        playsound("alert_sounds/beep.mp3")

    def play_sound(self):
        Thread(target=self._play_sound).start()

    def _play_sound_custom_notification(self, notification):
        playsound("alert_sounds/voices/" + notification + ".mp3")

    def play_sound_custom_notification(self, notification):
        """
        Start new thread to play custom alert sound.
        Plays alert only once ALERT_SEC_INTERVAL.
        """

        if notification not in self._alert_list:
            return

        if time() - self._last_time > ALERT_SEC_INTERVAL:
            Thread(
                target=self._play_sound_custom_notification, args=(notification,)
            ).start()

            self._last_time = time()


def test_sound_manager():

    sound_manager = SoundManager()

    sound_manager.play_sound_custom_notification("curve-left")
    sound_manager.play_sound_custom_notification("curve-left")
    sound_manager.play_sound_custom_notification("curve-left")
    sound_manager.play_sound_custom_notification("curve-left")
    sound_manager.play_sound_custom_notification("curve-left")
    sound_manager.play_sound_custom_notification("curve-left")

    sleep(4)

    sound_manager.play_sound_custom_notification("stop")


# test_sound_manager()
