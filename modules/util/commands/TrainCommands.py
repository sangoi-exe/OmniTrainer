import threading

from modules.util.config.SampleConfig import SampleConfig


class TrainCommands:
    def __init__(
            self,
            on_command=None# Callable[[TrainCommands], None] = lambda _: None
    ):
        self.__command_lock = threading.Lock()
        self.reset()
        self.__stop_command = False
        self.__on_command = on_command

    def reset(self):
        #don't reset stop
        with self.__command_lock:
            self.__sample_custom_commands = []
            self.__sample_default_command = False
            self.__backup_command = False
            self.__save_command = False
            self.__pause_requested = False
            self.__resume_requested = False

    def set_on_command(
            self,
            on_command#: Callable[[TrainCommands], None] = lambda _: None
    ):
        self.__on_command = on_command

    def get_and_reset_on_command(self):
        on_command = self.__on_command
        self.__on_command=None
        return on_command

    def stop(self):
        with self.__command_lock:
            self.__stop_command = True
        if self.__on_command:
            self.__on_command(self)

    def get_stop_command(self) -> bool:
        with self.__command_lock:
            return self.__stop_command

    def sample_custom(self, sample_params: SampleConfig):
        with self.__command_lock:
            self.__sample_custom_commands.append(sample_params)
        if self.__on_command:
            self.__on_command(self)

    def get_and_reset_sample_custom_commands(self) -> list[SampleConfig]:
        with self.__command_lock:
            sample_custom_commands = self.__sample_custom_commands
            self.__sample_custom_commands = []
            return sample_custom_commands

    def sample_default(self):
        with self.__command_lock:
            self.__sample_default_command = True
        if self.__on_command:
            self.__on_command(self)

    def get_and_reset_sample_default_command(self) -> bool:
        with self.__command_lock:
            sample_default_command = self.__sample_default_command
            self.__sample_default_command = False
            return sample_default_command

    def backup(self):
        with self.__command_lock:
            self.__backup_command = True
        if self.__on_command:
            self.__on_command(self)

    def get_and_reset_backup_command(self) -> bool:
        with self.__command_lock:
            backup_command = self.__backup_command
            self.__backup_command = False
            return backup_command

    def save(self):
        with self.__command_lock:
            self.__save_command = True
        if self.__on_command:
            self.__on_command(self)

    def get_and_reset_save_command(self) -> bool:
        with self.__command_lock:
            save_command = self.__save_command
            self.__save_command = False
            return save_command

    def request_pause(self) -> bool:
        with self.__command_lock:
            if self.__pause_requested or self.__resume_requested:
                return False
            self.__pause_requested = True
        if self.__on_command:
            self.__on_command(self)
        return True

    def request_resume(self) -> bool:
        with self.__command_lock:
            if self.__resume_requested or self.__pause_requested:
                return False
            self.__resume_requested = True
        if self.__on_command:
            self.__on_command(self)
        return True

    def get_and_reset_pause_request(self) -> bool:
        with self.__command_lock:
            pause_requested = self.__pause_requested
            self.__pause_requested = False
            return pause_requested

    def get_and_reset_resume_request(self) -> bool:
        with self.__command_lock:
            resume_requested = self.__resume_requested
            self.__resume_requested = False
            return resume_requested

    def is_pause_pending(self) -> bool:
        with self.__command_lock:
            return self.__pause_requested

    def is_resume_pending(self) -> bool:
        with self.__command_lock:
            return self.__resume_requested

    def merge(self, other):
        if other.get_stop_command():
            self.stop()
        for entry in other.get_and_reset_sample_custom_commands():
            self.sample_custom(entry)
        if other.get_and_reset_sample_default_command():
            self.sample_default()
        if other.get_and_reset_backup_command():
            self.backup()
        if other.get_and_reset_save_command():
            self.save()
        if other.get_and_reset_pause_request():
            self.request_pause()
        if other.get_and_reset_resume_request():
            self.request_resume()
