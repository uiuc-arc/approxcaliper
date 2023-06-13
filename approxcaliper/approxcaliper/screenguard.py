from subprocess import run


class ScreenGuard:
    def __init__(self, screen_name: str, *args) -> None:
        self.screen_name = screen_name
        self.args = args

    def __enter__(self) -> None:
        run(["screen", "-dm", "-S", self.screen_name, *self.args])

    def __exit__(self, *_) -> None:
        run(["screen", "-X", "-S", self.screen_name, "quit"])
