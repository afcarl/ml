from abc import abstractmethod


class AbstractControllerModel(object):
    @abstractmethod
    def get_force_and_rotation_delta(self):
        pass


class SimpleControllerModel(AbstractControllerModel):
    def __init__(self):
        self.counter = 0
        self.force = 0.0
        self.rotation = 0.0

    def get_force_and_rotation_delta(self):
        if self.counter < 150:
            self.force = 0.5
            self.rotation = 4
        elif self.counter < 200:
            self.force = -0.5
            self.rotation = -4
        else:
            self.force = 0
            self.rotation = 4
        self.counter += 1
        return self.force, self.rotation
