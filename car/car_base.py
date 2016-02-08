import pyglet
import math
import os

from controller_models import SimpleControllerModel


class Car(object):
    def __init__(self,
                 control_model,
                 force=0.0,
                 acceleration=0.0,
                 velocity=0.0,
                 x_position=0.0,
                 y_position=0.0,
                 rotation_delta=0.0,
                 rotation=0.0,
                 mass=10.0):
        self.control_model = control_model
        self.force = force
        self.acceleration = acceleration
        self.velocity = velocity
        self.x_position, self.y_position = x_position, y_position
        self.rotation_delta = rotation_delta
        self.rotation = rotation
        self.mass = mass
        self.total_mass = mass

    def get_position(self):
        return self.x_position, self.y_position

    def get_rotation(self):
        return self.rotation

    def set_extra_mass(self, mass):
        self.total_mass = self.mass + mass

    def step(self):
        self.force, self.rotation_delta = self.control_model.get_force_and_rotation_delta()

        self.acceleration = self.force / self.total_mass
        self.velocity += self.acceleration

        self.rotation += self.rotation_delta * self.velocity / 3

        x_part, y_part = math.cos(math.radians(self.rotation)), math.sin(math.radians(self.rotation))

        self.x_position -= x_part * self.velocity
        self.y_position += y_part * self.velocity


class CarAnimation(pyglet.window.Window):
    def __init__(self, control_model, width=600, height=600):
        pyglet.window.Window.__init__(self, width=width, height=height, resizable=True)
        self.drawable_objects = []
        self.car_sprite = None
        self.create_drawable_objects()
        self.car = Car(control_model, x_position=self.car_sprite.x, y_position=self.car_sprite.y)

    def create_drawable_objects(self):
        car_img = pyglet.resource.image('images/car.png')
        car_img.anchor_x = car_img.width / 2
        car_img.anchor_y = car_img.height / 2

        self.car_sprite = pyglet.sprite.Sprite(car_img)
        self.car_sprite.position = (self.width / 2, self.height / 2)
        self.drawable_objects.append(self.car_sprite)

    def adjust_window_size(self):
        w, h = self.car_sprite.width * 3, self.car_sprite.height * 3
        self.width, self.height = w, h

    def on_draw(self):
        self.clear()
        for d in self.drawable_objects:
            d.draw()

    def move_objects(self, t):
        self.car.step()
        self.car_sprite.x, self.car_sprite.y = self.car.get_position()
        self.car_sprite.rotation = self.car.get_rotation()

win = CarAnimation(SimpleControllerModel())

# pyglet.gl.glClearColor(0.5, 0.5, 0.5, 1)
pyglet.clock.schedule_interval(win.move_objects, 1.0/40)
pyglet.app.run()