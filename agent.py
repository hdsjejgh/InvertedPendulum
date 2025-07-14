import joblib
import numpy as np
import pygame
import pymunk
import random
import math
import csv
from sim import phi,conv_coords
from FVI import  ACTIONS

WIDTH,HEIGHT = 600,400
pygame.init()
screen = pygame.display.set_mode((WIDTH,HEIGHT))
space = pymunk.space.Space()
space.gravity=(0,-601)
FPS = 90
clock = pygame.time.Clock()

A = np.load("A.npy")
B = np.load("B.npy")
theta = np.load("Theta.npy")

scalar_s = joblib.load("scaler_s")
scalar_a = joblib.load("scaler_a")


class Car:
    def __init__(self,x,y,w,h):
        self.h = h
        self.body = pymunk.Body(mass=30,moment=9999999)
        self.body.position = (x,y)
        self.shape = pymunk.Poly(self.body, [(-w / 2, 0), (w / 2, 0), (w / 2, -h), (-w / 2, -h)])
        self.shape.filter = pymunk.ShapeFilter(group=1, categories=0b1, mask=0b10)
        self.shape.density=1
        space.add(self.body,self.shape)
    def draw(self):
        verts = self.shape.get_vertices()
        verts = [v.rotated(self.body.angle) + self.body.position for v in verts]
        verts = list(map(conv_coords,verts))
        pygame.draw.polygon(screen,"#440044",verts)

class Pendulum:
    def __init__(self,length,car):
        self.body = pymunk.Body(mass=5,moment=2750)
        self.body.position = car.body.position +pymunk.Vec2d(random.uniform(-1,1),5)
        self.shape = pymunk.Segment(self.body, (0, 0), (0, length), 2)
        self.shape.filter = pymunk.ShapeFilter(group=1, categories=0b100, mask=0b0)
        self.pivot = pymunk.PivotJoint(car.body,self.body,car.body.position)

        space.add(self.body,self.shape,self.pivot)
    def draw(self):
        verts = [self.body.local_to_world(self.shape.a),self.body.local_to_world(self.shape.b) ]
        verts = list(map(conv_coords, verts))
        pygame.draw.line(screen,'#000000',verts[0],verts[1],width=3)


floor = pymunk.Segment(space.static_body, (0, 0), (WIDTH, 0), 0)
wall1 = pymunk.Segment(space.static_body, (0, 0), (0,150), 0)
wall2 = pymunk.Segment(space.static_body, (WIDTH, 0), (WIDTH,150), 0)

space.add(floor)
space.add(wall1)
space.add(wall2)

c = Car(WIDTH/2,50,100,50)
p = Pendulum(200,c)




running = True
while running:
    state = np.array(phi([c.body.position[0], c.body.velocity[0], p.body.angle, p.body.angular_velocity]))
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    values = []
    for a in ACTIONS:
        s_prime = A@state.T + B@np.array(a).T
        values.append(theta.T @ s_prime)
    best = ACTIONS[values.index(max(values))]
    print(best)
    if best[0]:
        c.body.velocity += pymunk.Vec2d(-10, 0)
    if best[1]:
        c.body.velocity += pymunk.Vec2d(10, 0)

    screen.fill("#60B0FF")
    space.step(1/FPS)

    if not -math.pi/2<state[2]<math.pi/2:
        running=False
    clock.tick(FPS)
    c.draw()
    p.draw()
    pygame.display.flip()

