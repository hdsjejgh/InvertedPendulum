import joblib
import numpy as np
import pygame
import pymunk
import random
import math
from sim import phi,conv_coords,FPS,WIDTH,HEIGHT,Pendulum,Car,MAGNITUDE,GRAVITY
from FVI import ACTIONS,NUM_FEATURES


pygame.init()
screen = pygame.display.set_mode((WIDTH,HEIGHT))
space = pymunk.space.Space()
space.gravity=GRAVITY

clock = pygame.time.Clock()

A = np.load("A.npy")
B = np.load("B.npy")
theta = np.load("Theta.npy")
print(theta)

print(theta)
scalar_s = joblib.load("scaler_s")


floor = pymunk.Segment(space.static_body, (0, 0), (WIDTH, 0), 0)
wall1 = pymunk.Segment(space.static_body, (0, 0), (0,150), 0)
wall2 = pymunk.Segment(space.static_body, (WIDTH, 0), (WIDTH,150), 0)

space.add(floor)
space.add(wall1)
space.add(wall2)

c = Car(WIDTH/2,50,100,50,space)
p = Pendulum(200,c,space)




running = True
while running:
    state = np.array(phi([c.body.position[0], c.body.velocity[0], p.body.angle, p.body.angular_velocity]))
    state = np.reshape(scalar_s.transform(np.reshape(state,(1,-1))),(-1))
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    values = []
    state = np.array(state).reshape((NUM_FEATURES, 1))
    for a in ACTIONS:
        s_prime = A@state + B*a
        values.append(theta.T @ s_prime)
    best = ACTIONS[values.index(max(values))]
    print(best)
    if best==-1:
        c.body.velocity += pymunk.Vec2d(-MAGNITUDE, 0)
    if best==1:
        c.body.velocity += pymunk.Vec2d(MAGNITUDE, 0)

    screen.fill("#60B0FF")
    space.step(1/FPS)

    if not -math.pi/2<state[2]<math.pi/2:
        running=False
    clock.tick(FPS)
    c.draw(screen)
    p.draw(screen)
    pygame.display.flip()

