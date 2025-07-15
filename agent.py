import joblib
import numpy as np
import pygame
import pymunk
import math
from sim import phi,FPS,WIDTH,HEIGHT,Pendulum,Car,MAGNITUDE,GRAVITY
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
    x = state[1]
    #gets the angle to check if pendulum fell later
    angle = state[3]
    #scales the state vector using same scaler in FVI.py
    state = np.reshape(scalar_s.transform(np.reshape(state,(1,-1))),(-1))
    #state = np.reshape(np.reshape(state, (1, -1)), (-1))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    values = []
    #makes state a column vector
    state = np.array(state).reshape((NUM_FEATURES, 1))
    for a in ACTIONS:
        #s' is predicted next state
        s_prime = A@state + B*a
        #adds value of the state gotten from action a to values
        values.append(theta.T @ s_prime)
    #gets action with highest value
    best = ACTIONS[values.index(max(values))]
    print(f"x={x:.2f}, Q={values}, chosen={best}")
    if best==-1:
        #move left
        c.body.velocity += pymunk.Vec2d(-MAGNITUDE, 0)
    if best==1:
        #move right
        c.body.velocity += pymunk.Vec2d(MAGNITUDE, 0)

    screen.fill("#60B0FF")
    space.step(1/FPS)

    if not -math.pi/2<angle<math.pi/2:
        #If pendulum not angled in top half, pendulum falls
        running=False
    clock.tick(FPS)
    c.draw(screen)
    p.draw(screen)
    pygame.display.flip()