import pygame
import pymunk
import random
import math
import csv

WIDTH,HEIGHT = 600,400
pygame.init()
screen = pygame.display.set_mode((WIDTH,HEIGHT))
space = pymunk.space.Space()
space.gravity=(0,-601)
FPS = 90
clock = pygame.time.Clock()

def conv_coords(a):
    x,y = a
    return x,HEIGHT-y

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

def phi(state):
    x,dx,t,dt = state
    return [x,dx,t,dt,dx**2,dt**2]

def serialize(arr):
    arr = map(str,arr)
    arr = ','.join(arr)
    return '['+arr+']'


samples = []

state = None

running = True
while running:
    if state is not None:
        samples.append([serialize(prev_state),serialize(action),serialize(state),reward])
    action = [0, 0]
    state = phi([c.body.position[0], c.body.velocity[0], p.body.angle, p.body.angular_velocity])
    reward = 1
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        c.body.velocity += pymunk.Vec2d(-10,0)
        action[0]=1
    if keys[pygame.K_RIGHT]:
        c.body.velocity += pymunk.Vec2d(10,0)
        action[1]=1

    screen.fill("#60B0FF")
    space.step(1/FPS)
    if not -math.pi/18<state[2]<math.pi/18:
        reward = 0

    if not -math.pi/2<state[2]<math.pi/2:
        running=False
    clock.tick(FPS)
    c.draw()
    p.draw()
    print(state)

    prev_state = state

    pygame.display.flip()

# print(len(samples))
# with open("samples.csv","a",newline='') as f:
#     writer = csv.writer(f)
#     writer.writerows(samples)
#
