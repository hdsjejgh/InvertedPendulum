import pygame
import pymunk
import random
import math
import csv


WIDTH,HEIGHT = 600,400
FPS = 30
GRAVITY = (0,-500)
MAGNITUDE = 20

def phi(state):
    x,dx,theta,dtheta = state

    x -= WIDTH/2
    return [
        1.0,
        x,
        dx,
        theta,
        math.sin(theta),
        math.cos(theta),
        dtheta,
        dx * dtheta,
        x * dx,
        dtheta ** 2,
        dx ** 2,
        x ** 2
    ]

def conv_coords(a):
    x,y = a
    return x,HEIGHT-y


class Car:
    def __init__(self,x,y,w,h,space):
        self.h = h
        self.body = pymunk.Body(mass=30,moment=9999999)
        self.body.position = (x,y)
        self.shape = pymunk.Poly(self.body, [(-w / 2, 0), (w / 2, 0), (w / 2, -h), (-w / 2, -h)])
        self.shape.filter = pymunk.ShapeFilter(group=1, categories=0b1, mask=0b10)
        self.shape.density=1
        space.add(self.body,self.shape)
    def draw(self,screen):
        verts = self.shape.get_vertices()
        verts = [v.rotated(self.body.angle) + self.body.position for v in verts]
        verts = list(map(conv_coords,verts))
        pygame.draw.polygon(screen,"#440044",verts)

class Pendulum:
    def __init__(self,length,car,space):
        self.body = pymunk.Body(mass=5,moment=3000)
        self.body.position = car.body.position +pymunk.Vec2d(random.uniform(-1,1),5)
        self.shape = pymunk.Segment(self.body, (0, 0), (0, length), 2)
        self.shape.filter = pymunk.ShapeFilter(group=1, categories=0b100, mask=0b0)
        self.pivot = pymunk.PivotJoint(car.body,self.body,car.body.position)

        space.add(self.body,self.shape,self.pivot)
    def draw(self,screen):
        verts = [self.body.local_to_world(self.shape.a),self.body.local_to_world(self.shape.b) ]
        verts = list(map(conv_coords, verts))
        pygame.draw.line(screen,'#000000',verts[0],verts[1],width=3)


if __name__=="__main__":


    pygame.init()
    screen = pygame.display.set_mode((WIDTH,HEIGHT))
    space = pymunk.space.Space()
    space = pymunk.space.Space()
    space.gravity=GRAVITY

    clock = pygame.time.Clock()


    floor = pymunk.Segment(space.static_body, (0, 0), (WIDTH, 0), 0)
    wall1 = pymunk.Segment(space.static_body, (0, 0), (0,150), 0)
    wall2 = pymunk.Segment(space.static_body, (WIDTH, 0), (WIDTH,150), 0)

    space.add(floor)
    space.add(wall1)
    space.add(wall2)

    c = Car(WIDTH/2,50,100,50,space)
    p = Pendulum(200,c,space)



    def serialize(arr):
        arr = map(str,arr)
        arr = ','.join(arr)
        return '['+arr+']'


    samples = []



    running = True
    while running:

        action = 0
        state = [c.body.position[0], c.body.velocity[0], p.body.angle, p.body.angular_velocity]
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            c.body.velocity += pymunk.Vec2d(-MAGNITUDE,0)
            action-=1
        if keys[pygame.K_RIGHT]:
            c.body.velocity += pymunk.Vec2d(MAGNITUDE,0)
            action+=1

        samples.append([serialize(state),action])
        screen.fill("#60B0FF")
        space.step(1/FPS)


        if not -math.pi/2<state[2]<math.pi/2:
            running=False
        clock.tick(FPS)
        c.draw(screen)
        p.draw(screen)
        print(state)

        pygame.display.flip()

    print(len(samples))
    with open("samples.csv","a",newline='') as f:
        writer = csv.writer(f)
        writer.writerows(samples)

