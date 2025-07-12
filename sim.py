import pygame
import pymunk

WIDTH,HEIGHT = 600,600
pygame.init()
screen = pygame.display.set_mode((WIDTH,HEIGHT))
space = pymunk.space.Space()
space.gravity=(0,-981)
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
        self.body = pymunk.Body(mass=5,moment=3000)
        self.body.position = car.body.position +pymunk.Vec2d(0,5)
        self.shape = pymunk.Segment(self.body, (0, 0), (0, length), 2)
        self.shape.filter = pymunk.ShapeFilter(group=1, categories=0b100, mask=0b0)
        self.pivot = pymunk.PivotJoint(car.body,self.body,car.body.position)

        space.add(self.body,self.shape,self.pivot)
    def draw(self):
        verts = [self.body.local_to_world(self.shape.a),self.body.local_to_world(self.shape.b) ]
        verts = list(map(conv_coords, verts))
        print(verts)
        pygame.draw.line(screen,'#000000',verts[0],verts[1],width=3)



floor = pymunk.Segment(space.static_body, (0, 0), (WIDTH, 0), 0)
wall1 = pymunk.Segment(space.static_body, (0, 0), (0,150), 0)
wall2 = pymunk.Segment(space.static_body, (WIDTH, 0), (WIDTH,150), 0)

space.add(floor)
space.add(wall1)
space.add(wall2)

c = Car(WIDTH/2,100,100,100)
p = Pendulum(200,c)

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        c.body.velocity += pymunk.Vec2d(-10,0)
    if keys[pygame.K_RIGHT]:
        c.body.velocity += pymunk.Vec2d(10,0)

    screen.fill("#60B0FF")
    space.step(1/FPS)
    clock.tick(FPS)
    c.draw()
    p.draw()
    pygame.display.flip()