from Platformer.scripts.main.renderer import render
from Packages.PyVectors import *
from typing import List, Union
import pygame, time

pygame.init()


class mat4x4:  # row, collum
    def __init__(self):
        self.m = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]


def Matrix_PointAt(pos: Vec3, target: Vec3, up: Vec3) -> mat4x4:
    newForward = target - pos
    newForward = normalize(newForward)

    d = dot(up, newForward)
    a = newForward * Vec3(d, d, d)
    newUp = up - a
    newUp = normalize(newUp)

    newRight = cross(newUp, newForward)

    matrix = mat4x4()
    matrix.m[0][0] = newRight.x
    matrix.m[1][0] = newUp.x
    matrix.m[2][0] = newForward.x
    matrix.m[3][0] = pos.x
    matrix.m[0][1] = newRight.y
    matrix.m[1][1] = newUp.y
    matrix.m[2][1] = newForward.y
    matrix.m[3][1] = pos.y
    matrix.m[0][2] = newRight.z
    matrix.m[1][2] = newUp.z
    matrix.m[2][2] = newForward.z
    matrix.m[3][2] = pos.z
    matrix.m[0][3] = 0
    matrix.m[1][3] = 0
    matrix.m[2][3] = 0
    matrix.m[3][3] = 1
    return matrix


def Matrix_MakeRotationY(fAngleRad: float) -> mat4x4:
    matrix = mat4x4()
    matrix.m[0][0] = math.cos(fAngleRad)
    matrix.m[0][2] = math.sin(fAngleRad)
    matrix.m[2][0] = -math.sin(fAngleRad)
    matrix.m[1][1] = 1
    matrix.m[2][2] = math.cos(fAngleRad)
    matrix.m[3][3] = 1
    return matrix


def Matrix_MultiplyVector(m: mat4x4, i: Vec3) -> Vec3:
    v = Vec3(0, 0, 0)
    v.x = i.x * m.m[0][0] + i.y * m.m[1][0] + i.z * m.m[2][0] + i.w * m.m[3][0]
    v.y = i.x * m.m[0][1] + i.y * m.m[1][1] + i.z * m.m[2][1] + i.w * m.m[3][1]
    v.z = i.x * m.m[0][2] + i.y * m.m[1][2] + i.z * m.m[2][2] + i.w * m.m[3][2]
    v.w = i.x * m.m[0][3] + i.y * m.m[1][3] + i.z * m.m[2][3] + i.w * m.m[3][3]
    return Vec3(v.x, v.y, v.z)


def Matrix_QuickInverse(m: mat4x4) -> mat4x4:
    matrix = mat4x4()
    matrix.m[0][0] = m.m[0][0]
    matrix.m[0][1] = m.m[1][0]
    matrix.m[0][2] = m.m[2][0]
    matrix.m[0][3] = 0
    matrix.m[1][0] = m.m[0][1]
    matrix.m[1][1] = m.m[1][1]
    matrix.m[1][2] = m.m[2][1]
    matrix.m[1][3] = 0
    matrix.m[2][0] = m.m[0][2]
    matrix.m[2][1] = m.m[1][2]
    matrix.m[2][2] = m.m[2][2]
    matrix.m[2][3] = 0
    matrix.m[3][0] = -(m.m[3][0] * matrix.m[0][0] + m.m[3][1] * matrix.m[1][0] + m.m[3][2] * matrix.m[2][0])
    matrix.m[3][1] = -(m.m[3][0] * matrix.m[0][1] + m.m[3][1] * matrix.m[1][1] + m.m[3][2] * matrix.m[2][1])
    matrix.m[3][2] = -(m.m[3][0] * matrix.m[0][2] + m.m[3][1] * matrix.m[1][2] + m.m[3][2] * matrix.m[2][2])
    matrix.m[3][3] = 1
    return matrix


class Polygon:
    def __init__(self, point1: Vec3, point2: Vec3, point3: Vec3, normal: Vec3, color: Vec3, point1o: Vec3 = Vec3(0, 0, 0), point2o: Vec3 = Vec3(0, 0, 0), point3o: Vec3 = Vec3(0, 0, 0)) -> None:
        self.point1 = point1
        self.point2 = point2
        self.point3 = point3

        self.point1o = point1o
        self.point2o = point2o
        self.point3o = point3o

        self.normal = normal
        self.color = color
        light = dot(sun_dir, self.normal) * 0.5 + 0.5
        self.adjusted_color = clamp(self.color * Vec3(light, light, light), 0, 255)
    def Project(self, p: Vec3) -> None:
        o = [0, 0, 0]
        o[0] = p[0] * projMat.m[0][0] + p[1] * projMat.m[1][0] + p[2] * projMat.m[2][0] + projMat.m[3][0]
        o[1] = p[0] * projMat.m[0][1] + p[1] * projMat.m[1][1] + p[2] * projMat.m[2][1] + projMat.m[3][1]
        o[2] = p[0] * projMat.m[0][2] + p[1] * projMat.m[1][2] + p[2] * projMat.m[2][2] + projMat.m[3][2]
        w    = p[0] * projMat.m[0][3] + p[1] * projMat.m[1][3] + p[2] * projMat.m[2][3] + projMat.m[3][3]
        
        if w != 0:
            o[0] /= w
            o[1] /= w
            o[2] /= w
        
        return [((o[0] + 1) * 600, (o[1] + 1) * 375), p.z]
    def GetRotted(self, point: Vec3) -> Vec3:
        rotted = point - cam_pos
        rot = rotMatX.Rot([rotted.x, rotted.z])
        rotted = [rot[0], rotted[1], rot[1]]
        rot = rotMatY.Rot([rotted[1], rotted[2]])
        rotted = Vec3(rotted[0], rot[0], rot[1])
        return rotted
    def GetPolygons(self) -> List[object]:
        self.rotted1 = self.GetRotted(self.point1)
        self.rotted2 = self.GetRotted(self.point2)
        self.rotted3 = self.GetRotted(self.point3)
        return Triangle_ClipAgainstPlane(Vec3(0, 0, 0.1), Vec3(0, 0, 1), Polygon(self.rotted1, self.rotted2, self.rotted3, self.normal, self.color))
    def Render(self) -> Union[List[Vec2], bool]:
        data1 = self.Project(self.point1)
        data2 = self.Project(self.point2)
        data3 = self.Project(self.point3)

        z1 = data1[1]
        z2 = data2[1]
        z3 = data3[1]
        
        point1 = data1[0]
        point2 = data2[0]
        point3 = data3[0]

        light = dot(sun_dir, self.normal) * 0.5 + 0.5
        mid_point = (self.point3o + self.point2o + self.point1o) * Vec3(0.33333333333, 0.33333333333, 0.33333333333)
        rd = normalize(mid_point - cam_pos)
        specular = Specular(0.9, self.normal, rd, sun_dir)
        light += specular * light
        clipped = [[point1[0], point1[1]], [point2[0], point2[1]], [point3[0], point3[1]]]
        return [clipped, (z1 + z2 + z3) * 0.33333333333, clamp(self.color * Vec3(light, light, light), 0, 255)]  # clamp(self.color * Vec3(light, light, light), 0, 255)


def Vector_IntersectPlane(plane_p: Vec3, plane_n: Vec3, lineStart: Vec3, lineEnd: Vec3) -> Vec3:
    plane_n = normalize(plane_n)
    plane_d = -dot(plane_n, plane_p)
    ad = dot(lineStart, plane_n)
    bd = dot(lineEnd, plane_n)
    t = pyMath.divide0((-plane_d - ad), (bd - ad))
    lineStartToEnd = lineEnd - lineStart
    lineToIntersect = lineStartToEnd * Vec3(t, t, t)
    return lineStart + lineToIntersect


def Triangle_ClipAgainstPlane(plane_p: Vec3, plane_n: Vec3, in_tri: Polygon) -> List[Polygon]:
    plane_n = normalize(plane_n)

    def dist(p: Vec3) -> float:
        p = normalize(p)  # idk if this is needed or not
        return (plane_n.x * p.x + plane_n.y * p.y + plane_n.z * p.z - dot(plane_n, plane_p))
    
    inside_points = [None, None, None]
    outside_points = [None, None, None]
    nInsidePointCount = 0
    nOutsidePointCount = 0  

    d0 = dist(in_tri.point1)
    d1 = dist(in_tri.point2)
    d2 = dist(in_tri.point3)   
    
    if (d0 >= 0):
        inside_points[nInsidePointCount] = in_tri.point1
        nInsidePointCount += 1
    else:
        outside_points[nOutsidePointCount] = in_tri.point1
        nOutsidePointCount += 1
    if (d1 >= 0):
        inside_points[nInsidePointCount] = in_tri.point2
        nInsidePointCount += 1
    else:
        outside_points[nOutsidePointCount] = in_tri.point2
        nOutsidePointCount += 1
    if (d2 >= 0):
        inside_points[nInsidePointCount] = in_tri.point3
        nInsidePointCount += 1
    else:
        outside_points[nOutsidePointCount] = in_tri.point3
        nOutsidePointCount += 1

    if (nInsidePointCount == 0):
        return []
    if (nInsidePointCount == 3):
        return [in_tri]
    if (nInsidePointCount == 1 and nOutsidePointCount == 2):
        out_tri1 = Polygon(inside_points[0], Vector_IntersectPlane(plane_p, plane_n, inside_points[0], outside_points[0]), Vector_IntersectPlane(plane_p, plane_n, inside_points[0], outside_points[1]), in_tri.normal, in_tri.color)
        return [out_tri1]
    if (nInsidePointCount == 2 and nOutsidePointCount == 1):
        out_tri1 = Polygon(inside_points[0], inside_points[1], Vector_IntersectPlane(plane_p, plane_n, inside_points[0], outside_points[0]), in_tri.normal, in_tri.color, in_tri.point1, in_tri.point2, in_tri.point3)
        out_tri2 = Polygon(Vector_IntersectPlane(plane_p, plane_n, inside_points[1], outside_points[0]), out_tri1.point3, inside_points[1], in_tri.normal, in_tri.color, in_tri.point1, in_tri.point2, in_tri.point3)
        return [out_tri1, out_tri2]


class RotMat:
    def __init__(self, rot) -> None:
        self.sa = math.sin(rot)
        self.ca = math.cos(rot)
    def Rot(self, p: tuple) -> tuple:
        rottedX = p[1] * self.sa + p[0] * self.ca  # table cos and sin
        rottedY = p[1] * self.ca - p[0] * self.sa
        return (rottedX, rottedY)


class UI:
    def text(text: str, color, pos, size: float, center: bool = False, font: str = 'pixel.ttf', trans: int = 255):
        largeText = pygame.font.Font(font, size)
        textSurface = largeText.render(text, True, color)
        TextSurf, TextRect = textSurface, textSurface.get_rect()
        if trans != 255:
            surf = pygame.Surface(TextRect.size)
            if color == (0, 0, 0):
                surf.fill((255, 255, 255))
                surf.set_colorkey((255, 255, 255))
            else:
                surf.fill((0, 0, 0))
                surf.set_colorkey((0, 0, 0))
            surf.set_alpha(trans)
            n_pos = pos
            if center:
                pos = (TextRect.size[0] // 2, TextRect.size[1] // 2)
            else:
                pos = (0, 0)
        else:
            surf = screen
        if center:
            TextRect.center = pos
            sprite = surf.blit(TextSurf, TextRect)
        else:
            sprite = surf.blit(TextSurf, pos)
        
        if trans != 255:
            if center:
                screen.blit(surf, (n_pos[0] - TextRect.size[0] // 2, n_pos[1] - TextRect.size[1] // 2))
            else:
                screen.blit(surf, n_pos)
        return sprite


def clip(subjectPolygon, clipPolygon):
    def inside(p):
        return(cp2[0]-cp1[0])*(p[1]-cp1[1]) > (cp2[1]-cp1[1])*(p[0]-cp1[0])
    
    def computeIntersection():
        dc = [ cp1[0] - cp2[0], cp1[1] - cp2[1] ]
        dp = [ s[0] - e[0], s[1] - e[1] ]
        n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
        n2 = s[0] * e[1] - s[1] * e[0] 
        n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
        return [(n1*dp[0] - n2*dc[0]) * n3, (n1*dp[1] - n2*dc[1]) * n3]
    
    outputList = subjectPolygon
    cp1 = clipPolygon[-1]
    
    for clipVertex in clipPolygon:
        cp2 = clipVertex
        inputList = outputList
        outputList = []
        s = inputList[-1]
    
        for subjectVertex in inputList:
            e = subjectVertex
            if inside(e):
                if not inside(s):
                    outputList.append(computeIntersection())
                outputList.append(e)
            elif inside(s):
                outputList.append(computeIntersection())
            s = e
        cp1 = cp2
    return(outputList)


def V3(f: float) -> Vec3:
    return Vec3(f, f, f)


def RayTriangle(P1: Vec3, P2: Vec3, P3: Vec3, R1: Vec3, R2: Vec3) -> Union[bool, Vec3]:
    Normal = normalize(cross(P2 - P1, P3 - P1))

    Dist1 = dot(R1 - P1, Normal)
    Dist2 = dot(R2 - P1, Normal)

    if Dist1 * Dist2 >= 0: 
        return False; 

    if Dist1 == Dist2:
        return False 

    IntersectPos = R1 + (R2 - R1) * V3(-Dist1 / (Dist2 - Dist1))

    vTest = cross(Normal, P2 - P1)
    if dot(vTest, IntersectPos - P1) < 0:
        return False

    vTest = cross(Normal, P3 - P2)
    if dot(vTest, IntersectPos - P2) < 0:
        return False

    vTest = cross(Normal, P1 - P3)
    if dot(vTest, IntersectPos - P1) < 0:
        return False

    return IntersectPos


def Reflect(rd: Vec3, normal: Vec3) -> Vec3:
    return (rd - (normal * Vec3(2, 2, 2)) * (rd * normal))


class RenderPolygon:
    def __init__(self, polygon: Polygon) -> None:
        self.polygon = polygon
    def run(self) -> None:
        polygon_ = self.polygon
        clipped = polygon_.GetPolygons()
        for polygon in clipped:
            rendered = polygon.Render()
            rendered_polygons.append(rendered)


def Specular(smoothness: float, normal: Vec3, rd: Vec3, sun_dir: Vec3) -> float:
    try:
        specularAngle = math.acos(dot(normalize(sun_dir - rd), normal))
        specularExponent = specularAngle / (1. - smoothness)
        return math.exp(-specularExponent * specularExponent)
    except ValueError:
        return 0


def FV3(f: float) -> Vec3:
    return Vec3(f, f, f)


res = Vec2(1200, 750)
screen = pygame.display.set_mode(res)

sun_dir = normalize(Vec3(0.2, 0.7, -0.2))

radians = Vec2(math.atan2(sun_dir.x, sun_dir.z), math.atan2(sun_dir.y, sun_dir.z))

frame = 0

player_pos = Vec3(0, -10, 0)
cam_pos = player_pos - Vec3(0, 2, 0)
cam_rot = Vec2(0, 0)
velocity = Vec3(0, 0, 0)

player_chunk_pos = Vec2(-1200, -1200)

rotMatX = RotMat(cam_rot.x)
rotMatY = RotMat(cam_rot.y)

Near = 0.5
Far = 1000
fov = 90
aspectRatio = res.y / res.x
fovRad = 1 / pyMath.tan(fov * 0.5 / 180 * 3.14159)

projMat = mat4x4()
projMat.m[0][0] = aspectRatio * fovRad
projMat.m[1][1] = fovRad
projMat.m[2][2] = Far / (Far - Near)
projMat.m[3][2] = (Far * Near) / (Far - Near)
projMat.m[2][3] = 1
projMat.m[3][3] = 0

polygons = []

"""
map_size = [120, 120]
surface = png.getArray('Bedrock_mountains_template_Height_Output-1025.png')
for x in range(1025):
    for z in range(1025):
        surface[x][z] = (255 - surface[x][z].r) * 0.25


chunks = {}
for x in range(1024):
    for z in range(1024):
        chunk_pos = floor(Vec2(x / 16, z / 16))

        if chunk_pos not in chunks:
            chunks[chunk_pos] = []
        
        a = Vec3(x    , surface[x    ][z    ], z    )
        b = Vec3(x    , surface[x    ][z + 1], z + 1)
        c = Vec3(x + 1, surface[x + 1][z    ], z    )
        v1 = Vec3(b.x - a.x, b.y - a.y, b.z - a.z)
        v2 = Vec3(c.x - a.x, c.y - a.y, c.z - a.z)
        normal = normalize(cross(v1, v2)) * Vec3(-1, 1, -1)

        tl = Vec3(x, surface[x][z], z)
        tr = Vec3(x + 1, surface[x + 1][z], z)
        bl = Vec3(x, surface[x][z + 1], z + 1)
        br = Vec3(x + 1, surface[x + 1][z + 1], z + 1)

        polygon1 = Polygon(br.copy(), tr, tl.copy(), normal, Vec3(25, 200, 40))
        polygon2 = Polygon(tl.copy(), bl, br.copy(), normal, Vec3(25, 200, 40))
        chunks[chunk_pos].append(polygon1)
        chunks[chunk_pos].append(polygon2)
"""

#data = open('utah teapot.obj').read().split('\n')
#data = open('ocean.obj').read().split('\n')
#data = open('half earth.obj').read().split('\n')
data = open('terrain.obj').read().split('\n')

normals = []
verts = []

tri_number = 0

for line in data:
    if len(line) > 2:
        if line[0] == 'v':
            if line[1] == ' ':
                points = line.split(' ')
                verts.append(Vec3(float(points[1]), float(points[2]), float(points[3])))
            elif line[1] == 'n':
                points = line.split(' ')
                normals.append(Vec3(float(points[1]), float(points[2]), float(points[3])))
for line in data:
    if len(line) > 2:
        if line[0] == 'f':
            points = line.split(' ')
            point1 = verts[int(points[1].split('/')[0]) - 1] * Vec3(1, -1, 1)
            point2 = verts[int(points[2].split('/')[0]) - 1] * Vec3(1, -1, 1)
            point3 = verts[int(points[3].split('/')[0]) - 1] * Vec3(1, -1, 1)
            if tri_number < len(normals):
                polygons.append(Polygon(point1, point2, point3, normals[tri_number], Vec3(200, 200, 200)))
                tri_number += 1
            else:
                break

polygons = [polygons]

clipping_bounds = [[0, 0], [0, res.y], [res.x, res.y], [res.x, 0]]

rendered_polygons = []

held = {
    "w": False,
    "a": False,
    "s": False,
    "d": False,
    " ": False,
    "shift": False,
    "left": False,
    "right": False,
    "up": False,
    "down": False,
}

dt = 0

running = True


def Rotate(point: Vec3, XZ: RotMat, YZ: RotMat) -> Vec3:
    rotted = XZ.Rot(Vec2(point.x, point.z))
    point = Vec3(rotted[0], point.y, rotted[1])
    rotted = XZ.Rot(Vec2(point.y, point.z))
    point = Vec3(point.x, rotted[0], rotted[1])
    return point


while running:
    #sun_dir = normalize(Vec3(math.sin(frame) * 0.2, math.sin(frame), math.cos(frame) * 0.8))

    s = time.time()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            pygame.quit()
            break
        elif event.type == pygame.KEYDOWN:
            if event.key == ord('w'):
                held["w"] = True
            elif event.key == ord('a'):
                held["a"] = True
            elif event.key == ord('s'):
                held["s"] = True
            elif event.key == ord('d'):
                held["d"] = True
            elif event.key == ord(' '):
                held[" "] = True
            elif event.key == pygame.K_LSHIFT:
                held["shift"] = True
            elif event.key == pygame.K_LEFT:
                held["left"] = True
            elif event.key == pygame.K_RIGHT:
                held["right"] = True
            elif event.key == pygame.K_UP:
                held["up"] = True
            elif event.key == pygame.K_DOWN:
                held["down"] = True
        elif event.type == pygame.KEYUP:
            if event.key == ord('w'):
                held["w"] = False
            elif event.key == ord('a'):
                held["a"] = False
            elif event.key == ord('s'):
                held["s"] = False
            elif event.key == ord('d'):
                held["d"] = False
            elif event.key == ord(' '):
                held[" "] = False
            elif event.key == pygame.K_LSHIFT:
                held["shift"] = False
            elif event.key == pygame.K_LEFT:
                held["left"] = False
            elif event.key == pygame.K_RIGHT:
                held["right"] = False
            elif event.key == pygame.K_UP:
                held["up"] = False
            elif event.key == pygame.K_DOWN:
                held["down"] = False
    
    if not running:
        break

    # finding the aplied force
    movement_speed = 120 * dt
    pi = 3.14159
    rot_speed = pi * dt

    F_app = Vec3(0, 0, 0)
    if held["w"]:
        F_app += Vec3(movement_speed * -math.sin(cam_rot.x), 0, movement_speed * math.cos(cam_rot.x))
    if held["a"]:
        F_app += Vec3(movement_speed * -math.sin(cam_rot.x + pi * 0.5), 0, movement_speed * math.cos(cam_rot.x + pi * 0.5))
    if held["s"]:
        F_app += Vec3(movement_speed * math.sin(cam_rot.x), 0, movement_speed * -math.cos(cam_rot.x))
    if held["d"]:
        F_app += Vec3(movement_speed * -math.sin(cam_rot.x - pi * 0.5), 0, movement_speed * math.cos(cam_rot.x - pi * 0.5))
    if held["shift"]:
        F_app += Vec3(0, movement_speed, 0)
    if held[" "]:
        F_app += Vec3(0, -movement_speed, 0)
    if held["left"]:
        change_cam = Vec2(rot_speed, 0)
        cam_rot += change_cam
    if held["right"]:
        change_cam = Vec2(-rot_speed, 0)
        cam_rot += change_cam
    if held["up"]:
        change_cam = Vec2(0, rot_speed)
        cam_rot += change_cam
    if held["down"]:
        change_cam = Vec2(0, -rot_speed)
        cam_rot += change_cam

    # finding the net force

    #"""
    Fnet = Vec3(0, 0, 0)
    F_g = Vec3(0, 9.81, 0)  # gravity
    Fnet = Fnet + F_g
    a = Fnet
    
    ray_start = player_pos
    ray_end = player_pos + FV3(0.5) * a * FV3(pow(dt, 2)) + FV3(dt) * velocity  # the end of the ray
    movement_direction = normalize(ray_end)
    for mesh in polygons:
        for polygon in mesh:
            intersection = RayTriangle(polygon.point1, polygon.point2, polygon.point3, ray_start, ray_end)
            # check if the point is right on the surface (the intersection function only works when the point isnt on the surface)
            if intersection is not False:
                #dst_to_surface = length(intersection) - 0.000001  # the intersection function dosent work on the surface of an object so im moving it back a bit
                #collition_point = movement_direction * Vec3(dst_to_surface, dst_to_surface, dst_to_surface)

                surface_normal = polygon.normal

                v_dot_n = dot(surface_normal, velocity)
                velocity = velocity - surface_normal * FV3(v_dot_n)

                a_dot_n = dot(surface_normal, a)
                a = a - surface_normal * FV3(a_dot_n)
                
                break
        else:
            continue
        break
    
    #"""

    # changing the position and velocity

    player_pos += FV3(0.5) * a * FV3(pow(dt, 2)) + FV3(dt) * velocity
    velocity += a * Vec3(dt, dt, dt)
    #player_pos += F_app * Vec3(dt, dt, dt)
    cam_pos = player_pos - Vec3(0, 2, 0)

    rotMatX = RotMat(cam_rot.x)
    rotMatY = RotMat(cam_rot.y)

    vUp = Vec3(0, 1, 0)
    vTarget = Vec3(0, 0, 1)
    matCameraRot = Matrix_MakeRotationY(cam_rot.y)
    vLookDir = Matrix_MultiplyVector(matCameraRot, vTarget)
    vTarget = cam_pos + vLookDir
    matCamera = Matrix_PointAt(cam_pos, vTarget, vUp)

    matView = Matrix_QuickInverse(matCamera)

    forwards_vector = Rotate(Vec3(0, 0, -1), rotMatX, rotMatY)
    radians = Vec2(math.atan2(sun_dir.x - forwards_vector.x, sun_dir.z - forwards_vector.z), math.atan2(sun_dir.y - forwards_vector.y, sun_dir.z - forwards_vector.z))

    """
    # junk from something else
    polygons = []
    player_chunk_pos = floor(Vec2(cam_pos.x, cam_pos.z) / Vec2(16, 16))
    for X in range(-1, 2):
        for Z in range(-1, 2):
            cp = Vec2(player_chunk_pos.x + X, player_chunk_pos.y + Z)
            #if cp in chunks and (X, Z) not in [(0, 0), (2, 0), (0, 2), (2, 2)]:
            try:
                polygons.append(chunks[cp])
            except KeyError:
                pass
    """

    screen.fill((225, 225, 255))

    renderings = []
    rendered_polygons = []
    for polygonChunks in polygons:
        for polygon_ in polygonChunks:
            renderings.append(RenderPolygon(polygon_))
    
    PyThreading.Disperse(renderings, max_threads = 5)
    
    rendered_polygons.sort(key = lambda key: -key[1])

    for polygon in rendered_polygons:
        try:
            polygon_clipped = clip(clipping_bounds, polygon[0])
        except IndexError:
            continue
        if len(polygon_clipped) >= 3:
            pygame.draw.polygon(screen, polygon[2], polygon_clipped)
    
    UI.text(f'FPS: {round(1 / max(dt, 0.00000001))}', (0, 0, 40), (10, 10), 35)

    pygame.display.update()

    frame += dt

    e = time.time()
    dt = e - s

