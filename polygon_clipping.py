from typing import List, Union  # to specify the types a function takes or outputs (to help keep the code nice and clean)
from PyVectors import *  # a vector math library i made
from enum import Enum  # for enums
import pygame, time  # for rendering and time change

pygame.init()


# --------- Matricies ---------

# 4x4 matrix (just for storing the data)
class mat4x4:  # row, collum
    def __init__(self):
        self.m = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]


# idk, isnt mine, im not good with matrix stuff
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


# idk, isnt mine, im not good with matrix stuff but i think it rotates around the y axis
def Matrix_MakeRotationY(fAngleRad: float) -> mat4x4:
    matrix = mat4x4()
    matrix.m[0][0] = math.cos(fAngleRad)
    matrix.m[0][2] = math.sin(fAngleRad)
    matrix.m[2][0] = -math.sin(fAngleRad)
    matrix.m[1][1] = 1
    matrix.m[2][2] = math.cos(fAngleRad)
    matrix.m[3][3] = 1
    return matrix


# multiples a vector3 by a 4x4 matrix (not my own, once again i dont know matrix stuff sense i havent been tought it)
def Matrix_MultiplyVector(m: mat4x4, i: Vec3) -> Vec3:
    v = Vec3(0, 0, 0)
    v.x = i.x * m.m[0][0] + i.y * m.m[1][0] + i.z * m.m[2][0] + i.w * m.m[3][0]
    v.y = i.x * m.m[0][1] + i.y * m.m[1][1] + i.z * m.m[2][1] + i.w * m.m[3][1]
    v.z = i.x * m.m[0][2] + i.y * m.m[1][2] + i.z * m.m[2][2] + i.w * m.m[3][2]
    v.w = i.x * m.m[0][3] + i.y * m.m[1][3] + i.z * m.m[2][3] + i.w * m.m[3][3]
    return Vec3(v.x, v.y, v.z)


# idk, isnt mine, im not good with matrix stuff
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


# projects a point from 3d space to 2d space on an infinit plane (using a 4x4 projection matrix and the matrix vector multiplacation function above)
def Project(p: Vec3) -> List[Union[Tuple[float], float]]:
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


# ---------- clipping ----------

# idk, part of the clipping, didnt make myself as this stuff require much more advanced math than i know at this time, i did convert it from c++ to python3 though
def Vector_IntersectPlane(plane_p: Vec3, plane_n: Vec3, lineStart: Vec3, lineEnd: Vec3) -> Vec3:
    plane_n = normalize(plane_n)
    plane_d = -dot(plane_n, plane_p)
    ad = dot(lineStart, plane_n)
    bd = dot(lineEnd, plane_n)
    t = pyMath.divide0((-plane_d - ad), (bd - ad))
    lineStartToEnd = lineEnd - lineStart
    lineToIntersect = lineStartToEnd * Vec3(t, t, t)
    return lineStart + lineToIntersect


# idk, clips a polygon to the veiw plane, didnt make myself as this stuff require much more advanced math than i know at this time, i did convert it from c++ to python3 though
def Triangle_ClipAgainstPlane(plane_p: Vec3, plane_n: Vec3, in_tri: Vec3) -> List[Vec3]:
    plane_n = normalize(plane_n)

    def dist(p: Vec3) -> float:
        p = normalize(p)  # idk if this is needed or not
        return (plane_n.x * p.x + plane_n.y * p.y + plane_n.z * p.z - dot(plane_n, plane_p))
    
    inside_points = [None, None, None]
    outside_points = [None, None, None]
    nInsidePointCount = 0
    nOutsidePointCount = 0  

    d0 = dist(in_tri[0])
    d1 = dist(in_tri[1])
    d2 = dist(in_tri[2])   
    
    if (d0 >= 0):
        inside_points[nInsidePointCount] = in_tri[0]
        nInsidePointCount += 1
    else:
        outside_points[nOutsidePointCount] = in_tri[0]
        nOutsidePointCount += 1
    if (d1 >= 0):
        inside_points[nInsidePointCount] = in_tri[1]
        nInsidePointCount += 1
    else:
        outside_points[nOutsidePointCount] = in_tri[1]
        nOutsidePointCount += 1
    if (d2 >= 0):
        inside_points[nInsidePointCount] = in_tri[2]
        nInsidePointCount += 1
    else:
        outside_points[nOutsidePointCount] = in_tri[2]
        nOutsidePointCount += 1

    if (nInsidePointCount == 0):
        return []
    if (nInsidePointCount == 3):
        return [in_tri]
    if (nInsidePointCount == 1 and nOutsidePointCount == 2):
        out_tri1 = [inside_points[0], Vector_IntersectPlane(plane_p, plane_n, inside_points[0], outside_points[0]), Vector_IntersectPlane(plane_p, plane_n, inside_points[0], outside_points[1])]
        return [out_tri1]
    if (nInsidePointCount == 2 and nOutsidePointCount == 1):
        out_tri1 = [inside_points[0], inside_points[1], Vector_IntersectPlane(plane_p, plane_n, inside_points[0], outside_points[0])]
        out_tri2 = [Vector_IntersectPlane(plane_p, plane_n, inside_points[1], outside_points[0]), out_tri1[2], inside_points[1]]
        return [out_tri1, out_tri2]


# clips a 2d polygon to the bounds of the screen to not waset time rendering unseen pixels (not my own but i know why/how it works, there was just an already made version and im lazy)
def clip(subjectPolygon, clipPolygon):
    """
    def ClipLineVirticle(p1: Vec2, p2: Vec2, x: float) -> Union[Vec2, bool]:
        if (p2.x < x and p1.x > x):
            start = p2
            dif = p1 - p2
        elif (p2.x < x and p1.x > x):
            dif = p2 - p1
            start = p1
        else:
            return False  # no intersection
        slope = pyMath.divide0(dif.y, dif.x)
        ny = x - start.x
        return slope * ny + start.y

    def ClipLineHorizontal(p1: Vec2, p2: Vec2, y: float) -> Union[Vec2, bool]:
        if (p2.y < y and p1.y > y):
            start = p2
            dif = p1 - p2
        elif (p2.y < y and p1.y > y):
            dif = p2 - p1
            start = p1
        else:
            return False  # no intersection
        slope = pyMath.divide0(dif.y, dif.x)
        nx = y - start.y
        return pyMath.divide0(nx, slope) - start.y
    
    def VectorizeV2(p: List[float]) -> Vec2:
        return Vec2(p[0], p[1])

    def LowerPoint(p1: Vec2, p2: Vec2) -> List[Vec2]:  # sense the points dont come in sorted correctly
        if p1.y < p2.y:
            return [p2, p2]
        return [p1, p2]

    edge1 = [VectorizeV2(clipPolygon[0]), VectorizeV2(clipPolygon[1])]
    edge2 = [VectorizeV2(clipPolygon[1]), VectorizeV2(clipPolygon[2])]
    edge3 = [VectorizeV2(clipPolygon[2]), VectorizeV2(clipPolygon[0])]
    edges = [edge1, edge2, edge3]

    new_edges = []
    for edge in edges:
        new_edge_height = ClipLineHorizontal(edge[0], edge[1], 0)
        if new_edge_height is False:
            new_edge = edge
        else:
            sorted_edge = LowerPoint(edge[0], edge[1])
            new_edge = [sorted_edge[0], Vec2(new_edge_height, 0)]
        new_edge_height = ClipLineHorizontal(new_edge[0], new_edge[1], 750)
        if new_edge_height is not False:
            sorted_edge = LowerPoint(edge[0], edge[1])
            new_edge_height = [sorted_edge[1], Vec2(new_edge_height, 750)]
        # clip horizontaly
        #new_edge = ClipLineVirticle(new_edge[0], new_edge[1], 0)
        #new_edge = ClipLineVirticle(new_edge[0], new_edge[1], 1200)

        new_edges.append(new_edge)
    
    return [new_edges[0][0], new_edges[0][1], new_edges[1][0], new_edges[1][1], new_edges[2][0], new_edges[2][1]]
    #"""
    # clipPolygon is the polygon/triangle
    """
    
    #"""

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
    return outputList


# finds the intersection of a ray/vector with a 3d triangle (not my own, i dont know the advanced math required yet)
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


# ------------ polygons and verticies ------------


# for tags for physics with polygons
class Tags(Enum):
    bouncy = 1


# stores data on a polygon (3d triangle)
class Polygon:
    # initalizing the polygon
    def __init__(self, point1: int, point2: int, point3: int, normal: Vec3, color: Vec3, tags: dict[Tags] = {}) -> None:
        # assigning the points
        self.point1 = point1
        self.point2 = point2
        self.point3 = point3

        # keeping a copy of the origonal points 
        self.point1o = verts[point1].point
        self.point2o = verts[point2].point
        self.point3o = verts[point3].point

        # the normal and pre calculated light (dosent give much of a preformace boost, maybe 0.5 fps)
        self.normal = normal
        self.color = mix(color, undertone, undertone_strength)

        # the light level (insnt being used because it only saves 1 or 2 fps by precalculating it)
        light = dot(sun_dir, self.normal) * 0.5 + 0.5
        self.adjusted_color = clamp(self.color * Vec3(light, light, light), 0, 255)

        # the tags
        self.tags = tags
    def GetVerts(self, verts_rotted: List[Vec3]) -> None:  # gets the projected verts based on the vert indexes
        self.rotted1 = verts_rotted[self.point1]
        self.rotted2 = verts_rotted[self.point2]
        self.rotted3 = verts_rotted[self.point3]
    def GetPolygons(self) -> List[Vec3]:  # clips the polygon agains the veiw plane
        return Triangle_ClipAgainstPlane(Vec3(0, 0, 0.1), Vec3(0, 0, 1), [self.rotted1, self.rotted2, self.rotted3])
    def Render(self, point_projected: List[Vec3]) -> Union[List[Vec2], bool]:  # renders the polygon to the scren
        # projected the points
        data1 = Project(point_projected[0])
        data2 = Project(point_projected[1])
        data3 = Project(point_projected[2])
        
        # finding the depth of the polygon
        z1 = data1[1]
        z2 = data2[1]
        z3 = data3[1]
        
        # the poitns
        point1 = data1[0]
        point2 = data2[0]
        point3 = data3[0]

        normal_rotted = Rotate(self.normal, rotMatY, rotMatX)

        #mid_point_projected1 = Project((self.rotted1 + self.rotted2 + self.rotted3) * FV3(0.33333333))
        #mid_point_projected2 = Project((self.rotted1 + self.rotted2 + self.rotted3) * FV3(0.33333333) - normal_rotted * FV3(1.5))

        # basic diffuse shading (flat shading in this case to)
        light = dot(sun_dir, self.normal) * 0.5 + 0.5
        # the mid point of the triangle
        mid_point = (self.point3o + self.point2o + self.point1o) * Vec3(0.33333333333, 0.33333333333, 0.33333333333)
        # the direction from the camera to the mid point of the triangle
        rd = normalize(mid_point - cam_pos)
        # the specular lighting
        specular = Specular(0.9, self.normal, rd, sun_dir) * 255
        # the clipped polygon
        clipped = [[point1[0], point1[1]], [point2[0], point2[1]], [point3[0], point3[1]]]
        # returning the information
        return [clipped, (z1 + z2 + z3) * 0.33333333333, clamp(self.color * Vec3(light, light, light) + Vec3(specular, specular, specular), 0, 255)]#, self.normal, mid_point_projected1, mid_point_projected2]  # clamp(self.color * Vec3(light, light, light), 0, 255)


# stores a point and rotates it
class Vert:
    def __init__(self, pos: Vec3):
        self.point = pos
    def GetRotted(self) -> Vec3:
        point = self.point
        rotted = point - cam_pos
        rot = rotMatX.Rot([rotted.x, rotted.z])
        rotted = [rot[0], rotted[1], rot[1]]
        rot = rotMatY.Rot([rotted[1], rotted[2]])
        rotted = Vec3(rotted[0], rot[0], rot[1])
        return rotted


# used for the thread dispersing (in my pyvectors library)
class RenderPolygon:
    def __init__(self, polygon: Polygon) -> None:
        self.polygon = polygon
    def run(self) -> None:
        polygon = self.polygon
        clipped = polygon.GetPolygons()
        for point_set in clipped:
            rendered = polygon.Render(point_set)
            if rendered is not None:
                rendered_polygons.append(rendered)


# specular reflection (part of the shading of the polygons)
def Specular(smoothness: float, normal: Vec3, rd: Vec3, sun_dir: Vec3) -> float:
    try:
        specularAngle = math.acos(dot(normalize(sun_dir - rd), normal))
        specularExponent = specularAngle / (1. - smoothness)
        return math.exp(-specularExponent * specularExponent)
    except ValueError:
        return 0


# ------------ rotation stuff ------------

# a rotation matrix
class RotMat:
    def __init__(self, rot) -> None:
        self.sa = math.sin(rot)
        self.ca = math.cos(rot)
    def Rot(self, p: tuple) -> tuple:
        rottedX = p[1] * self.sa + p[0] * self.ca  # table cos and sin
        rottedY = p[1] * self.ca - p[0] * self.sa
        return (rottedX, rottedY)


# rotates a point
def Rotate(point: Vec3, XZ: RotMat, YZ: RotMat) -> Vec3:
    rotted = XZ.Rot(Vec2(point.x, point.z))
    point = Vec3(rotted[0], point.y, rotted[1])
    rotted = XZ.Rot(Vec2(point.y, point.z))
    point = Vec3(point.x, rotted[0], rotted[1])
    return point


# ------------ ui elements ------------

# ui (just text for now)
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


# ------------ other vector functions ------------

# turns a float into a vector3 with the x, y, z being the equal to the float inputed
def V3(f: float) -> Vec3:
    return Vec3(f, f, f)


# reflects a vector (not being used currently)
def Reflect(rd: Vec3, normal: Vec3) -> Vec3:
    return (rd - (normal * Vec3(2, 2, 2)) * (rd * normal))


def FV3(f: float) -> Vec3:
    return Vec3(f, f, f)


# ------------ main loop and initalization ------------

# creating the screen
res = Vec2(1200, 750)
screen = pygame.display.set_mode(res)

# the sun direction and rotation (rotation not being used)
sun_dir = normalize(Vec3(0.2, 0.7, -0.2))
radians = Vec2(math.atan2(sun_dir.x, sun_dir.z), math.atan2(sun_dir.y, sun_dir.z))

# the frame
frame = 0

# the tone of all the colors and the color of the sky
undertones = [Vec3(138,10,10), Vec3(250,235,215), Vec3(245,245,220), Vec3(255,240,245), Vec3(240,255,240), Vec3(0,47,167), Vec3(255,255,0), Vec3(128,0,0), Vec3(160,82,45)]
undertones = [Vec3(250,235,215), Vec3(128,0,0)]
undertone = undertones[1]
undertone_strength = 0.1
sky_color = Vec3(135,206,250)

# different parts of the player (not being used currently, im going to hook up physics soon)
player_pos = Vec3(0, -10, 0)
cam_pos = player_pos - Vec3(0, 2, 0)
cam_rot = Vec2(0, 0)
velocity = Vec3(0, 0, 0)

forwards_vector = Vec3(0, 0, 0)

grounded = -100
last_jumped = -10

#player_chunk_pos = Vec2(-1200, -1200)

# the rotations (so i only need to do the cos and sin of the camera angle once per frame to save time rendering)
rotMatX = RotMat(cam_rot.x)
rotMatY = RotMat(cam_rot.y)

# creating the projection matrix (4x4 matrix)
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

# creating the polygons being rendered
polygons = []
#"""
verts = []

"""
tri_num = 0

perlin_color = array([21, 21], 'perlin', [[0.9, 1.2, 4, 'add']])
for x in range(-10, 10, 3):
    for z in range(-10, 10, 3):
        height = 0
        tl = Vert(Vec3(x, height, z))
        tr = Vert(Vec3(x + 3, height, z))
        bl = Vert(Vec3(x, height, z + 3))
        br = Vert(Vec3(x + 3, height, z + 3))
        new_verts = [tl, tr, bl, br]
        for vert in new_verts:
            verts.append(vert)
        tl = 0 + tri_num
        tr = 1 + tri_num
        bl = 2 + tri_num
        br = 3 + tri_num
        polygon1 = Polygon(br, tr, tl, normalize(Vec3(0, 1, 0)), Vec3(25, 200, 40) * FV3(perlin_color[x + 10][z + 10]))
        polygon2 = Polygon(tl, bl, br, normalize(Vec3(0, 1, 0)), Vec3(25, 200, 40) * FV3(perlin_color[x + 10][z + 11]))
        polygons.append(polygon1)
        polygons.append(polygon2)
        tri_num += 4

height = 2
for z in [-3, -1.5]:
    x = 2
    tl = Vert(Vec3(x, z, height))
    tr = Vert(Vec3(x + 3, z, height))
    bl = Vert(Vec3(x, z + 1.5, height + 3))
    br = Vert(Vec3(x + 3, z + 1.5, height + 3))
    new_verts = [tl, tr, bl, br]
    for vert in new_verts:
        verts.append(vert)
    tl = 0 + tri_num
    tr = 1 + tri_num
    bl = 2 + tri_num
    br = 3 + tri_num
    polygon1 = Polygon(br, tr, tl, normalize(Vec3(0, 2, -1)), Vec3(25, 200, 40) * FV3(perlin_color[x + 10][height + 10]))
    polygon2 = Polygon(tl, bl, br, normalize(Vec3(0, 2, -1)), Vec3(25, 200, 40) * FV3(perlin_color[x + 10][height + 11]))
    polygons.append(polygon1)
    polygons.append(polygon2)
    tri_num += 4
    height += 3

z = -1
x = 2
height = -3
tl = Vert(Vec3(x, height, z))
tr = Vert(Vec3(x + 3, height, z))
bl = Vert(Vec3(x, height, z + 3))
br = Vert(Vec3(x + 3, height, z + 3))
new_verts = [tl, tr, bl, br]
for vert in new_verts:
    verts.append(vert)
tl = 0 + tri_num
tr = 1 + tri_num
bl = 2 + tri_num
br = 3 + tri_num
polygon1 = Polygon(br, tr, tl, normalize(Vec3(0, 1, 0)), Vec3(25, 200, 40) * FV3(perlin_color[x + 10][z + 10]), tags = {Tags.bouncy: 100})
polygon2 = Polygon(tl, bl, br, normalize(Vec3(0, 1, 0)), Vec3(25, 200, 40) * FV3(perlin_color[x + 10][z + 11]), tags = {Tags.bouncy: 10})
polygons.append(polygon1)
polygons.append(polygon2)
tri_num += 4
#"""


#"""

# reading an object file and converting its data to my data
#"""
polygons = []
data = open('test_map.obj').read().split('\n')

#normals = []
verts = []

tri_number = 0

# object files are farly easy to convert luckily and this took 15 mins
for line in data:
    if len(line) > 2:
        if line[0] == 'v':
            if line[1] == ' ':
                points = line.split(' ')
                verts.append(Vert(Vec3(float(points[1]), float(points[2]), float(points[3])) * Vec3(1, -1, 1)))
            #elif line[1] == 'n':
            #    points = line.split(' ')
            #    normals.append(Vec3(float(points[1]), float(points[2]), float(points[3])))
for line in data:
    if len(line) > 2:
        if line[0] == 'f':
            points = line.split(' ')
            point1 = int(points[1].split('/')[0]) - 1
            point2 = int(points[2].split('/')[0]) - 1
            point3 = int(points[3].split('/')[0]) - 1

            vert1 = verts[point1].point
            vert2 = verts[point2].point
            vert3 = verts[point3].point
            
            line1 = Vec3(None, None, None)
            line2 = Vec3(None, None, None)
            normal = Vec3(None, None, None)

            line1 = vert2 - vert1
            line2 = vert3 - vert1

            normal.x = line1.y * line2.z - line1.z * line2.y
            normal.y = line1.z * line2.x - line1.x * line2.z
            normal.z = line1.x * line2.y - line1.y * line2.x

            polygons.append(Polygon(point1, point2, point3, normalize(Vec3(normal.x, normal.y, normal.z)), Vec3(205,133,63)))
            #tri_number += 1
#"""

# i can render multiple meshes by adding a , then another list of Polygons
polygons = [polygons]

# the bounds of the screen for clipping 2d polygons to
clipping_bounds = [[0, 0], [0, res.y], [res.x, res.y], [res.x, 0]]

# the rendered polygons (for depth sorting followed by rendering)
#projected_verts = []
rendered_polygons = []

# the keys being held/not held
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

last_normal = Vec3(0, 0, 0)

# the delta time (for physics and movement and rotation of camera smoothly)
dt = 0

# if the screen is running
running = True

# the main loop
while running:
    # making the sun rotate around the sky
    #sun_dir = normalize(Vec3(math.sin(frame) * 0.2, math.sin(frame), math.cos(frame)))

    s = time.time()  # the start time
    for event in pygame.event.get():  # looping through the events of this frame
        if event.type == pygame.QUIT:  # checking if the windo has been closed
            running = False
            pygame.quit()
            break
        elif event.type == pygame.KEYDOWN:  # checking for keys being pressed
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
            elif event.key in [ord('1'), ord('2')]:
                number = [0, 1, 2, 3, 4, 5, 6, 7, 8][[ord('1'), ord('2')].index(event.key)]
                undertone = undertones[number]
        elif event.type == pygame.KEYUP:  # checking for keys being released
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
    
    if not running:  # checking if the game is still running
        break
    
    # some different speed things for movement and rotation of the camera
    movement_speed = 50 # * dt
    if grounded <= 0:
        movement_speed = 20
    pi = 3.14159
    rot_speed = pi * dt * 1.25

    F_app = Vec3(0, 0, 0)
    # moving the camera/player
    if held["w"]:
        F_app += Vec3(movement_speed * -math.sin(cam_rot.x), 0, movement_speed * math.cos(cam_rot.x))
    if held["a"]:
        F_app += Vec3(movement_speed * -math.sin(cam_rot.x + pi * 0.5), 0, movement_speed * math.cos(cam_rot.x + pi * 0.5))
    if held["s"]:
        F_app += Vec3(movement_speed * math.sin(cam_rot.x), 0, movement_speed * -math.cos(cam_rot.x))
    if held["d"]:
        F_app += Vec3(movement_speed * -math.sin(cam_rot.x - pi * 0.5), 0, movement_speed * math.cos(cam_rot.x - pi * 0.5))
    #if held["shift"]:
    #    F_app += Vec3(0, movement_speed, 0)
    if held[" "] and grounded >= 0 and last_jumped <= 0:
        F_app += Vec3(0, -225, 0)
        last_jumped = 1
        grounded = -20
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
 
    def IsOnSurface(triangle: Polygon, point: Vec3) -> bool:
        normal = triangle.normal  # the normal
        mid_point = (verts[triangle.point1].point + verts[triangle.point2].point + verts[triangle.point3].point) * FV3(0.33333333)  # the mid point of the triangle
        direction_to_mid_point = normalize(mid_point - point)  # the direction to the center of the triangle (is 90˚ to the surface normal if it is parallel to the polygon)
        perpandicular = abs(dot(normal, direction_to_mid_point)) < 0.00002  # if the direction is 90˚ to the normal then it is parallel to the triangle
        if perpandicular:
            # check if the point is in the bounds of the triangles edges
            return True
        return False

    Fnet = Vec3(0, 0, 0)
    #F_g = Vec3(0, 9.81, 0)  # gravity
    F_f = Vec3(0, 0, 0)
    if grounded <= 0:
        F_f = -Vec3(velocity.x, 0, velocity.z) * FV3(1)

    F_g = Vec3(0, 9.81, 0)  # gravity
    Fnet = Fnet + F_g + F_app + F_f
    a = Fnet

    #if grounded < 0:
    #    a += -velocity * FV3(1)

    grounded -= dt
    last_jumped -= dt

    ray_start = player_pos
    ray_endXZ  = player_pos + FV3(0.5) * Vec3(a.x, 0, a.z) * FV3(pow(dt, 2)) + FV3(dt) * Vec3(velocity.x, 0, velocity.z)  # the end of the ray
    ray_endXYZ = player_pos + FV3(0.5) * a * FV3(pow(dt, 2)) + FV3(dt) * velocity  # the end of the ray

    #movement_direction = normalize(ray_end)
    for mesh in polygons:
        for polygon in mesh:
            intersectionXZ = RayTriangle(verts[polygon.point1].point, verts[polygon.point2].point, verts[polygon.point3].point, ray_start, ray_endXZ)
            intersectionXYZ = RayTriangle(verts[polygon.point1].point, verts[polygon.point2].point, verts[polygon.point3].point, ray_start, ray_endXYZ)
            #on_surface = IsOnSurface(polygon, ray_start)
            # check if the point is right on the surface (the intersection function only works when the point isnt on the surface)
            if intersectionXZ is not False or intersectionXYZ is not False:# or on_surface:
                #dst_to_surface = length(intersection) - 0.000001  # the intersection function dosent work on the surface of an object so im moving it back a bit
                #collition_point = movement_direction * Vec3(dst_to_surface, dst_to_surface, dst_to_surface)
                surface_normal = polygon.normal  # * Vec3(1, -1, 1)
                last_normal = surface_normal

                v_dot_n = dot(surface_normal, velocity)
                velocity_n = velocity - surface_normal * FV3(v_dot_n)

                a_dot_n = dot(surface_normal, a)
                na = a - surface_normal * FV3(a_dot_n)
                
                mewK = 0.4
                mewS = 0.55
                normal_force = a - na
                dir = normalize(-Vec3(velocity.x - F_app.x * dt, 0, velocity.z - F_app.z * dt))
                f_max = mewS * length(normal_force)
                lv = length(Vec2(velocity.x - F_app.x * dt, velocity.z - F_app.z * dt))

                F_sliding = lv / dt# + length(Vec2(na.x - F_app.x, na.z - F_app.z))
                if F_sliding < f_max:
                    F_f = FV3(F_sliding) * dir
                else:
                    F_f = FV3(F_sliding - (mewK * length(normal_force))) * dir
                
                velocity = velocity_n

                a = a + F_f

                a_dot_n = dot(surface_normal, a)
                a = a - surface_normal * FV3(a_dot_n)

                if dot(Vec3(0, 1, 0), surface_normal) >= 0.3:
                    grounded = 0.4
                
                #if Tags.bouncy in polygon.tags:
                #    length_of_vector_in_this_direction = 10
                #    a += surface_normal * Vec3(1, -1, 1) * FV3(polygon.tags[Tags.bouncy] * length_of_vector_in_this_direction)
                
                ray_endXZ  = player_pos + FV3(0.5) * Vec3(a.x, 0, a.z) * FV3(pow(dt, 2)) + FV3(dt) * Vec3(velocity.x, 0, velocity.z)  # the end of the ray
                ray_endXYZ = player_pos + FV3(0.5) * a * FV3(pow(dt, 2)) + FV3(dt) * velocity  # the end of the ray
    
    velocity = Vec3(pyMath.clamp(velocity.x, -5, 5), velocity.y, pyMath.clamp(velocity.z, -5, 5))    
    # changing the position and velocity
    
    player_pos += FV3(0.5) * a * FV3(pow(dt, 2)) + FV3(dt) * velocity
    velocity += a * Vec3(dt, dt, dt)
    #player_pos += F_app * Vec3(dt, dt, dt)
    cam_pos = player_pos - Vec3(0, 2, 0)

    if player_pos.y > 100:
        player_pos = Vec3(0, -10, 0)
        player_velocity = Vec3(0, 0, 0)

    # creating the rotation matricies
    rotMatX = RotMat(cam_rot.x)
    rotMatY = RotMat(cam_rot.y)

    # some different things for 3d clipping (not my own)
    vUp = Vec3(0, 1, 0)
    vTarget = Vec3(0, 0, 1)
    matCameraRot = Matrix_MakeRotationY(cam_rot.y)
    vLookDir = Matrix_MultiplyVector(matCameraRot, vTarget)
    vTarget = cam_pos + vLookDir
    matCamera = Matrix_PointAt(cam_pos, vTarget, vUp)

    matView = Matrix_QuickInverse(matCamera)

    forwards_vector = Rotate(Vec3(0, 0, -1), rotMatX, rotMatY)
    radians = Vec2(math.atan2(sun_dir.x - forwards_vector.x, sun_dir.z - forwards_vector.z), math.atan2(sun_dir.y - forwards_vector.y, sun_dir.z - forwards_vector.z))
    
    # filling the screen with a color (creating a sky box)
    v = fov / 360 * 3.14159 / 750
    for y in range(0, 750, 1):
        height = math.sin(cam_rot.y + (750 - y) * v - 1) * 0.3 + 0.7
        n_color = mix(sky_color * Vec3(height, height, height), undertone, undertone_strength)
        pygame.draw.rect(screen, n_color, [0, y, 1200, 1])
    #screen.fill(mix(sky_color, undertone, undertone_strength))

    # setting up some variables
    renderings = []
    rendered_polygons = []

    # rotating the verticies
    rotted_verts = []
    #projected_verts = []
    for vert in verts:
        rotted = vert.GetRotted()
        rotted_verts.append(rotted)
        #projected = vert.Project(rotted)
        #projected_verts.append(projected)

    # setting the verticies of the polygons to the rotated ones
    for polygonChunks in polygons:
        for polygon_ in polygonChunks:
            polygon_.GetVerts(rotted_verts)
            #if min([polygon_.rotted1.z, polygon_.rotted2.z, polygon_.rotted3.z]) > 1:
            renderings.append(RenderPolygon(polygon_))

    # rendering the polygons
    PyThreading.Disperse(renderings, max_threads = 5)
    
    # doing a depth based sort on the polygons
    rendered_polygons.sort(key = lambda key: -key[1])

    # rendering the polygons and clipping them to the screen (in 2d)
    for polygon in rendered_polygons:
        try:
            polygon_clipped = clip(clipping_bounds, polygon[0])
        except IndexError:
            continue
        if len(polygon_clipped) >= 3:
            pygame.draw.polygon(screen, polygon[2], polygon_clipped)
            #normal = polygon[3]
            #mid_point1 = polygon[4]
            #mid_point2 = polygon[5]
            #pygame.draw.line(screen, (0, 0, 0), [mid_point1[0][0], mid_point1[0][1]], [mid_point2[0][0], mid_point2[0][1]])
    
    # rendering the fps
    UI.text(f'FPS: {round(1 / max(dt, 0.00000001))}', (0, 0, 40), (10, 30), 35)
    UI.text(f'Pos: {round(player_pos.x, 1)}, {round(player_pos.y, 1)}, {round(player_pos.z, 1)}', (0, 0, 40), (10, 10), 35)

    # updating the display
    pygame.display.update()

    # increasing the frame count by dt to make it smooth
    frame += dt

    e = time.time()  # the end time
    dt = e - s  # find dt (delta time or end_time - start_time or the change in time)

