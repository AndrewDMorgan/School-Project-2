import math, time, pygame, threading, json, pyclipper
from typing import List, Union

pygame.init()

EXIT_LOADING = threading.Event()


class UI:
    def text(screen: pygame.Surface, text: str, color: tuple, pos: tuple, size: float, center: bool = False, font: str = 'pixel.ttf'):
        size = math.floor(size / 1.5)
        largeText = pygame.font.Font(font, size)
        textSurface = largeText.render(text, True, color)
        TextSurf, TextRect = textSurface, textSurface.get_rect()
        if center:
            TextRect.center = pos
            sprite = screen.blit(TextSurf, TextRect)
        else:
            sprite = screen.blit(TextSurf, pos)
        return sprite 


class Vector2:  # a 2d vector or 2x1 matrix
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y
    def Dot(self, other: object) -> float:
        return self.x * other.x + self.y * other.y
    def Length(self) -> float:
        return math.sqrt(self.Dot(self))
    def Normalize(self) -> object:
        l = self.Length()
        if l == 0:
            return Vector2(0, 0)
        return Vector2(self.x / l, self.y / l)


class Vector3:  # a 3d vector or 3x1 matrix
    def __init__(self, x: float, y: float, z: float) -> None:
        self.x = x
        self.y = y
        self.z = z
    def Dot(self, vector: object) -> float:
        return self.x * vector.x + self.y * vector.y + self.z * vector.z
    def Length(self) -> float:
        return math.sqrt(self.Dot(self))
    def Normalize(self) -> object:
        l = self.Length()
        if l == 0:
            return Vector3(0, 0, 0)
        return Vector3(self.x / l, self.y / l, self.z / l)
    def __sub__(self, other: object) -> object:
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)
    def __add__(self, other: object) -> object:
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)
    def __mul__(self, other: object) -> object:
        return Vector3(self.x * other.x, self.y * other.y, self.z * other.z)
    def __str__(self) -> str:
        return f'({self.x}, {self.y}, {self.z})'


class Mat2x2:  # 2 by 2 matrix
    def __init__(self, tl: float, tr: float, bl: float, br: float) -> None:
        self.mat = [[tl, tr], [bl, br]]
    def MultVector2(self, vector: Vector2) -> Vector2:  # multiplies a 2x1 or 2d vector by the 2x2 matrix
        return Vector2(self.mat[0][0] * vector.y + self.mat[0][1] * vector.x, self.mat[1][0]  * vector.x + self.mat[1][1] * vector.y)


class RotationMat (Mat2x2):  # rotation matrix (2x2 matrix)
    def __init__(self, angle: float):
        ca = math.cos(angle)
        sa = math.sin(angle)
        self.ca = ca
        self.sa = sa
    def MultVector2(self, p: Vector2) -> Vector2:
        rottedX = p.y * self.sa + p.x * self.ca
        rottedY = p.y * self.ca - p.x * self.sa
        return Vector2(rottedX, rottedY)


class Mat4x4:
    def __init__(self, t1, t2, t3, t4, mt1, mt2, mt3, mt4, mb1, mb2, mb3, mb4, b1, b2, b3, b4) -> None:
        self.mat = [[t1, t2, t3, t4], [mt1, mt2, mt3, mt4], [mb1, mb2, mb3, mb4], [b1, b2, b3, b4]]
    def MultVector3(self, vector: Vector3) -> Vector3:
        p = [vector.x, vector.y, vector.z]

        o = [0, 0, 0]
        o[0] = p[0] * self.mat[0][0] + p[1] * self.mat[1][0] + p[2] * self.mat[2][0] + self.mat[3][0]
        o[1] = p[0] * self.mat[0][1] + p[1] * self.mat[1][1] + p[2] * self.mat[2][1] + self.mat[3][1]
        o[2] = p[0] * self.mat[0][2] + p[1] * self.mat[1][2] + p[2] * self.mat[2][2] + self.mat[3][2]
        w    = p[0] * self.mat[0][3] + p[1] * self.mat[1][3] + p[2] * self.mat[2][3] + self.mat[3][3]
        
        if w != 0:
            o[0] /= w
            o[1] /= w
            o[2] /= w
        
        projected = ((o[0] + 1) * 600, (o[1] + 1) * 375)

        return Vector3(projected[0], projected[1], vector.z)


fNear = 0.1
fFar = 1000
fFov = 90
fAspectRatio = 750 / 1200
fFovRad = 1.0 / math.tan(fFov * 0.5 / 180 * 3.14159)

projectionMatrix = Mat4x4(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
projectionMatrix.mat[0][0] = fAspectRatio * fFovRad
projectionMatrix.mat[1][1] = fFovRad
projectionMatrix.mat[2][2] = fFar / (fFar - fNear)
projectionMatrix.mat[3][2] = (-fFar * fNear) / (fFar - fNear)
projectionMatrix.mat[2][3] = 1
projectionMatrix.mat[3][3] = 0

half_res = Vector2(600, 375)


def Project(point: Vector3) -> Vector3:  # screen x, screen y, depth
    projected = projectionMatrix.MultVector3(point)
    return projected


def Rotate(point: Vector3, cp: Vector3, rotXZ: RotationMat, rotYZ: RotationMat, rotXY: RotationMat) -> Vector3:
    point = point - cp
    
    rotted = rotXZ.MultVector2(Vector2(point.x, point.z))
    point = Vector3(rotted.x, point.y, rotted.y)
    rotted = rotYZ.MultVector2(Vector2(point.y, point.z))
    point = Vector3(point.x, rotted.x, rotted.y)
    rotted = rotXY.MultVector2(Vector2(point.x, point.y))
    point = Vector3(rotted.x, rotted.y, point.z)

    return point


def LoadingScreen(screen: pygame.Surface) -> None:
    frame = 0
    while True:
        s = time.time()
        if EXIT_LOADING.is_set():  # exiting the loading screen
            return None
        
        pygame.event.get()

        screen.fill((225, 225, 255))

        #UI.text(screen, f"Loading{'.' * (frame % 4)}", (0, 0, 15), (600, 350), 35, centered = True)  # rendering the loading screen text

        pygame.display.update()

        frame += 1

        e = time.time()
        time.sleep(max(0.25 - (e - s), 0))  # waiting 0.25 seconds per frame


def saturate(v: float) -> float:
    return max(min(v, 255), 0)


class Polygon:
    def __init__(self, index1: Vector3, index2: Vector3, index3: Vector3, normal: Vector3, color: Vector3) -> None:
        self.index1 = index1
        self.index2 = index2
        self.index3 = index3
        self.normal = normal
        self.color = color
    def Project(self, cp: Vector3, rotXZ: RotationMat, rotYZ: RotationMat, rotXY: RotationMat) -> None:
        point1 = Rotate(self.index1, cp, rotXZ, rotYZ, rotXY)
        point2 = Rotate(self.index2, cp, rotXZ, rotYZ, rotXY)
        point3 = Rotate(self.index3, cp, rotXZ, rotYZ, rotXY)

        self.projected1 = Project(point1)
        self.projected2 = Project(point2)
        self.projected3 = Project(point3)


class LevelData:
    def __init__(self, polygons: List[Polygon]) -> None:
        self.polygons = polygons


def LoadLevel(level_number: int, screen: pygame.Surface) -> LevelData:
    #EXIT_LOADING = threading.Event()

    #loading_screen = threading.Thread(target=LoadingScreen, args=(screen, ))
    #loading_screen.daemon = True
    #loading_screen.start()

    # loading the level
    data = json.load(open('levels.json'))
    mapData = data["mapData"][f'{level_number}']

    indecies = mapData["indecies"]
    for i in range(len(indecies)):
        indecies[i] = Vector3(indecies[i][0], indecies[i][1], indecies[i][2])
    colors = mapData["colors"]
    normals = mapData["normals"]
    polygons = mapData["polygons"]
    for i in range(len(polygons)):
        polygon = polygons[i]
        normal = Vector3(normals[i][0], normals[i][1], normals[i][2]).Normalize()
        polygons[i] = Polygon(indecies[polygon[0]], indecies[polygon[1]], indecies[polygon[2]], normal, Vector3(colors[i][0], colors[i][1], colors[i][2]))

    levelData = LevelData(polygons)

    #EXIT_LOADING.set()
    #loading_screen.join()

    return levelData


level = 0

cam_rot = Vector3(0, 0, 0)
cam_pos = Vector3(0, 4, -2)

sun_dir = Vector3(0.2, 0.7, -0.2).Normalize()

dt = 0
running = True
res = Vector2(1200, 750)
screen = pygame.display.set_mode((1200, 750))

held = {
    "w": False,
    "a": False,
    "s": False,
    "d": False,
    "space": False,
    "left": False,
    "right": False,
    "up": False,
    "down": False,
    "shift": False,
}

level_data = LoadLevel(1, screen)
polygons = level_data.polygons

while running:
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
                held["space"] = True
            elif event.key == pygame.K_LEFT:
                held["left"] = True
            elif event.key == pygame.K_RIGHT:
                held["right"] = True
            elif event.key == pygame.K_UP:
                held["up"] = True
            elif event.key == pygame.K_DOWN:
                held["down"] = True
            elif event.key == pygame.K_LSHIFT:
                held["shift"] = True
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
                held["space"] = False
            elif event.key == pygame.K_LEFT:
                held["left"] = False
            elif event.key == pygame.K_RIGHT:
                held["right"] = False
            elif event.key == pygame.K_UP:
                held["up"] = False
            elif event.key == pygame.K_DOWN:
                held["down"] = False
            elif event.key == pygame.K_LSHIFT:
                held["shift"] = False

    if not running:
        break
    
    rot_speed = 3.14159 * dt
    half_pi = 3.14159 * 0.5
    speed = 10 * dt
    if held["w"]:
        cam_pos = Vector3(cam_pos.x - math.sin(cam_rot.x) * speed, cam_pos.y, cam_pos.z + math.cos(cam_rot.x) * speed)
    if held["a"]:
        cam_pos = Vector3(cam_pos.x - math.sin(cam_rot.x + half_pi) * speed, cam_pos.y, cam_pos.z + math.cos(cam_rot.x + half_pi) * speed)
    if held["s"]:
        cam_pos = Vector3(cam_pos.x + math.sin(cam_rot.x) * speed, cam_pos.y, cam_pos.z - math.cos(cam_rot.x) * speed)
    if held["d"]:
        cam_pos = Vector3(cam_pos.x - math.sin(cam_rot.x - half_pi) * speed, cam_pos.y, cam_pos.z + math.cos(cam_rot.x - half_pi) * speed)
    if held["shift"]:
        cam_pos = Vector3(cam_pos.x, cam_pos.y + speed, cam_pos.z)
    if held["space"]:
        cam_pos = Vector3(cam_pos.x, cam_pos.y - speed, cam_pos.z)
    if held["left"]:
        cam_rot = Vector3(cam_rot.x + rot_speed, cam_rot.y, cam_rot.z)
    if held["right"]:
        cam_rot = Vector3(cam_rot.x - rot_speed, cam_rot.y, cam_rot.z)
    if held["up"]:
        cam_rot = Vector3(cam_rot.x, cam_rot.y + rot_speed, cam_rot.z)
    if held["down"]:
        cam_rot = Vector3(cam_rot.x, cam_rot.y - rot_speed, cam_rot.z)
    
    screen.fill((225, 225, 255))

    # the rotation matricies only need to be made once per time the camera angle changes (instead of being done for each vertex)
    rot_matXZ = RotationMat(cam_rot.x)  # left right | yaw
    rot_matYZ = RotationMat(cam_rot.y)  # up down    | pitch
    rot_matXY = RotationMat(cam_rot.z)  # roll       | roll

    projected_polygons = []

    for polygon in polygons:
        polygon.Project(cam_pos, rot_matXZ, rot_matYZ, rot_matXY)
        if max([polygon.projected1.z, polygon.projected2.z, polygon.projected3.z]) >= 0:
            projected_polygons.append(polygon)
    
    projected_polygons.sort(key = lambda key: -key.projected1.z)  # doing a depth sort and then rendering back to front

    for polygon in projected_polygons:
        color = polygon.color
        #light = polygon.normal.Dot(sun_dir) * 0.5 + 0.5  # adding basic diffuse shading (not smooth)
        light = 1
        color = [saturate(color.x * light), saturate(color.y * light), saturate(color.z * light)]
        pygame.draw.polygon(screen, color, [[polygon.projected1.x, polygon.projected1.y], [polygon.projected2.x, polygon.projected2.y], [polygon.projected3.x, polygon.projected3.y]])
    
    UI.text(screen, f'FPS: {round(1 / max(dt, 0.001))}', (0, 0, 25), (10, 10), 35)
    UI.text(screen, f'POS: {round(cam_pos.x)}, {round(cam_pos.y)}, {round(cam_pos.z)}', (0, 0, 25), (10, 25), 35)
    
    pygame.display.update()

    e = time.time()
    dt = e - s

