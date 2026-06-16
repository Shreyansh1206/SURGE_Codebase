
from __future__ import annotations

import os
import random

import numpy as np
import pygame
from pygame import RLEACCEL

GAME_DIR = os.path.dirname(os.path.abspath(__file__))
SPRITE_DIR = os.path.join(GAME_DIR, "sprites")

WIDTH, HEIGHT = 600, 150
FPS = 60
GRAVITY = 0.6
GROUND_Y = int(0.98 * HEIGHT)
DINO_X = WIDTH / 15.0


def _sprite_path(name: str) -> str:
    return os.path.join(SPRITE_DIR, name)


def load_image(name, sizex=-1, sizey=-1, colorkey=None):
    image = pygame.image.load(_sprite_path(name))
    image = image.convert()
    if colorkey is not None:
        if colorkey == -1:
            colorkey = image.get_at((0, 0))
        image.set_colorkey(colorkey, RLEACCEL)
    if sizex != -1 or sizey != -1:
        image = pygame.transform.scale(image, (sizex, sizey))
    return image, image.get_rect()


def load_sprite_sheet(sheetname, nx, ny, scalex=-1, scaley=-1, colorkey=None):
    sheet = pygame.image.load(_sprite_path(sheetname))
    sheet = sheet.convert()
    sheet_rect = sheet.get_rect()
    sprites = []
    sizex = sheet_rect.width // nx
    sizey = sheet_rect.height // ny
    for i in range(ny):
        for j in range(nx):
            rect = pygame.Rect((j * sizex, i * sizey, sizex, sizey))
            image = pygame.Surface(rect.size).convert()
            image.blit(sheet, (0, 0), rect)
            if colorkey is not None:
                if colorkey == -1:
                    colorkey = image.get_at((0, 0))
                image.set_colorkey(colorkey, RLEACCEL)
            if scalex != -1 or scaley != -1:
                image = pygame.transform.scale(image, (scalex, scaley))
            sprites.append(image)
    return sprites, sprites[0].get_rect()


class Dino(pygame.sprite.Sprite):
    def __init__(self, sizex=-1, sizey=-1):
        super().__init__()
        self.images, self.rect = load_sprite_sheet("dino.png", 5, 1, sizex, sizey, -1)
        self.images1, self.rect1 = load_sprite_sheet(
            "dino_ducking.png", 2, 1, 59, sizey, -1
        )
        self.rect.bottom = GROUND_Y
        self.rect.left = WIDTH / 15
        self.image = self.images[0]
        self.index = 0
        self.counter = 0
        self.score = 0
        self.isJumping = False
        self.isDead = False
        self.isDucking = False
        self.isBlinking = False
        self.movement = [0, 0]
        self.jumpSpeed = 11.5
        self.stand_pos_width = self.rect.width
        self.duck_pos_width = self.rect1.width

    def checkbounds(self):
        if self.rect.bottom > GROUND_Y:
            self.rect.bottom = GROUND_Y
            self.isJumping = False

    def update(self):
        if self.isJumping:
            self.movement[1] = self.movement[1] + GRAVITY
        if self.isJumping:
            self.index = 0
        elif self.isDucking:
            if self.counter % 5 == 0:
                self.index = (self.index + 1) % 2
        else:
            if self.counter % 5 == 0:
                self.index = (self.index + 1) % 2 + 2
        if self.isDead:
            self.index = 4
        if not self.isDucking:
            self.image = self.images[self.index]
            self.rect.width = self.stand_pos_width
        else:
            self.image = self.images1[self.index % 2]
            self.rect.width = self.duck_pos_width
        self.rect = self.rect.move(self.movement)
        self.checkbounds()
        if not self.isDead and self.counter % 7 == 6:
            self.score += 1
        self.counter += 1


class Cactus(pygame.sprite.Sprite):
    def __init__(self, speed=5, sizex=-1, sizey=-1):
        super().__init__()
        self.images, self.rect = load_sprite_sheet(
            "cacti-small.png", 3, 1, sizex, sizey, -1
        )
        self.rect.bottom = GROUND_Y
        self.rect.left = WIDTH + self.rect.width
        self.image = self.images[random.randrange(0, 3)]
        self.movement = [-1 * speed, 0]
        self.kind = "CACTUS"

    def update(self):
        self.rect = self.rect.move(self.movement)
        if self.rect.right < 0:
            self.kill()


class Ptera(pygame.sprite.Sprite):
    def __init__(self, speed=5, sizex=-1, sizey=-1):
        super().__init__()
        self.images, self.rect = load_sprite_sheet("ptera.png", 2, 1, sizex, sizey, -1)
        heights = [int(HEIGHT * 0.82), int(HEIGHT * 0.75), int(HEIGHT * 0.60)]
        self.rect.centery = heights[random.randrange(0, 3)]
        self.rect.left = WIDTH + self.rect.width
        self.image = self.images[0]
        self.movement = [-1 * speed, 0]
        self.index = 0
        self.counter = 0
        self.kind = "PTERA"

    def update(self):
        if self.counter % 10 == 0:
            self.index = (self.index + 1) % 2
        self.image = self.images[self.index]
        self.rect = self.rect.move(self.movement)
        self.counter += 1
        if self.rect.right < 0:
            self.kill()


class Ground:
    def __init__(self, speed=-5):
        self.image, self.rect = load_image("ground.png", -1, -1, -1)
        self.image1, self.rect1 = load_image("ground.png", -1, -1, -1)
        self.rect.bottom = HEIGHT
        self.rect1.bottom = HEIGHT
        self.rect1.left = self.rect.right
        self.speed = speed

    def draw(self, surface):
        surface.blit(self.image, self.rect)
        surface.blit(self.image1, self.rect1)

    def update(self):
        self.rect.left += self.speed
        self.rect1.left += self.speed
        if self.rect.right < 0:
            self.rect.left = self.rect1.right
        if self.rect1.right < 0:
            self.rect1.left = self.rect.right


class Cloud(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.image, self.rect = load_image("cloud.png", int(90 * 30 / 42), 30, -1)
        self.speed = 1
        self.rect.left = x
        self.rect.top = y
        self.movement = [-1 * self.speed, 0]

    def update(self):
        self.rect = self.rect.move(self.movement)
        if self.rect.right < 0:
            self.kill()


class DinoGameEngine:

    def __init__(self, render: bool = False, seed: int | None = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        if not pygame.get_init():
            if not render:
                os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            pygame.init()
            try:
                pygame.mixer.init()
            except pygame.error:
                pass
        self.render = render
        if render:
            surf = pygame.display.get_surface()
            if surf is None or surf.get_size() != (WIDTH, HEIGHT):
                self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            else:
                self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Dino Run")
        elif pygame.display.get_surface() is None:
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        else:
            self.screen = pygame.display.get_surface()
        self.clock = pygame.time.Clock()
        self._new_episode()

    def _new_episode(self):
        self.gamespeed = 4
        self.player = Dino(44, 47)
        self.ground = Ground(-1 * self.gamespeed)
        self.cacti = pygame.sprite.Group()
        self.pteras = pygame.sprite.Group()
        self.clouds = pygame.sprite.Group()
        self.last_obstacle = pygame.sprite.Group()
        self.counter = 0
        self.prev_score = 0
        self._prev_nearest_x = None

    def reset(self):
        self._new_episode()
        return self.get_state()

    def _nearest_obstacles(self):
        obstacles = list(self.cacti) + list(self.pteras)
        obstacles.sort(key=lambda o: o.rect.left)
        o1 = obstacles[0] if len(obstacles) > 0 else None
        o2 = obstacles[1] if len(obstacles) > 1 else None
        return o1, o2

    def _pack_obstacle(self, obs):
        if obs is None:
            return [9999.0, 0.0, 0.0, 0.0], ""
        return (
            [
                float(obs.rect.left),
                float(obs.rect.top),
                float(obs.rect.width),
                float(obs.rect.height),
            ],
            getattr(obs, "kind", ""),
        )

    def get_state(self):
        o1, o2 = self._nearest_obstacles()
        o1_vals, o1_type = self._pack_obstacle(o1)
        o2_vals, o2_type = self._pack_obstacle(o2)
        return {
            "crashed": self.player.isDead,
            "playing": not self.player.isDead,
            "score": int(self.player.score),
            "speed": float(self.gamespeed),
            "dinoY": float(self.player.rect.top),
            "jumping": 1 if self.player.isJumping else 0,
            "ducking": 1 if self.player.isDucking else 0,
            "o1": o1_vals,
            "o2": o2_vals,
            "o1type": o1_type,
            "o2type": o2_type,
        }

    def _apply_action(self, action: int):
        if action == 0:
            self.player.isDucking = False
        elif action == 1:
            self.player.isDucking = False
            if self.player.rect.bottom == GROUND_Y:
                self.player.isJumping = True
                self.player.movement[1] = -1 * self.player.jumpSpeed
        elif action == 2:
            if not (self.player.isJumping and self.player.isDead):
                self.player.isDucking = True

    def _spawn_obstacles(self):
        if len(self.cacti) < 2:
            if len(self.cacti) == 0:
                self.last_obstacle.empty()
                cactus = Cactus(self.gamespeed, 40, 40)
                self.cacti.add(cactus)
                self.last_obstacle.add(cactus)
            else:
                for last in self.last_obstacle:
                    if last.rect.right < WIDTH * 0.7 and random.randrange(0, 50) == 10:
                        self.last_obstacle.empty()
                        cactus = Cactus(self.gamespeed, 40, 40)
                        self.cacti.add(cactus)
                        self.last_obstacle.add(cactus)

        if len(self.pteras) == 0 and random.randrange(0, 200) == 10 and self.counter > 500:
            for last in self.last_obstacle:
                if last.rect.right < WIDTH * 0.8:
                    self.last_obstacle.empty()
                    ptera = Ptera(self.gamespeed, 46, 40)
                    self.pteras.add(ptera)
                    self.last_obstacle.add(ptera)

        if len(self.clouds) < 5 and random.randrange(0, 300) == 10:
            cloud = Cloud(WIDTH, random.randrange(HEIGHT // 5, HEIGHT // 2))
            self.clouds.add(cloud)

    def refresh_display(self):
        if not self.render:
            return
        if not pygame.get_init():
            pygame.init()
        want = (WIDTH, HEIGHT)
        surf = pygame.display.get_surface()
        if surf is None or surf.get_size() != want:
            self.screen = pygame.display.set_mode(want)
        else:
            self.screen = surf
        pygame.display.set_caption("Dino Run - Training")

    def _maybe_render(self, throttle: bool = True):
        if not self.render:
            return
        if self.screen is None:
            self.refresh_display()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.player.isDead = True
        self.screen.fill((235, 235, 235))
        self.ground.draw(self.screen)
        self.clouds.draw(self.screen)
        self.cacti.draw(self.screen)
        self.pteras.draw(self.screen)
        self.screen.blit(self.player.image, self.player.rect)
        pygame.display.flip()
        if throttle:
            self.clock.tick(FPS)

    def step(self, action: int, *, draw: bool | None = None, throttle: bool = True):
        self._apply_action(action)

        for cactus in self.cacti:
            cactus.movement[0] = -1 * self.gamespeed
            if pygame.sprite.collide_mask(self.player, cactus):
                self.player.isDead = True

        for ptera in self.pteras:
            ptera.movement[0] = -1 * self.gamespeed
            if pygame.sprite.collide_mask(self.player, ptera):
                self.player.isDead = True

        self._spawn_obstacles()
        self.player.update()
        self.cacti.update()
        self.pteras.update()
        self.clouds.update()
        self.ground.update()

        if self.counter % 700 == 699:
            self.ground.speed -= 1
            self.gamespeed += 1
        self.counter += 1

        state = self.get_state()
        done = bool(self.player.isDead)
        info = {
            "score": state["score"],
            "speed": state["speed"],
            "o1_type": state["o1type"],
            "o2_type": state["o2type"],
        }
        should_draw = self.render if draw is None else draw
        if should_draw:
            self._maybe_render(throttle=throttle)
        return state, done, info

    def close(self):
        pass
