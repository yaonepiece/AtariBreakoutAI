import pygame, sys, time, colors
from math import cos, sin, pi
import numpy as np
from sklearn.preprocessing import normalize as norm


class Atarigame:
    def __init__(self):
        # basic datas
        self.windowWidth = 400
        self.windowHeight = 300
        self.runspeed = 1
        self.fps = 50
        self.fpsClock = pygame.time.Clock()
        self.brick = np.ones((5, 8), dtype=int)
        self.brickRect = [[pygame.Rect(50 * j, 20 * i + 40, 50, 20) for j in range(8)] for i in range(5)]
        self.brickstatus = np.zeros((8), dtype=float)
        self.ballPos = np.array([200.0, 200.0])
        self.ballSpeed = 5
        self.ballVector = norm([[-1, -1]]).ravel()
        self.ballSize = 8
        self.barLength = 50
        self.barWidth = 10
        self.barinitX, self.barinitY = 200.0, 295.0
        self.barRect = pygame.Rect(self.barinitX - self.barLength // 2, self.barinitY - self.barWidth // 2,
                                   self.barLength, self.barWidth)
        self.seconds = 0
        self.ticks = 0
        self.barSpeed = 8
        self.barMove = 0
        self.start = True
        self.end = False

        self.ballrand()

    def getBrick(self):
        for i in range(8):
            self.brickstatus[i] = 0.
        for j in range(8):
            for i in range(4, -1, -1):
                if self.brick[i][j]:
                    self.brickstatus[j] = (i + 1) * 0.2
                    break

    def die(self, surface):
        self.draw(surface)
        fontsize = 50
        font = pygame.font.SysFont(None, fontsize)

        text = font.render('You lose!', 1, colors.White)
        textRect = text.get_rect()  # Set the center of the text box
        textRect.centerx = surface.get_rect().centerx
        textRect.centery = surface.get_rect().centery
        surface.blit(text, textRect)
        pygame.display.update()

    def gameclear(self, surface):
        self.draw(surface)
        fontsize = 50
        font = pygame.font.SysFont(None, fontsize)

        text = font.render('You win!', 1, colors.White)
        textRect = text.get_rect()  # Set the center of the text box
        textRect.centerx = surface.get_rect().centerx
        textRect.centery = surface.get_rect().centery
        surface.blit(text, textRect)
        pygame.display.update()

    def timesup(self, surface):
        self.draw(surface)
        fontsize = 50
        font = pygame.font.SysFont(None, fontsize)

        text = font.render('Time\'s up!', 1, colors.White)
        textRect = text.get_rect()  # Set the center of the text box
        textRect.centerx = surface.get_rect().centerx
        textRect.centery = surface.get_rect().centery
        surface.blit(text, textRect)
        pygame.display.update()

    def restart(self):
        self.brick = np.ones((5, 8), dtype=int)
        self.ballPos = np.array([200.0, 200.0])
        self.ballrand()
        self.barRect = pygame.Rect(self.barinitX - self.barLength // 2, self.barinitY - self.barWidth // 2,
                                   self.barLength, self.barWidth)
        self.seconds = 0
        self.ticks = 0
        self.start = True
        self.end = False

    def draw(self, surface):
        surface.fill(colors.Black)
        pygame.draw.circle(surface, colors.Red, tuple(self.ballPos.astype(int)), self.ballSize)
        for i in range(5):
            for j in range(8):
                if self.brick[i][j] == 1:
                    if (i + j) % 2 == 0:
                        rectcolor = colors.Orange
                    else:
                        rectcolor = colors.Blue
                    pygame.draw.rect(surface, rectcolor, self.brickRect[i][j])
        pygame.draw.rect(surface, colors.White, self.barRect)
        pygame.display.update()

    def startpause(self, surface):
        fontsize = 50
        font = pygame.font.SysFont(None, fontsize)

        self.draw(surface)
        text = font.render('Ready?', 1, colors.White)
        textRect = text.get_rect()  # Set the center of the text box
        textRect.centerx = surface.get_rect().centerx
        textRect.centery = surface.get_rect().centery
        surface.blit(text, textRect)
        pygame.display.update()
        time.sleep(0.5)

    def ballrand(self):
        angle = (np.random.rand() * 0.5 + 1.27) * pi
        self.ballVector = np.array([cos(angle), sin(angle)])

    def ballMove(self):
        # move
        prePos = np.copy(self.ballPos)
        self.ballPos += self.ballSpeed * self.ballVector

        # touchwall
        if self.ballPos[0] <= self.ballSize:
            self.ballPos[0] = float(self.ballSize)
            self.ballVector[0] *= -1
        elif self.ballPos[0] >= (400.0 - self.ballSize):
            self.ballPos[0] = 400.0 - self.ballSize
            self.ballVector[0] *= -1
        if self.ballPos[1] <= self.ballSize:
            self.ballPos[1] = float(self.ballSize)
            self.ballVector[1] *= -1
        elif self.ballPos[1] >= (300.0 - self.ballSize):
            return True

        # breakbrick
        for i in range(5):
            for j in range(8):
                if self.brick[i][j] == 1:
                    # touch from right
                    if self.ballPos[0] <= (j * 50 + self.ballSize + 50) and self.ballPos[0] > (
                                j * 50 - self.ballSize + 50) and prePos[0] > self.ballPos[0] and self.ballPos[1] >= (
                            20 * i + 40) and self.ballPos[1] < (20 * i + 60):
                        self.ballVector[0] *= -1
                        self.brick[i][j] = 0
                    # touch from left
                    if self.ballPos[0] >= (j * 50 - self.ballSize) and self.ballPos[0] < (j * 50 + self.ballSize) and \
                                    prePos[0] < self.ballPos[0] and self.ballPos[1] >= (20 * i + 40) and self.ballPos[
                        1] < (20 * i + 60):
                        self.ballVector[0] *= -1
                        self.brick[i][j] = 0
                    # touch from bottom
                    if self.ballPos[1] <= (i * 20 + self.ballSize + 60) and self.ballPos[1] > (
                                i * 20 - self.ballSize + 60) and prePos[1] > self.ballPos[1] and self.ballPos[0] >= (
                        50 * j) and self.ballPos[0] < (50 * j + 50):
                        self.ballVector[1] *= -1
                        self.brick[i][j] = 0
                    # touch from top
                    if self.ballPos[1] >= (i * 20 - self.ballSize + 40) and self.ballPos[1] < (
                                i * 20 + self.ballSize + 40) and prePos[1] < self.ballPos[1] and self.ballPos[0] >= (
                        50 * j) and self.ballPos[0] < (50 * j + 50):
                        self.ballVector[1] *= -1
                        self.brick[i][j] = 0

        tmpVector = np.copy(self.ballVector)

        # touchpad
        ballRect = pygame.Rect(self.ballPos[0] - self.ballSize, self.ballPos[1] - self.ballSize, self.ballSize * 2,
                               self.ballSize * 2)
        if pygame.Rect.colliderect(ballRect, self.barRect) == True:
            tmpVector = norm([self.ballPos - np.array(self.barRect.center)]).ravel()
            if tmpVector[1] >= 0:
                tmpVector = norm([[tmpVector[0], 1]]).ravel()
            elif tmpVector[1] > -0.3:
                if tmpVector[0] > 0:
                    tmpVector += norm(np.array([[0.95, -0.3]])).ravel() - tmpVector
                else:
                    tmpVector += norm(np.array([[-0.95, -0.3]])).ravel() - tmpVector

        self.ballVector = tmpVector

        if np.sum(self.brick) > 0:
            return False
        else:
            return True

    def moveBar(self):
        self.barRect.left += self.barSpeed * self.barMove
        if self.barRect.left >= 400.0 - self.barLength:
            self.barRect.left = int(400.0 - self.barLength)
        if self.barRect.left <= 0:
            self.barRect.left = 0

    def render(self, surface, move=0):
        # startPause
        if self.start:
            self.startpause(surface)
            self.start = False

        # clearScreen
        surface.fill(colors.Black)

        # draw things
        self.draw(surface)

        # update
        self.barMove = move
        self.moveBar()
        self.end = self.ballMove()

        # time step
        self.ticks += 1
        if self.ticks == self.fps:
            self.seconds, self.ticks = self.seconds + 1, 0
        if self.seconds == 180:
            self.end = True
            self.timesup(surface)
            return True

        # check gameover
        if self.end:
            if np.sum(self.brick) > 0:
                self.die(surface)
            else:
                self.gameclear(surface)
            return True

        pygame.event.clear()
        pygame.display.update()
        return False


if __name__ == '__main__':
    # Game start
    atari = Atarigame()
    pygame.init()
    surface = pygame.display.set_mode((atari.windowWidth, atari.windowHeight))
    pygame.display.set_caption('Atari Breakout')
    end = False
    while True:
        # getkey
        pressedKeys = pygame.key.get_pressed()
        if pressedKeys[ord('q')]:
            pygame.quit()
            sys.exit()
        elif pressedKeys[ord('a')] and not pressedKeys[ord('d')]:
            move = -1
        elif pressedKeys[ord('d')] and not pressedKeys[ord('a')]:
            move = 1
        else:
            move = 0

        end = atari.render(surface,move=move)

        if end:
            time.sleep(1.5)
            atari.restart()
        else:
            atari.fpsClock.tick(atari.fps * atari.runspeed)
