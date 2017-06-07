import pygame, sys, time
import colors
from math import cos, sin, pi
import numpy as np
from sklearn.preprocessing import normalize as norm


class Atarigame:
    def __init__(self, speed=1, brickx=8,bricky=5,white=False,banner=False):
        # basic datas
        self.windowWidth = 400
        self.windowHeight = 300
        self.runspeed = speed
        self.fps = 50
        self.fpsClock = pygame.time.Clock()
        self.brickx, self.bricky=brickx, bricky
        self.brick = np.ones((bricky, brickx), dtype=int)
        self.brickRect = [[pygame.Rect(self.windowWidth * j // brickx, 20 * i + 40,  (self.windowWidth * (j+1) // brickx)-(self.windowWidth * j // brickx), 20) for j in range(brickx)] for i in range(bricky)]
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
        self.white=white
        self.banner=banner

        self.ballrand()
        pygame.init()
        self.surface = pygame.display.set_mode((self.windowWidth, self.windowHeight))
        pygame.display.set_caption('Atari Breakout')

    def getBrick(self):
        for i in range(8):
            self.brickstatus[i] = 0.
        for j in range(8):
            for i in range(4, -1, -1):
                if self.brick[i][j]:
                    self.brickstatus[j] = (i + 1) * 0.2
                    break

    def die(self):
        self.draw()
        fontsize = 50
        font = pygame.font.SysFont(None, fontsize)

        if self.white:
            text = font.render('You lose!', 1, colors.Red)
        else:
            text = font.render('You lose!', 1, colors.White)
        textRect = text.get_rect()  # Set the center of the text box
        textRect.centerx = self.surface.get_rect().centerx
        textRect.centery = self.surface.get_rect().centery
        self.surface.blit(text, textRect)
        pygame.display.update()

    def gameclear(self):
        self.draw()
        fontsize = 50
        font = pygame.font.SysFont(None, fontsize)

        if self.white:
            text = font.render('You win!', 1, colors.Red)
        else:
            text = font.render('You win!', 1, colors.White)
        textRect = text.get_rect()  # Set the center of the text box
        textRect.centerx = self.surface.get_rect().centerx
        textRect.centery = self.surface.get_rect().centery
        self.surface.blit(text, textRect)
        pygame.display.update()

    def timesup(self):
        self.draw()
        fontsize = 50
        font = pygame.font.SysFont(None, fontsize)

        if self.white:
            text = font.render('Time\'s up!', 1, colors.Red)
        else:
            text = font.render('Time\'s up!', 1, colors.White)
        textRect = text.get_rect()  # Set the center of the text box
        textRect.centerx = self.surface.get_rect().centerx
        textRect.centery = self.surface.get_rect().centery
        self.surface.blit(text, textRect)
        pygame.display.update()

    def restart(self):
        self.brick = np.ones((self.bricky, self.brickx), dtype=int)
        self.ballPos = np.array([200.0, 200.0])
        self.ballrand()
        self.barRect = pygame.Rect(self.barinitX - self.barLength // 2, self.barinitY - self.barWidth // 2,
                                   self.barLength, self.barWidth)
        self.seconds = 0
        self.ticks = 0
        self.start = True
        self.end = False

    def draw(self):
        self.surface.fill(colors.Black)
        if self.white:
            pygame.draw.circle(self.surface, colors.White, tuple(self.ballPos.astype(int)), self.ballSize)
        else:
            pygame.draw.circle(self.surface, colors.Red, tuple(self.ballPos.astype(int)), self.ballSize)
        for i in range(self.bricky):
            for j in range(self.brickx):
                if self.brick[i][j] == 1:
                    if not self.white:
                        if (i + j) % 2 == 0:
                            rectcolor = colors.Orange
                        else:
                            rectcolor = colors.Blue
                    else:
                        rectcolor = colors.White
                    pygame.draw.rect(self.surface, rectcolor, self.brickRect[i][j])
        pygame.draw.rect(self.surface, colors.White, self.barRect)
        pygame.display.update()

    def startpause(self):
        fontsize = 50
        font = pygame.font.SysFont(None, fontsize)

        self.draw()
        if self.white:
            text = font.render('Ready?', 1, colors.Red)
        else:
            text = font.render('Ready?', 1, colors.White)
        textRect = text.get_rect()  # Set the center of the text box
        textRect.centerx = self.surface.get_rect().centerx
        textRect.centery = self.surface.get_rect().centery
        self.surface.blit(text, textRect)
        pygame.display.update()
        time.sleep(0.3)

    def ballrand(self):
        angle = (np.random.rand() * 0.5 + 1.27) * pi
        self.ballVector = np.array([cos(angle), sin(angle)])

    def ballMove(self):
        reward=0.01
        # move
        prePos = np.copy(self.ballPos)
        self.ballPos += self.ballSpeed * self.ballVector

        # touchwall
        if self.ballPos[0] <= self.ballSize:
            self.ballPos[0] = float(self.ballSize)
            self.ballVector[0] *= -1
        elif self.ballPos[0] >= (self.windowWidth - self.ballSize):
            self.ballPos[0] = self.windowWidth - self.ballSize
            self.ballVector[0] *= -1
        if self.ballPos[1] <= self.ballSize:
            self.ballPos[1] = float(self.ballSize)
            self.ballVector[1] *= -1
        elif self.ballPos[1] >= (self.windowHeight - self.ballSize):
            return True, -5

        # breakbrick
        for i in range(self.bricky):
            for j in range(self.brickx):
                if self.brick[i][j] == 1:
                    # touch from right
                    if self.ballPos[0] <= (self.brickRect[i][j].right + self.ballSize) and self.ballPos[0] > (
                                self.brickRect[i][j].right - self.ballSize) and prePos[0] > self.ballPos[0] and self.ballPos[1] >= self.brickRect[i][j].top and self.ballPos[1] < self.brickRect[i][j].bottom:
                        self.ballVector[0] *= -1
                        self.brick[i][j] = 0
                    # touch from left
                    if self.ballPos[0] >= (self.brickRect[i][j].left - self.ballSize) and self.ballPos[0] < (self.brickRect[i][j].right + self.ballSize) and \
                                    prePos[0] < self.ballPos[0] and self.ballPos[1] >= self.brickRect[i][j].top and self.ballPos[1] < self.brickRect[i][j].bottom:
                        self.ballVector[0] *= -1
                        self.brick[i][j] = 0
                    # touch from bottom
                    if self.ballPos[1] <= (self.brickRect[i][j].bottom + self.ballSize) and self.ballPos[1] > (
                                self.brickRect[i][j].bottom - self.ballSize) and prePos[1] > self.ballPos[1] and self.ballPos[0] >= self.brickRect[i][j].left and self.ballPos[0] < self.brickRect[i][j].right:
                        self.ballVector[1] *= -1
                        self.brick[i][j] = 0
                    # touch from top
                    if self.ballPos[1] >= (self.brickRect[i][j].top - self.ballSize) and self.ballPos[1] < (
                                self.brickRect[i][j].top + self.ballSize) and prePos[1] < self.ballPos[1] and self.ballPos[0] >= self.brickRect[i][j].left and self.ballPos[0] < self.brickRect[i][j].right:
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
            reward=0.1

        self.ballVector = tmpVector

        if np.sum(self.brick) > 0:
            return False, reward
        else:
            return True, 20

    def moveBar(self):
        self.barRect.left += self.barSpeed * self.barMove
        if self.barRect.left >= 400.0 - self.barLength:
            self.barRect.left = int(400.0 - self.barLength)
        if self.barRect.left <= 0:
            self.barRect.left = 0

    def render(self, move=0):
        terminal=False
        # startPause
        if self.start:
            if self.banner:
                self.startpause()
            self.start = False

        # clearScreen
        self.surface.fill(colors.Black)

        # draw things
        self.draw()

        # update
        self.barMove = move
        self.moveBar()
        preb=np.sum(self.brick)
        self.end, reward = self.ballMove()
        aftb=np.sum(self.brick)
        if not end and aftb<preb:
            reward=preb-aftb

        # time step
        self.ticks += 1
        if self.ticks == self.fps:
            self.seconds, self.ticks = self.seconds + 1, 0
        if self.seconds == 180:
            self.end = True
            if self.banner:
                self.timesup()
            else:
                self.restart()

            terminal=True
        # check gameover
        elif self.end:
            if self.banner:
                if np.sum(self.brick) > 0:
                    self.die()
                else:
                    self.gameclear()
            else:
                self.restart()
            terminal=True

        pygame.event.clear()
        pygame.display.update()
        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        return image_data, reward, terminal


if __name__ == '__main__':
    # Game start
    atari = Atarigame(speed=1,brickx=16,bricky=5,white=True,banner=True)
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

        screen, reward, end = atari.render(move=move)

        if end:
            time.sleep(0.5)
            atari.restart()
        else:
            atari.fpsClock.tick(atari.fps * atari.runspeed)
