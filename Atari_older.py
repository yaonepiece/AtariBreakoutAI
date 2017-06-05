import pygame, random, sys, time, colors, gameFunc
from math import cos, sin, pi
from pygame.locals import *
import numpy as np
from sklearn.preprocessing import normalize as norm
import tensorflow as tf
import cv2
from collections import deque

#basic datas
windowWidth=400
windowHeight=300
runspeed=8
fps=25
fpsClock=pygame.time.Clock()
brick=np.ones((5,8),dtype=int)
brickRect=[[pygame.Rect(50*j,20*i+40,50,20) for j in range(8)] for i in range(5)]
brickstatus=np.zeros((8),dtype=float)
def getBrick():
	global brick, brickstatus
	for i in range(8):
		brickstatus[i]=0.
	for j in range(8):
		for i in range(4,-1,-1):
			if brick[i][j]:
				brickstatus[j]=(i+1)*0.2
				break
ballPos=np.array([200.0,200.0])
ballSpeed=10
ballVector=norm([[-1,-1]]).ravel()
ballSize=8
barLength=50
barWidth=10
barinitX,barinitY=200.0,295.0
barRect=pygame.Rect(barinitX-barLength//2,barinitY-barWidth//2,barLength,barWidth)
seconds=0
ticks=0
AST=0.
barSpeed=16
barMove=0
start=True
end=False
epi=1
MAXEPISODE=10000000

#CNN+DQN


def die(surface):
	global seconds
	fontsize=50
	font=pygame.font.SysFont(None, fontsize)
	
	text=font.render('You lose!', 1, colors.White)
	textRect=text.get_rect() #Set the center of the text box
	textRect.centerx=surface.get_rect().centerx
	textRect.centery=surface.get_rect().centery
	surface.blit(text, textRect)
	pygame.display.update()
	time.sleep(0.5)
	#pygame.quit()
	#sys.exit()
	restart()

def gameclear(surface):
	global seconds
	fontsize=50
	font=pygame.font.SysFont(None, fontsize)
	
	text=font.render('You win!', 1, colors.White)
	textRect=text.get_rect() #Set the center of the text box
	textRect.centerx=surface.get_rect().centerx
	textRect.centery=surface.get_rect().centery
	surface.blit(text, textRect)
	pygame.display.update()
	time.sleep(0.5)
	#pygame.quit()
	#sys.exit()
	restart()

def timesup(surface):
	global seconds
	fontsize=50
	font=pygame.font.SysFont(None, fontsize)
	
	text=font.render('Time\'s up!', 1, colors.White)
	textRect=text.get_rect() #Set the center of the text box
	textRect.centerx=surface.get_rect().centerx
	textRect.centery=surface.get_rect().centery
	surface.blit(text, textRect)
	pygame.display.update()
	time.sleep(0.5)
	#pygame.quit()
	#sys.exit()
	restart()

def restart():
	global brick, ballPos, barinitX, barinitY, barRect, seconds, ticks
	brick=np.ones((5,8),dtype=int)
	ballPos=np.array([200.0,200.0])
	ballrand()
	barRect=pygame.Rect(barinitX-barLength//2,barinitY-barWidth//2,barLength,barWidth)
	seconds=0
	ticks=0

def startpause(surface):
	global ballPos, ballSize, brickRect, barRect
	surface.fill(colors.Black)
	pygame.draw.circle(surface, colors.Red, tuple(ballPos.astype(int)),ballSize)
	for i in range(5):
		for j in range(8):
			if brick[i][j]==1:
				if (i+j)%2==0:
					rectcolor=colors.Orange
				else:
					rectcolor=colors.Blue
				pygame.draw.rect(surface, rectcolor, brickRect[i][j])
	pygame.draw.rect(surface, colors.White, barRect)
	fontsize=50
	font=pygame.font.SysFont(None, fontsize)
	
	text=font.render('Ready?', 1, colors.White)
	textRect=text.get_rect() #Set the center of the text box
	textRect.centerx=surface.get_rect().centerx
	textRect.centery=surface.get_rect().centery
	surface.blit(text, textRect)
	pygame.display.update()

def ballrand():
	global ballVector
	angle=(np.random.rand()*0.5+1.27)*pi
	ballVector=np.array([cos(angle),sin(angle)])

#Game start
pygame.init()
surface=pygame.display.set_mode((windowWidth,windowHeight))
pygame.display.set_caption('Atari Breakout')
ballrand()

def frame_step(action):
	global ticks, fps, seconds, surface, ballVector, ballSize, barSpeed, barLength, barMove, brick, barRect, ballSpeed
	end=False
	ticks+=1
	if ticks==fps:
		seconds, ticks=seconds+1, 0
	if seconds==180:
		end=True
		reward=-1
	
	#clearScreen
	surface.fill(colors.Black)
	
	#draw things
	pygame.draw.circle(surface, colors.Red, tuple(ballPos.astype(int)),ballSize)
	for i in range(5):
		for j in range(8):
			if brick[i][j]==1:
				if (i+j)%2==0:
					rectcolor=colors.Orange
				else:
					rectcolor=colors.Blue
				pygame.draw.rect(surface, rectcolor, brickRect[i][j])
	pygame.draw.rect(surface, colors.White, barRect)
	
	if action[0]==1:
		barMove=0
	elif action[1]==1:
		barMove=-1
	else:
		barMove=1
	
	#update
	preb=np.sum(brick)
	gameFunc.barMove(barRect, barSpeed, barMove, barLength)
	ballVector, end, rewardgain=gameFunc.ballMove(ballPos, ballVector, ballSpeed, ballSize, brick, barRect)
	#ballVector, end=gameFunc.ballMove(ballPos, ballVector, ballSpeed, ballSize, brick, barRect)
	aftb=np.sum(brick)
	if aftb==0:
		rewardgain=6-seconds/60
	elif preb>aftb:
		rewardgain=0.1
	
	if end:
		restart()
	
	pygame.event.clear()
	image_data = pygame.surfarray.array3d(pygame.display.get_surface())
	pygame.display.update()
	fpsClock.tick(fps*runspeed)
	return image_data, rewardgain, end