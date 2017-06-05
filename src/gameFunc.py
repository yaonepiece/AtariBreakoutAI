from __future__ import division
import pygame, sys, time, colors
import numpy as np
from sklearn.preprocessing import normalize as norm

def ballMove(ballPos, ballVector, ballSpeed, ballSize, brick, barRect):
	reward=0.01
	
	#move
	prePos=np.copy(ballPos)
	ballPos+=ballSpeed*ballVector
	
	#touchwall
	if ballPos[0]<=ballSize:
		ballPos[0]=float(ballSize)
		ballVector[0]*=-1
	elif ballPos[0]>=(400.0-ballSize):
		ballPos[0]=400.0-ballSize
		ballVector[0]*=-1
	if ballPos[1]<=ballSize:
		ballPos[1]=float(ballSize)
		ballVector[1]*=-1
	elif ballPos[1]>=(300.0-ballSize):
		return ballVector, True, -1
	
	#breakbrick
	for i in range(5):
		for j in range(8):
			if brick[i][j]==1:	
				#touch from right
				if ballPos[0]<=(j*50+ballSize+50) and ballPos[0]>(j*50-ballSize+50) and prePos[0]>ballPos[0] and ballPos[1]>=(20*i+40) and ballPos[1]<(20*i+60):
					ballVector[0]*=-1
					brick[i][j]=0
				#touch from left
				if ballPos[0]>=(j*50-ballSize) and ballPos[0]<(j*50+ballSize) and prePos[0]<ballPos[0] and ballPos[1]>=(20*i+40) and ballPos[1]<(20*i+60):
					ballVector[0]*=-1
					brick[i][j]=0
				#touch from bottom
				if ballPos[1]<=(i*20+ballSize+60) and ballPos[1]>(i*20-ballSize+60) and prePos[1]>ballPos[1] and ballPos[0]>=(50*j) and ballPos[0]<(50*j+50):
					ballVector[1]*=-1
					brick[i][j]=0
				#touch from top
				if ballPos[1]>=(i*20-ballSize+40) and ballPos[1]<(i*20+ballSize+40) and prePos[1]<ballPos[1] and ballPos[0]>=(50*j) and ballPos[0]<(50*j+50):
					ballVector[1]*=-1
					brick[i][j]=0
	
	tmpVector=np.copy(ballVector)
	
	#touchpad
	ballRect=pygame.Rect(ballPos[0]-ballSize,ballPos[1]-ballSize,ballSize*2,ballSize*2)
	if pygame.Rect.colliderect(ballRect, barRect)==True:
		tmpVector=norm([ballPos-np.array(barRect.center)]).ravel()
		if tmpVector[1]>=0:
			tmpVector=norm([[tmpVector[0],1]]).ravel()
		else:
			if tmpVector[1]>-0.3:
				if tmpVector[0]>0:
					tmpVector=norm(np.array([[0.95,-0.3]])).ravel()
				else:
					tmpVector=norm(np.array([[-0.95,-0.3]])).ravel()
			reward=1
	
	for i in range(5):
		for j in range(8):
			if brick[i][j]==1:
				return tmpVector, False, reward
	return tmpVector, True, reward

def barMove(barRect, barSpeed, barMove, barLength):
	barRect.left+=barSpeed*barMove
	if barRect.left>=400.0-barLength:
		barRect.left=int(400.0-barLength)
	if barRect.left<=0:
		barRect.left=0

