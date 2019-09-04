import string;
import numpy as np;
from numpy import exp, array, random, dot;
from scipy.special import expit as activation_function;
from scipy.stats import truncnorm;

score_requirement = 1000;
initial_games = 5000;
gameBoard = np.array([0,0,0,0,0,0])
inputVector = np.array([gameBoard[0],gameBoard[1],gameBoard[2],gameBoard[3],gameBoard[4],gameBoard[5],random.randint(1,9)])

inputNodes = 7;
outputNodes = 6;
hiddenNodes = 7;
learnRate = 1;

@np.vectorize
def sigmoid(what):
    return 1/(1+np.e ** - what);

activation_function = sigmoid;
def truncated_normal(mean, sd, low, up):
    return truncnorm((low-mean)/sd,(up-mean)/sd,loc=mean, scale=sd)

def insertNumber(inNum, output,rewards):
    maxValue = 0;
    spot = 0;
    #find max value (argmax wasn't working)
    #for i in range(0,6):
    #    if(output[i] > maxValue):
    #        maxValue = output[i];
    #spot = np.where(output == maxValue);
    spot = np.argmax(output);
    print(output, inNum, spot+1);
    if(gameBoard[spot] == 0):
        gameBoard[spot] = inNum;
        rewards[spot] += 100;
        return 0;
    else:
        rewards[spot] -= 10;
        return 1000;

def judgeGame():
    numlist = [gameBoard[0],gameBoard[1],gameBoard[2]];
    num1 = int("".join(str(x) for x in numlist));
    numlist = [gameBoard[3],gameBoard[4],gameBoard[5]];
    num2 = int("".join(str(x) for x in numlist));
    score = num1 - num2;
    #print(gameBoard);
    print(score);
    if(score < 0):
        return 1000;
    else:
        return score;

def outputMaker(x):
    return np.average(x);

def printGameboard():
    print(gameBoard[0] , "|" , gameBoard[1], "|",gameBoard[2]);
    print(gameBoard[3] , "|" , gameBoard[4], "|",gameBoard[5]);


class theBrain:
    def __init__(self, nIn, nOn, nHn, learnRate):
        self.nIn = nIn;
        self.nOn = nOn;
        self.nHn = nHn;
        self.learnRate = learnRate;
        self.createWeights();

    def createWeights(self):
        #Input to Hidden Synapses
        ryan = 1 / np.sqrt(self.nIn);
        x = truncated_normal(mean=0, sd=1, low=-ryan, up=ryan);
        self.hWeights = x.rvs(self.nHn,self.nIn);
        #Hidden to Output Synapses
        ryan = 1 / np.sqrt(self.nHn);
        x = truncated_normal(mean=0, sd=1, low=-ryan, up=ryan);
        self.oWeights = x.rvs((self.nOn,self.nHn));

    def train(self, inputVector):
        mistakes = 0;
        rewards = np.array([0,0,0,0,0,0]);
        gameBoard = np.array([0,0,0,0,0,0]);
        for i in range(0,6):
            inputVector = np.array([gameBoard[0],gameBoard[1],gameBoard[2],gameBoard[3],gameBoard[4],gameBoard[5],random.randint(1,9)])
            #inputVector= np.array(inputVector, ndmin=2).T;

            #self.hWeights = np.array(self.hWeights, ndmin=2).T;
            #self.oWeights = np.array(self.oWeights, ndmin=2).T;

            outputVector = np.dot(self.hWeights,inputVector);
            outputHiddenVector = activation_function(outputVector);

            outputVector1 = np.dot(self.oWeights,outputHiddenVector);
            outputNetworkVector = activation_function(outputVector1);
            #printGameboard();
            outputFloats = np.apply_along_axis(outputMaker,axis=1,arr=outputNetworkVector);
            mistakes += insertNumber(inputVector[6],outputFloats,rewards);

        mistakes += judgeGame();
        print(rewards, " | ", mistakes);
        #Learn
        wwt = np.dot(rewards, outputNetworkVector) * (1.0 - outputNetworkVector);
        wwt = self.learnRate * np.dot(wwt, outputHiddenVector.T);
        print(wwt);
        self.oWeights += wwt;

        wwt = np.dot(rewards, outputHiddenVector) * (1.0 - outputHiddenVector);
        self.hWeights += self.learnRate * np.dot(wwt,inputVector.T)
        gameBoard = np.array([0,0,0,0,0,0]);

    def run(self, inputVector):

        #Layer 1
        inputVector= np.array(inputVector, ndmin=2).T;
        outputVector= np.dot(self.hWeights,inputVector);
        outputVector= activation_function(outputVector);
        #Output Layer
        outputVector = np.dot(self.oWeights,outputVector);
        outputVector= activation_function(outputVector);

        return outputVector;

watermelonium = theBrain(inputNodes, outputNodes, hiddenNodes, learnRate);
for i in range(0,initial_games):
    watermelonium.train(inputVector);
