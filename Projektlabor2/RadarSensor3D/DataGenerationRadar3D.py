'''
This script simulates a 3D-Radar-Sensor capable of outputing 
detected position and velocity and amplitude of an object.

The simulation is done at 100 times the measurement rate of the sensor,
then decimate it to the actual measurement rate
'''

'''
Sensor characteristics
'''

import numpy
minRange = 0.3  # m
maxRange = 25.0  # m
maxVelocity = 25  # m/s --> 90 km/h

rangeAccuracy = 0.05  # m
velocityAccuracy = 0.005  # m/s
measurementRate = 30  # Hz

sensorPosition = numpy.array([0,0,0.8]) # x,y,z-coordinate of the sensor

'''
Class definition of a target
'''

class Target:
    def __init__(self, opt):
        self.ValidateOption(opt)
        self.currentPosition = self.opt['InitialPosition']
        self.velocityVector = numpy.zeros((3,))
        self.pathCounter = 0  
        self.reachedEnd = False      

    def ValidateOption(self, opt):
        if(('InitialPosition' in opt) == False):
            raise Exception("Missing value for 'InitialPosition'")
        else:
            if(type(opt['InitialPosition']) is not numpy.ndarray):
                raise Exception("InitialPosition should be a numpy.ndarray with the shape (3,).")
            else:
                if(numpy.shape(opt['InitialPosition']) != (3,)):
                    raise Exception("InitialPosition should be a numpy.ndarray with the shape (3,).")

        if(('Path' in opt) == False):
            raise Exception("Missing value for 'Path'")
        else:
            if(type(opt['Path']) is not numpy.ndarray):
                raise Exception("Path should be a numpy.ndarray with the shape (3,n).")
            else:
                if(numpy.size(opt['Path'],0) != 3):
                    raise Exception("Path should be a numpy.ndarray with the shape (3,n).")

        if(('Velocities' in opt) == False):
            raise Exception("Missing value for 'Velocities'")
        else:
            if(type(opt['Velocities']) is not numpy.ndarray):
                raise Exception("Velocities should be a numpy.ndarray with the shape (1,n).")
            else:
                if(numpy.size(opt['Velocities'],0) != 1):
                    raise Exception("Velocities should be a numpy.ndarray with the shape (1,n).")

                if(numpy.size(opt['Velocities'],1) != numpy.size(opt['Path'],1)):
                    raise Exception("Velocities and Path should have the same length.")

        self.opt = opt

    def Step(self, deltaTime):
        # check if we are at the end
        if(self.pathCounter > numpy.size(self.opt['Path'],1) - 1):
            self.reachedEnd = True
            return self.currentPosition, self.velocityVector
                
        velocityVector = self.__GetVelocityVector(self.currentPosition, self.opt['Path'][:, self.pathCounter], self.opt['Velocities'][0, self.pathCounter])

        # try to step
        nextPosition = self.currentPosition + velocityVector * deltaTime

        # now check if we are within the next path target
        if(numpy.linalg.norm(nextPosition - self.currentPosition) < numpy.linalg.norm(self.opt['Path'][:, self.pathCounter] - self.currentPosition)):
            self.currentPosition = nextPosition
            self.velocityVector = velocityVector
            return self.currentPosition, self.velocityVector

        else: # we have to microstep
            # get the time to current path target
            countDown = deltaTime
            stepFurther = True
            while(stepFurther == True):
                microtime = (numpy.linalg.norm(self.opt['Path'][:, self.pathCounter] - self.currentPosition)/numpy.linalg.norm(nextPosition - self.currentPosition)) * countDown
                resttime = countDown - microtime

                if(self.pathCounter + 1 == numpy.size(self.opt['Path'],1)):
                    stepFurther == False
                    self.pathCounter = self.pathCounter + 1
                    self.velocityVector = numpy.zeros((3,))
                    self.currentPosition = self.opt['Path'][:, self.pathCounter - 1]
                    self.reachedEnd = True
                    return self.currentPosition, self.velocityVector

                velocityVector = self.__GetVelocityVector(self.opt['Path'][:, self.pathCounter], self.opt['Path'][:, self.pathCounter + 1], self.opt['Velocities'][:, self.pathCounter + 1])
                # try to step
                nextPosition = self.opt['Path'][:, self.pathCounter] + velocityVector * resttime

                if(numpy.linalg.norm(nextPosition - self.opt['Path'][:, self.pathCounter]) < numpy.linalg.norm(self.opt['Path'][:, self.pathCounter + 1] - self.opt['Path'][:, self.pathCounter])):
                    stepFurther = False 

                # we proceed to next path target
                self.pathCounter = self.pathCounter + 1

            self.velocityVector = velocityVector
            self.currentPosition = nextPosition
            return self.currentPosition, self.velocityVector

    def __GetVelocityVector(self, Position1, Position2, Velocity):
        targetPosition = Position2
        movementDirection = targetPosition - Position1
        movementDirection = movementDirection / numpy.linalg.norm(movementDirection)

        return Velocity * movementDirection
    
class RadarSensor:
    def __init__(self, opt):
        self.ValidateOption(opt)
        opt['MinRange'] = minRange
        opt['MaxRange'] = maxRange
        opt['MaxVelocity'] = maxVelocity
        opt['RangeAccuracy'] = rangeAccuracy
        opt['VelocityAccuracy'] = velocityAccuracy
        opt['MeasurementRate'] = 30

    def ValidateOption(self, opt):
        if(('Position' in opt) == False):
            raise Exception("Missing value for 'Position'")
        else:
            if(type(opt['Position']) is not numpy.ndarray):
                raise Exception("Position should be a numpy.ndarray with the shape (3,).")
            else:
                if(numpy.shape(opt['Position']) != (3,)):
                    raise Exception("Position should be a numpy.ndarray with the shape (3,).")

        if(('OpeningAngle' in opt) == False):
            raise Exception("Missing value for 'OpeningAngle'")
        else:
            if(type(opt['OpeningAngle']) is not numpy.ndarray):
                raise Exception("OpeningAngle should be a numpy.ndarray with the shape (2,).")
            else:
                if(numpy.size(opt['OpeningAngle'],0) != 2):
                    raise Exception("OpeningAngle should be a numpy.ndarray with the shape (2,).")

        self.opt = opt

    def Detect(self, targets):
        if len(targets) == 0:
            return None

        # initiate list
        detections = []

        # looping through targets
        for target in targets:
            visibleHor = False
            visibleVer = False
            
            # check horizontal angle
            horAngle = numpy.rad2deg(numpy.arctan((target.currentPosition[0] - self.opt['Position'][0])/(target.currentPosition[1] - self.opt['Position'][1])))
            if(numpy.abs(horAngle) < self.opt['OpeningAngle'][0]/2.0):
                visibleHor = True

            # check vertical angle
            verAngle = numpy.rad2deg(numpy.arctan((target.currentPosition[2] - self.opt['Position'][2])/(target.currentPosition[1] - self.opt['Position'][1])))
            if(numpy.abs(verAngle) < self.opt['OpeningAngle'][1]/2.0):
                visibleVer = True

            # if target visible
            if(visibleVer == True and visibleHor == True):            
                currPos = target.currentPosition - self.opt['Position']
                
                if(numpy.linalg.norm(currPos) > self.opt['MinRange'] and numpy.linalg.norm(currPos) < self.opt['MaxRange']):
                    bVector = self.opt['Position'] - target.currentPosition
                    radialVelocityToSensor = (numpy.dot(target.velocityVector, bVector)/numpy.dot(bVector, bVector)) + numpy.random.uniform(-1*self.opt['VelocityAccuracy'],self.opt['VelocityAccuracy'],1)
                    currPos = currPos + numpy.random.uniform(-1*self.opt['RangeAccuracy'],self.opt['RangeAccuracy'],3)
                    currPos = numpy.append(currPos, radialVelocityToSensor)

                    if(radialVelocityToSensor < self.opt['MaxVelocity']):
                        detections.append(currPos)
            else:
                print("Non Visible Point")


        # add noise / false detection ?
        if(('FalseDetection' in self.opt) == True):
            if(self.opt['FalseDetection'] == True):
                for i in range(20):
                    randPos = numpy.random.uniform(self.opt['MinRange'], 2 * self.opt['MaxRange'], 3)
                    randVel = numpy.random.uniform(0, 2 * self.opt['MaxVelocity'], 1)
                    visibleHor = False
                    visibleVer = False
                    
                    # check horizontal angle
                    horAngle = numpy.rad2deg(numpy.arctan((randPos[0] - self.opt['Position'][0])/(randPos[1] - self.opt['Position'][1])))
                    if(numpy.abs(horAngle) < self.opt['OpeningAngle'][0]/2.0):
                        visibleHor = True

                    # check vertical angle
                    verAngle = numpy.rad2deg(numpy.arctan((randPos[2] - self.opt['Position'][2])/(randPos[1] - self.opt['Position'][1])))
                    if(numpy.abs(verAngle) < self.opt['OpeningAngle'][1]/2.0):
                        visibleVer = True

                    # if target visible
                    if(visibleVer == True and visibleHor == True):            
                        randPos = randPos - self.opt['Position']

                        if(numpy.linalg.norm(randPos) > self.opt['MinRange'] and numpy.linalg.norm(randPos) < self.opt['MaxRange']):
                            bVector = self.opt['Position'] - randPos
                            randPos = randPos + numpy.random.uniform(-1*self.opt['RangeAccuracy'],self.opt['RangeAccuracy'],3)
                            randPos = numpy.append(randPos, randVel)
                            if(randVel < self.opt['MaxVelocity']):
                                detections.append(randPos)

        return detections
