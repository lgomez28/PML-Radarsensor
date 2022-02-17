'''
This script simulates a 1D-Radar-Sensor capable of outputing 
detected position and velocity and amplitude of an object.

The simulation is done at 100 times the measurement rate of the sensor,
then decimate it to the actual measurement rate
'''

'''
Sensor characteristics
'''

import copy
import numpy

minRange = 0.3  # m
maxRange = 25.0  # m
maxVelocity = 25  # m/s --> 90 km/h

rangeAccuracy = 0.02  # m
velocityAccuracy = 0.005  # m/s
measurementRate = 100  # Hz

'''
Function to generate measurement data 
of a 1D radar sensor.

Example: 

    opt = {
        "initialDistance": 5
        "stopTime": 1
    }
    timeAxis, distValues, velValues = GenerateData(type="Static", options=opt)
'''
def GenerateData(type="Static", options={}):

    # static
    if(type == "Static"):
        # sanity check
        if(("initialDistance" in options) == False) \
                or (("stopTime" in options) == False):
            return None, None

        timeAxis = numpy.arange(0, options["stopTime"], 0.01/measurementRate)
        distValues = options["initialDistance"] * \
            numpy.ones(numpy.shape(timeAxis))
        truthDistValues = copy.copy(distValues)

        distValues += numpy.random.uniform(-1*rangeAccuracy,
                                           rangeAccuracy, numpy.shape(timeAxis))
        velValues = numpy.zeros(numpy.shape(timeAxis))
        truthVelValues = copy.copy(velValues)

        velValues += numpy.random.uniform(-1*velocityAccuracy,
                                          velocityAccuracy, numpy.shape(timeAxis))
        velValues[distValues > maxRange] = numpy.NaN
        distValues[distValues > maxRange] = numpy.NaN
        velValues[distValues < minRange] = numpy.NaN
        distValues[distValues < minRange] = numpy.NaN        
        velValues[velValues > maxVelocity] = numpy.NaN
        velValues[velValues < -1 * maxVelocity] = numpy.NaN

        # decimate to actual measurement rate
        timeAxis = timeAxis[0::100]
        distValues = distValues[0::100]
        velValues = velValues[0::100]
        truthDistValues = truthDistValues[0::100]
        truthVelValues = truthVelValues[0::100]
    
        if("SporadicError" in options):
            rng = numpy.random.default_rng()
            ind = rng.choice(numpy.size(timeAxis), size=options["SporadicError"], replace=False)

            distValues[ind] = numpy.random.uniform(minRange,
                                           maxRange, numpy.shape(ind))

            velValues[ind] = numpy.random.uniform(-1*maxVelocity,
                                           maxVelocity, numpy.shape(ind))

        return timeAxis, distValues, velValues, truthDistValues, truthVelValues

    # constant velocity
    if(type == "ConstantVelocity"):
        # sanity check
        if(("initialDistance" in options) == False) \
                or (("stopTime" in options) == False) \
                or (("velocity" in options) == False):
            return None, None

        timeAxis = numpy.arange(0, options["stopTime"], 0.01/measurementRate)

        distValues = options["initialDistance"] + options["velocity"]*timeAxis
        truthDistValues = copy.copy(distValues)

        distValues += numpy.random.uniform(-1*rangeAccuracy,
                                           rangeAccuracy, numpy.shape(timeAxis))

        velValues = options["velocity"] * numpy.ones(numpy.shape(timeAxis))
        truthVelValues = copy.copy(velValues)

        velValues += numpy.random.uniform(-1*velocityAccuracy,
                                          velocityAccuracy, numpy.shape(timeAxis))
        velValues[distValues > maxRange] = numpy.NaN
        distValues[distValues > maxRange] = numpy.NaN
        velValues[distValues < minRange] = numpy.NaN
        distValues[distValues < minRange] = numpy.NaN
        velValues[velValues > maxVelocity] = numpy.NaN
        velValues[velValues < -1 * maxVelocity] = numpy.NaN
        
        # decimate to actual measurement rate
        timeAxis = timeAxis[0::100]
        distValues = distValues[0::100]
        velValues = velValues[0::100]
        truthDistValues = truthDistValues[0::100]
        truthVelValues = truthVelValues[0::100]
    
        if("SporadicError" in options):
            rng = numpy.random.default_rng()
            ind = rng.choice(numpy.size(timeAxis), size=options["SporadicError"], replace=False)

            distValues[ind] = numpy.random.uniform(minRange,
                                           maxRange, numpy.shape(ind))

            velValues[ind] = numpy.random.uniform(-1*maxVelocity,
                                           maxVelocity, numpy.shape(ind))

        return timeAxis, distValues, velValues, truthDistValues, truthVelValues

    # constant acceleration
    if(type == "ConstantAcceleration"):
        # sanity check
        if(("initialDistance" in options) == False) \
                or (("stopTime" in options) == False) \
                or (("initialVelocity" in options) == False) \
                or (("acceleration" in options) == False):
            return None, None

        timeAxis = numpy.arange(0, options["stopTime"], 0.01/measurementRate)

        velValues = options["initialVelocity"] + \
            options["acceleration"] * timeAxis
        distValues = options["initialDistance"] + 0.5 * \
            options["acceleration"] * timeAxis * timeAxis

        truthVelValues = copy.copy(velValues)
        truthDistValues = copy.copy(distValues)

        velValues += numpy.random.uniform(-1*velocityAccuracy,
                                          velocityAccuracy, numpy.shape(timeAxis))
        distValues += numpy.random.uniform(-1*rangeAccuracy,
                                           rangeAccuracy, numpy.shape(timeAxis))

        velValues[distValues > maxRange] = numpy.NaN
        distValues[distValues > maxRange] = numpy.NaN
        velValues[distValues < minRange] = numpy.NaN
        distValues[distValues < minRange] = numpy.NaN
        velValues[velValues > maxVelocity] = numpy.NaN
        velValues[velValues < -1 * maxVelocity] = numpy.NaN
        
        # decimate to actual measurement rate
        timeAxis = timeAxis[0::100]
        distValues = distValues[0::100]
        velValues = velValues[0::100]
        truthDistValues = truthDistValues[0::100]
        truthVelValues = truthVelValues[0::100]
    
        if("SporadicError" in options):
            rng = numpy.random.default_rng()
            ind = rng.choice(numpy.size(timeAxis), size=options["SporadicError"], replace=False)

            distValues[ind] = numpy.random.uniform(minRange,
                                           maxRange, numpy.shape(ind))

            velValues[ind] = numpy.random.uniform(-1*maxVelocity,
                                           maxVelocity, numpy.shape(ind))

        return timeAxis, distValues, velValues, truthDistValues, truthVelValues

    # sinus movement
    if(type == "Sinus"):
        # sanity check
        if(("initialDistance" in options) == False) \
                or (("stopTime" in options) == False) \
                or (("movementRange" in options) == False) \
                or (("frequency" in options) == False):
            return None, None

        timeAxis = numpy.arange(0, options["stopTime"], 0.01/measurementRate)

        distValues = options["initialDistance"] + options["movementRange"] * \
            numpy.sin(2*numpy.pi*options["frequency"]*timeAxis)

        truthDistValues = copy.copy(distValues)

        velValues = 2*numpy.pi*options["frequency"] * options["movementRange"] * numpy.cos(
            2*numpy.pi*options["frequency"]*timeAxis)

        truthVelValues = copy.copy(velValues)

        velValues += numpy.random.uniform(-1*velocityAccuracy,
                                          velocityAccuracy, numpy.shape(timeAxis))
        distValues += numpy.random.uniform(-1*rangeAccuracy,
                                           rangeAccuracy, numpy.shape(timeAxis))

        velValues[distValues > maxRange] = numpy.NaN
        distValues[distValues > maxRange] = numpy.NaN
        velValues[distValues < minRange] = numpy.NaN
        distValues[distValues < minRange] = numpy.NaN
        velValues[velValues > maxVelocity] = numpy.NaN
        velValues[velValues < -1 * maxVelocity] = numpy.NaN
        
        # decimate to actual measurement rate
        timeAxis = timeAxis[0::100]
        distValues = distValues[0::100]
        velValues = velValues[0::100]
        truthDistValues = truthDistValues[0::100]
        truthVelValues = truthVelValues[0::100]
    
        if("SporadicError" in options):
            rng = numpy.random.default_rng()
            ind = rng.choice(numpy.size(timeAxis), size=options["SporadicError"], replace=False)

            distValues[ind] = numpy.random.uniform(minRange,
                                           maxRange, numpy.shape(ind))

            velValues[ind] = numpy.random.uniform(-1*maxVelocity,
                                           maxVelocity, numpy.shape(ind))

        return timeAxis, distValues, velValues, truthDistValues, truthVelValues

    # triangle movement
    if(type == "Triangle"):
        # sanity check
        if(("initialDistance" in options) == False) \
                or (("stopTime" in options) == False) \
                or (("movementRange" in options) == False) \
                or (("frequency" in options) == False):
            return None, None

        timeAxis = numpy.arange(0, options["stopTime"], 0.01/measurementRate)

        distValues = numpy.zeros(numpy.shape(timeAxis))
        velValues = numpy.zeros(numpy.shape(timeAxis))

        for i in range(numpy.size(timeAxis)):
            t = timeAxis[i]
            while (t > 1/options["frequency"]):
                t = t - 1/options["frequency"]

            if (t <= 1/(2*options["frequency"])):
                if(i == 0):
                    distValues[i] = options["initialDistance"] + (2 * options["frequency"] * options["movementRange"])*0.01/measurementRate
                else:
                    distValues[i] = distValues[i-1] + (2 * options["frequency"] * options["movementRange"])*0.01/measurementRate
                
                velValues[i] = 2 * options["frequency"] * options["movementRange"]   
            else:
                distValues[i] = distValues[i-1] - (2 * options["frequency"] * options["movementRange"])*0.01/measurementRate
                velValues[i] = -2 * options["frequency"] * options["movementRange"]   

        truthDistValues = copy.copy(distValues)
        truthVelValues = copy.copy(velValues)

        velValues += numpy.random.uniform(-1*velocityAccuracy,
                                          velocityAccuracy, numpy.shape(timeAxis))
        distValues += numpy.random.uniform(-1*rangeAccuracy,
                                           rangeAccuracy, numpy.shape(timeAxis))

        velValues[distValues > maxRange] = numpy.NaN
        distValues[distValues > maxRange] = numpy.NaN
        velValues[distValues < minRange] = numpy.NaN
        distValues[distValues < minRange] = numpy.NaN
        velValues[velValues > maxVelocity] = numpy.NaN
        velValues[velValues < -1 * maxVelocity] = numpy.NaN
        # decimate to actual measurement rate
        timeAxis = timeAxis[0::100]
        distValues = distValues[0::100]
        velValues = velValues[0::100]
        truthDistValues = truthDistValues[0::100]
        truthVelValues = truthVelValues[0::100]
    
        if("SporadicError" in options):
            rng = numpy.random.default_rng()
            ind = rng.choice(numpy.size(timeAxis), size=options["SporadicError"], replace=False)

            distValues[ind] = numpy.random.uniform(minRange,
                                           maxRange, numpy.shape(ind))

            velValues[ind] = numpy.random.uniform(-1*maxVelocity,
                                           maxVelocity, numpy.shape(ind))

        return timeAxis, distValues, velValues, truthDistValues, truthVelValues

    else:
        return 0, 0

