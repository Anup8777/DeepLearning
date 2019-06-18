import numpy as np
import scipy

import tensorflow as tf
from keras.models import Sequential,clone_model,Model
from keras.layers import Dense,Input
from keras.optimizers import Adam, SGD
from keras import regularizers
from keras.layers import Dropout
import keras.backend as K 
from keras.callbacks import TensorBoard
from scipy.interpolate import interp1d,griddata,interp2d
from scipy.interpolate import interpolate
from scipy import integrate
import matplotlib.pyplot as plt
from scipy import io
from fc_net import FullyConnectedNet

def GetGearRatio(gearnum,gear_ratios):
    maxGear = gear_ratios.size-1
    print(gearnum)
    gearnum[gearnum>maxGear] = maxGear
    gear = gear_ratios.ravel()[gearnum] 
    return gear  

def EngineMapModel(matFile):
#     xx, yy = np.meshgrid(EngSpeed,EngTorque)
	d = io.loadmat(matFile)
	eng_speed_points, eng_trq_points = np.meshgrid(d['eng_speed_rpm'],d['eng_torque'])
	minEff = np.nanmin(d['eng_eff'])
	d['eng_eff'][np.isnan(d['eng_eff'])]=10
	d['eng_eff'][np.where(eng_speed_points>=4000)]=10
	d['eng_eff'][np.where(eng_speed_points<=900)]=10
	d['eng_eff'][np.where(eng_trq_points>=250)]=10
	d['eng_eff'][np.where(d['eng_eff']<10)]=10
	data = np.vstack((eng_speed_points[:][:].ravel(),eng_trq_points[:][:].ravel())).T
	f = scipy.interpolate.NearestNDInterpolator(data, d['eng_eff'].ravel()*0.01)
	return f

def EmMapModel(matFile): #motor map
	#     xx, yy = np.meshgrid(motorSpeed,motorTorque)
	d = io.loadmat(matFile)
	em_speed_points,em_trq_points, = np.meshgrid(d['loss_N_rpm_x'], d['loss_T_y'])
	em_eff = d['efficiency_z']
	em_eff[np.isnan(em_eff)]=0.1
	em_eff[np.where(em_eff<0.1)]= 0.1
	data = np.vstack((em_speed_points[0:][:].ravel(),em_trq_points[0:][:].ravel())).T
	if not((em_trq_points<0).any()):
		data = np.vstack((data.ravel(),data.ravel())).T
		em_eff = np.tile(em_eff,2)
	f = scipy.interpolate.NearestNDInterpolator(data, em_eff.ravel())
	return f


# def UpdateGearArray(shift_signal,VehRPM,DemTorque,gear_ratios,gear_Array):
#     new_RPM = []
#     new_Trq = []
#     gear_Array = np.round( gear_Array + shift_signal.ravel() ).astype('int64')
#     print(gear_Array())
#     gr = GetGearRatio(gear_Array,gear_ratios)
#     new_RPM = VehRPM*gr
#     new_Trq = DemTorque/gr
#     return gear_Array,new_RPM,new_Trq

def UpdateGearArray(shift_signal,gear_ratios,gear_init):
    gear_list = []
    shift_signal = shift_signal.ravel()
    for i in shift_signal:
        gear = np.round(gear_init + i).astype('int64')
        gear_init = gear
        gear_list.append(gear)
        gr = GetGearRatio(gear_init,gear_ratios)
        
    return gear,gr

def GetWeightsArray(wLayered):
	'''
	wLayered: Output from model.get_weights
	'''
	wList = []
	if isinstance(wLayered,list):
		#print(wLayered)
		for wLayer in wLayered:
			wList += GetWeightsArray(wLayer)
	else:
		return wLayered.ravel().tolist()
	return wList

# def GetWeightsArray(neuralNetwork):
    # wList = []
    # wLayered = []
# #     for lay in neuralNetwork.layers:
    # wLayered.append(neuralNetwork.get_weights())
    # weights_gear_up_down = neuralNetwork.get_weights()
    
    # wList.append(np.concatenate((weights_gear_up_down[0].ravel(), weights_gear_up_down[1].ravel() ,weights_gear_up_down[2].ravel(), 
    # weights_gear_up_down[3].ravel(),weights_gear_up_down[4].ravel() ,weights_gear_up_down[5].ravel(),weights_gear_up_down[6].ravel(),
    # weights_gear_up_down[7].ravel(),     weights_gear_up_down[8].ravel(), weights_gear_up_down[9].ravel() ,weights_gear_up_down[9].ravel(),     weights_gear_up_down[10].ravel(),
    # weights_gear_up_down[11].ravel(),weights_gear_up_down[12].ravel(),weights_gear_up_down[13].ravel(),weights_gear_up_down[14].ravel(),
    # weights_gear_up_down[15].ravel(),weights_gear_up_down[16].ravel(),weights_gear_up_down[17].ravel())))
    # wArray = np.concatenate(tuple(wList))
# #     print((wArray))
    # return wArray


def SetWeightsIntoNN(w,neuralNetwork):
    iFirst=0
    iNext = 0
    wLayerStructured = neuralNetwork.get_weights()
    for ii,wArrayLayer in enumerate(wLayerStructured):
        iNext += wArrayLayer.size
        shapeArray = wArrayLayer.shape
        wLayerStructured[ii] = w[iFirst:iNext].reshape(shapeArray)
        iFirst = iNext
    neuralNetwork.set_weights(wLayerStructured)
    return neuralNetwork

def Create_Model(nInputs,listHiddenLayerNeurons,nOutputs,lr =0.001):
	model = Sequential()
	model.add(Dense(listHiddenLayerNeurons[0], input_dim = nInputs,activation='relu'))
	for nNeuronsLayer in listHiddenLayerNeurons[1:]:
		model.add(Dense(nNeuronsLayer, activation='relu'))
	model.add(Dense(nOutputs, activation=None)) # engine gear
	model.compile(optimizer=Adam(lr=lr),loss='mean_squared_error')
	model.summary()
	modelPython = FullyConnectedNet(listHiddenLayerNeurons,input_dim=nInputs,num_classes=nOutputs)
	modelPython.set_weights(model.get_weights())
	return model, modelPython

# def Create_Model(nInputs,listHiddenLayerNeurons=[],lr =0.001):
	# # model = Sequential()
	# inputs = Input(shape=(nInputs,))
	# x1 = Dense(3,kernel_regularizer=None,activity_regularizer=None,activation='relu')(inputs)
	# x2 = Dense(5, activation='relu', kernel_initializer='lecun_uniform')(x1)
	# x3 = Dense(2, activation='relu', kernel_initializer='lecun_uniform')(x2)
	# x4 = Dense(2, activation='relu', kernel_initializer='lecun_uniform')(x3)   
	# output1 = Dense(1, activation='tanh')(x4) # engine gear
	# output2 = Dense(1, activation='tanh')(x4)  # motor gear
	# output3 = Dense(1, activation='sigmoid')(x4)  # mode selection
	# output4 = Dense(1, activation='sigmoid')(x4)  # alpha #sigmoid
	# output5 = Dense(1, activation='sigmoid')(x4)  # beta  #sigmoid
	# '''
	# output1 = Dense(1, activation='tanh')(x4) # engine gear
	# output2 = Dense(1, activation='tanh')(x4)  # motor gear
	# output3 = Dense(1, activation='sigmoid')(x4)  # mode selection
	# output4 = Dense(1, activation='relu')(x4)  # alpha #sigmoid
	# output5 = Dense(1, activation='relu')(x4)  # beta  #sigmoid '''
	# model = Model(input=inputs, output=[output1,output2,output3,output4,output5])
	# model.compile(optimizer=Adam(lr=lr),loss='mean_squared_error')
	# model.summary()      
	# return model

def OutputNNToGear(NN,data):
	outData = (NN.predict(data))
# 	plt.plot(outData)
	return outData
	
def GetOptimalEfficiency(inputData,gear_ratios,mapEngine):
	effList = []
	maxGear = gear_ratios.size-1
	for shift in [-1,0,1]:
		current_Gear_Array = inputData['Gear'].values + shift
		current_Gear_Array[current_Gear_Array>maxGear] = maxGear
		newRPM = inputData['Vehicle Speed'].values*gear_ratios[current_Gear_Array]
		newTorque = inputData['Demand Torque'].values/gear_ratios[current_Gear_Array]
		effList.append(mapEngine(newRPM,newTorque))
	eff0 = effList[1]
	effMax = np.max(np.array(effList),axis=0)
	optimalShift = np.argmax(np.array(effList),axis=0)-1
	return effMax,eff0,effList
	
def VehicleToOptimizeForGA(vehicle):
    def OptimizeFunction(wPopulation):
        lossList = []
        for wNN in wPopulation:
            vehicle.ControllerNN = SetWeightsIntoNN(wNN,vehicle.ControllerNN)
            vehicle.RunSimulation()
            lossList.append(LossFunction(vehicle))
        loss = np.array(lossList)
        return loss
    return OptimizeFunction
	
	
def Kmph2RPS(VehSpeed,radius): # radius 
	return (VehSpeed/radius)/(2*np.pi)

def VehicleToOptimizeForGA(vehicle):
    def OptimizeFunction(wPopulation):
        lossList = []
        for wNN in wPopulation:
            vehicle.ControllerNN = SetWeightsIntoNN(wNN,vehicle.ControllerNN)
            vehicle.RunSimulation()
            lossList.append(LossFunction(vehicle))
        loss = np.array(lossList)
        return loss
    return OptimizeFunction

def LossFunction(vehicle):
	lossDistance = 1000 - vehicle.x[-1]/11000
	lossFuel = vehicle.fuelFlow.sum()*100*1000/vehicle.x[-1]
	lossSOC = np.maximum(vehicle.SOC_target - vehicle.soc[-1],0)*1
	if vehicle.completed: #  [minFuel to ~50]
		loss = lossSOC + lossFuel
	else: # ~[999]
		loss = lossDistance
	return loss
