'''

The code is an pure electric vehicle simulator, that run a complete New European Drive Cycyle or other drive cycles. The code works in collabration with a NN that gives 
the optimal shift signals for transmission. The code is a part of my master thesis and was carried out at CEVT AB. The data used is the integral part of
 the compnay and hence has been replaced here, so that the data rights are not violated here :)
 Curious?
     Write to me -anup.padaki@gmail.com

'''

from scipy import io
from scipy.interpolate import interpolate
import numpy as np
import Utilities_PHEV as UT
import matplotlib.pyplot as plt
import sys
from scipy.special import expit

'''
	vehSpeed - m/s
	whlSpeed - Revolutions per second
'''
	
class ELE(object):

	def __init__(self,velProfileFile,KerasModel,config={}):
		
		# Initialize Controller NN parameters
		self.ControllerNN = KerasModel
		self.wStructure = KerasModel.get_weights()
		self.nInputs = self.wStructure[0].shape[0]
		config.setdefault('nOutputs',5)
		# Mechanical Parameters
		
		config.setdefault('mass',1800)
		config.setdefault('cd',.36)
		config.setdefault('area',2.53)
		config.setdefault('g',9.81)
		config.setdefault('theta',0)
		config.setdefault('f_r',0.01)
		config.setdefault('p',1.5)
		config.setdefault('initEngGear',0)
		config.setdefault('engGearRatios',np.array([1, 1, 1, 1, 1, 1, 1])) # gear ratios include final drive ratio)
		self.nGears = config['engGearRatios'].size
		config.setdefault('radius',0.25)
		config.setdefault('engineMapMat','GEP3_MPplus.mat')
		self.EngineMap = UT.EngineMapModel(config['engineMapMat'])
		config.setdefault('transmissionMap','loss_lookup_table.xlsx')
		self.TransimissionEfficiencyMap = UT.TransmmissionMap(config['transmissionMap'],config['engGearRatios'])
		# Electrical parameters
		
		config.setdefault('motor_eff','dan2.mat')
		self.MotorMap = UT.EmMapModel(config['motor_eff'])
		config.setdefault('emFixRatio', 1.25)
		config.setdefault('initEmGear',0)
		config.setdefault('SOC_min',15)
		config.setdefault('SOC_target',95)
		config.setdefault('SOC_init',95)
		config.setdefault('Q_heat',41.206e6*0.748)  # J/l 0.748
		config.setdefault('cellResistance',1.64e-3) # Ohm
		config.setdefault('battery_capacity',15) 
		config.setdefault('nCellsHVBattery',100)
		self.batteryEnergy = config['battery_capacity']*3.600e6 
		self.batteryResistance = config['cellResistance']*config['nCellsHVBattery']
		
		config.setdefault('batteryModelMat','SOC_HV_51Amh.mat')
		dataBatteryMat = io.loadmat(config['batteryModelMat'])
		self.BatterySOCToV = interpolate.interp1d(dataBatteryMat['SOC'].ravel(),dataBatteryMat['HV'].ravel()*config['nCellsHVBattery'])
		
		# Drive Cycle
		self.velProfileFile = velProfileFile
		for key,value in config.items():
			setattr(self,key,value)
		self.emGearRatios = self.engGearRatios#*1.53
		self.UpdateCycleDynamics()
		pass
	
	
	def UpdateCycleDynamics(self):
		tmpData = np.loadtxt(self.velProfileFile)
		self.time, self.vehSpeed = tmpData[:,0], tmpData[:,1]
		self.timeStep = np.diff(self.time[:2])[0]#  Assuming constant timeStep
		self.x = np.cumsum(self.vehSpeed*self.timeStep)
		self.acc = np.gradient(self.vehSpeed,self.time)
		#self.dragForce = .5*self.p*self.cd*self.area*(self.vehSpeed**2)
		#self.gravityForce = self.mass*self.f_r*np.cos(self.theta)+self.mass*self.g*np.sin(self.theta)
		#self.totalForce = self.dragForce + self.mass*self.acc + self.gravityForce
		#self.demTrq = self.radius*(self.dragForce + self.mass*self.acc + self.gravityForce)
		RLC = lambda v: 140.15 + 0.4135*v + 0.03826*v**2 # depends on mass
		#self.totalForce = self.dragForce + self.mass*self.acc + self.gravityForce
		self.totalForce = RLC(self.vehSpeed*3.6) + self.mass*self.acc
		self.demTrq = self.radius*self.totalForce
		self.whlSpeed = UT.Kmph2RPS(self.vehSpeed,self.radius)
		self.powerRequired = self.totalForce*self.vehSpeed
		pass
		
	
	def RunSimulation(self):
		# start
		self.ii = 0
		nPoints = self.time.size
		# Signals required and initialization
		listSignalsRequired = ['engGear','emGear','soc','normaLizedSOC','emShiftGear','engShiftGear','emRPM',
								'emTorque','engRPM','engTorque','emEff','engEff','volt','fuelFlow','regenerationMode','alpha','beta',
								'normalizedSOCLimit','normalizedSOCTarget','mode','elecCurrent','powerMech','powerEm','engPower','energyEm','tEff','powerMotor']
		for key in listSignalsRequired:
			setattr(self,key,np.zeros((nPoints,)))
		self.engGear[0] = self.initEngGear
		self.emGear[0] = self.initEmGear
		self.soc[0] = self.SOC_init
		self.outputNN = np.zeros((nPoints,self.nOutputs))
		self.completed = False
		while self.vehSpeed[self.ii]>=0 and self.ii<nPoints-1:
			self.ii += 1
			#self.gearOpt = [1]
			_,_,self.gearOpt,_,_ = UT.GetOptimalGear(self.whlSpeed[self.ii]*60,self.demTrq[self.ii],self.emGearRatios,self.MotorMap)
			self.gearOpt = self.gearOpt
			np.array(self.gOpt.append(self.gearOpt))
			if self.gearOpt[0]>self.emGear[self.ii-1]:
				emShiftGear = 2
			elif self.gearOpt[0] == self.emGear[self.ii-1]:
				emShiftGear = 0
			else:
				emShiftGear = -2
			mode = 0
			engShiftGear = 1
			#self.Engine_Mode(0)
			self.EV_Mode(0)
			self.UpdateGearArray(engShiftGear,emShiftGear)
			
			if mode==0:
				self.mode[self.ii] = 0
				self.EV_Mode(self.demTrq[self.ii])
			elif mode==1:                # Change later
				self.mode[self.ii] = 1
				self.Engine_Mode(self.demTrq[self.ii])
			elif mode==2:
				self.mode[self.ii] = 2
				self.Enginemode_With_Charging()
			elif mode==3:
				self.mode[self.ii] = 3
				self.Power_Mode()
			# End while loop
			'''
			else:
				self.UpdateGearArray()
				self.soc[self.ii] = self.soc[self.ii-1]
				self.emShiftGear[self.ii], self.engShiftGear[self.ii], self.mode[self.ii], self.alpha[self.ii], self.beta[self.ii] = 0,0,self.mode[self.ii-1],self.alpha[self.ii-1], self.beta[self.ii-1]
			'''
			self.CheckConstraints()
			self.completed = True
		if self.completed:
			print('completed simulation',self.x[-1], self.soc[-1])
		pass
		
	def CheckConstraints(self):
		if self.soc[self.ii]< self.SOC_min:
		    print('completed simulation',self.x[self.ii])
		    sys.exit()
			
		pass
	
	@staticmethod
	def OutputNNToGear(outData):
		#print('Output is :',outData)
		
		emShiftGear = np.tanh(outData[0])#*2    
		engShiftGear = np.tanh(outData[1])
		#print('The NN output for eng gear looks like',outData[1])
		if emShiftGear <= -0.33: # -1 to 1
			emShiftGear = -2
		elif emShiftGear <= 0.33:
			emShiftGear = 0
		else:
			emShiftGear = 2
		
		if engShiftGear <= -0.333:
			engShiftGear = -1
		elif engShiftGear <= 0.33:
			engShiftGear = 0
		else:
			engShiftGear = 1
			
		# emShiftGear = np.round(np.tanh(outData[0]))*2    # Rounding was just returning values of -0 and -1 
		# engShiftGear = np.round(np.tanh(outData[1]))
		#print('The shift gear array signal is: Motor',emShiftGear,'Engine:',engShiftGear)
		outputMode = np.tanh(outData[2])
		outAlpha = np.maximum(outData[3],0)  # ReLu
		outBeta = expit(outData[4])

		if outputMode <= -0.5:
			mode = 0
		elif outputMode<= 0:
			mode = 1
		elif outputMode<=.5:
			mode = 2
		else:
			mode = 3
			
		return emShiftGear,engShiftGear,mode,outAlpha,outBeta
	
	def UpdateGearArray(self,engShiftGear,emShiftGear):
		self.emShiftGear[self.ii] = emShiftGear
		self.engShiftGear[self.ii] = engShiftGear
		gearEngChange = self.engGear[self.ii-1] + engShiftGear
		self.engGear[self.ii] = np.clip(gearEngChange,1,self.nGears)
		gearEmChange = (min([self.emGear[self.ii-1] + emShiftGear, self.engGear[self.ii-1]]) // 2) * 2 #  engGear[self.ii-1]
		self.emGear[self.ii] = np.clip(gearEmChange,2, 6)
		if self.time[self.ii]>=836 and self.time[self.ii]<=1152:
			self.emGear[self.ii] = 4
		else:
			self.emGear[self.ii] = 2
		pass
	
	#no fuel consumption, no battery cahrging, propulsion via even gears
	def EV_Mode(self,demTrq):
		self.emRPM[self.ii] = self.whlSpeed[self.ii]*self.emGearRatios[int(self.emGear[self.ii]-1)]*60*self.emFixRatio
		self.emTorque[self.ii] = (demTrq/(self.emGearRatios[int(self.emGear[self.ii]-1)]*self.emFixRatio))
		self.tEff[self.ii] = self.TransimissionEfficiencyMap(self.emGear[self.ii],self.whlSpeed[self.ii]*60,np.abs(demTrq))
		if demTrq > 0:
			#print('Motor Gear',self.emGear[self.ii],'T eff is',self.tEff[self.ii])
			self.emTorque[self.ii] = self.emTorque[self.ii]/self.tEff[self.ii]
		else:
			self.emTorque[self.ii] = self.emTorque[self.ii]*self.tEff[self.ii]
		self.powerMotor[self.ii] = 2*np.pi*self.emRPM[self.ii]*self.emTorque[self.ii]/60
		#print('The em RPM and Em torque at',self.ii,'is',self.emRPM[self.ii],'and',self.emTorque[self.ii])
		self.CalculateSOC(self.emRPM[self.ii],self.emTorque[self.ii])
		pass
		
	def CalculateSOC(self,emRPM,emTorque):
		#print('EM RPM is:',emRPM,'em Torque is:',emTorque,'timestep is',self.ii)
		self.emEff[self.ii] = self.MotorMap((emRPM,emTorque))
		#print(self.soc[int(self.ii-1)])
		self.volt[self.ii] = self.BatterySOCToV(self.soc[self.ii-1])
		if emTorque > 0 :
			self.powerEm[self.ii] = self.powerMotor[self.ii]/self.emEff[self.ii]  #motor efficiency
		else:
			self.powerEm[self.ii] = self.powerMotor[self.ii]*self.emEff[self.ii]	
		self.powerMech[self.ii] += self.powerEm[self.ii]
		self.energyEm[self.ii] = self.timeStep*self.powerEm[self.ii]
		self.elecCurrent[self.ii] = self.powerEm[self.ii] /self.volt[self.ii]
		batteryLoss = np.abs(self.elecCurrent[self.ii]**2) * self.batteryResistance + 354  # Resistance battery assumed to be 0.17 ohm
		deltaSOC = (self.energyEm[self.ii] + batteryLoss)*100/self.batteryEnergy
		self.soc[self.ii] = self.soc[self.ii-1] - deltaSOC
		
		if deltaSOC>0:  
			self.regenerationMode[self.ii] = 1
		pass
	
	def Engine_Mode(self,demTrq): #without charging the battery
		#demTrq = demTrq/self.TransimissionEfficiencyMap(self.engGear[self.ii],self.engRPM[self.ii],self.engTorque[self.ii])
		self.engRPM[self.ii] = self.whlSpeed[self.ii]*self.engGearRatios[int(self.engGear[self.ii]-1)]*60
		self.engTorque[self.ii] = demTrq/self.engGearRatios[int(self.engGear[self.ii]-1)]
		self.tEff[self.ii] = self.TransimissionEfficiencyMap(self.engGear[self.ii],self.whlSpeed[self.ii]*60,np.abs(demTrq))
		if demTrq > 0:
			#print('Motor Gear',self.emGear[self.ii],'T eff is',self.tEff[self.ii])
			self.engTorque[self.ii] = self.engTorque[self.ii]/self.tEff[self.ii]
			self.engEff[self.ii] = self.EngineMap(self.engRPM[self.ii],self.engTorque[self.ii])
			self.engPower[self.ii] = 2*np.pi*self.engRPM[self.ii]*self.engTorque[self.ii]/60
			self.powerMech[self.ii] += self.engPower[self.ii]/self.engEff[self.ii]
			self.fuelFlow[self.ii] = self.engPower[self.ii]/self.engEff[self.ii]/self.Q_heat
		else:
			self.engTorque[self.ii] = self.engTorque[self.ii]*self.tEff[self.ii]
			self.EV_Mode(demTrq)
			'''if demTrq>0:
				if self.engGear[self.ii] != 0:
					self.tEff[self.ii] = np.nanmax(np.hstack((self.TransimissionEfficiencyMap(self.engGear[self.ii],self.engRPM[self.ii],self.engTorque[self.ii]), 0.7)))
					demTrq = demTrq/self.tEff[self.ii]
				else:
					demTrq = 0
				self.engRPM[self.ii] = self.whlSpeed[self.ii]*self.engGearRatios[int(self.engGear[self.ii]-1)]*60
				self.engTorque[self.ii] = demTrq/self.engGearRatios[int(self.engGear[self.ii]-1)]'''
		
			#self.mode[self.ii] = 1
			'''self.engEff[self.ii] = self.EngineMap(self.engRPM[self.ii],self.engTorque[self.ii])
			self.engPower[self.ii] = 2*np.pi*self.engRPM[self.ii]*self.engTorque[self.ii]/60
			self.powerMech[self.ii] += self.engPower[self.ii]/self.engEff[self.ii]
			self.fuelFlow[self.ii] = self.engPower[self.ii]/self.engEff[self.ii]/self.Q_heat'''
			
			# self.EV_Mode(0) #because soc is consuming in this mode
			'''
			self.engPower[self.ii] = 2*np.pi*self.whlSpeed[self.ii]*demTrq
			self.fuelFlow[self.ii] = self.EngineMap(self.engRPM[self.ii],self.engTorque[self.ii])
			'''
		    # Else braking which is out of the scope. If thermal analysis might be interesting.
		#else:  # Could be fishy. We may want to set it to elif <0
			#self.mode[self.ii] = 0
			#self.EV_Mode(demTrq)
		pass

	def Enginemode_With_Charging(self): #hybrid mode, motor acts as generator
		if (self.engGear[self.ii] % 2 == 0) and self.demTrq[self.ii]>0:
			regenerationTrqElMotor = self.alpha[self.ii]*self.demTrq[self.ii]  # Maybe adding
			demTrqEngine = regenerationTrqElMotor + self.demTrq[self.ii]
			self.Engine_Mode(demTrqEngine)
			# Load to Generator
			self.EV_Mode(-regenerationTrqElMotor)
			#self.beta[self.ii] = 0
		elif self.demTrq[self.ii]<=0:
			#self.alpha[self.ii] = 0
			#self.mode[self.ii] = 0
			self.EV_Mode(self.demTrq[self.ii])
		else:
			#self.mode[self.ii] = 1
			#self.alpha[self.ii] = 0
			self.Engine_Mode(self.demTrq[self.ii])
		pass

	def Power_Mode(self): #engine through all gears and motor through even gears
		if self.demTrq[self.ii]>0:
			demTorqueEng = self.demTrq[self.ii]*self.beta[self.ii]
			demToqueEm = self.demTrq[self.ii]*(1-self.beta[self.ii])
			self.Engine_Mode(demTorqueEng)
			self.EV_Mode(demToqueEm)
		else:
			#self.mode[self.ii] = 0
			self.beta[self.ii] = 0
			self.EV_Mode(self.demTrq[self.ii])
		pass

# CarMaker demamnd torque is found by using Gas pedal and vehicle velocity from a 2D lookup table
# Battery capacity is 