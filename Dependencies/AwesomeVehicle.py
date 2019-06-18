from scipy import io
from scipy.interpolate import interpolate
import numpy as np
import Utilities_PHEV as UT
import matplotlib.pyplot as plt
import sys
from scipy.special import expit
'''
	A plug-in Vehicle model with 4 modes used for energy optimisation !!
	
'''
	
class PHEV(object):

	def __init__(self,velProfileFile,KerasModel,config={}):
		
		# Initialize Controller NN parameters
		self.ControllerNN = KerasModel
		self.wStructure = KerasModel.get_weights()
		self.nInputs = self.wStructure[0].shape[0]
		config.setdefault('nOutputs',5)
		# Mechanical Parameters
		
		config.setdefault('mass',1000)
		config.setdefault('cd',.999)
		config.setdefault('area',1.99)
		config.setdefault('g',9.81)
		config.setdefault('theta',0)
		config.setdefault('f_r',0.01)
		config.setdefault('p',1.225)
		config.setdefault('initEngGear',0)
		config.setdefault('gearRatios',np.array([1, 1, 1, 1, 1, 1, 1])) # gear ratios include final drive ratio)
		self.nGears = config['gearRatios'].size
		config.setdefault('radius',0.1)
		config.setdefault('engineMapMat','GEP3_MPplus.mat')
		self.EngineMap = UT.EngineMapModel(config['engineMapMat'])
		# Electrical parameters
		
		config.setdefault('motor_eff','dan2.mat')
		self.MotorMap = UT.EmMapModel(config['motor_eff'])
		config.setdefault('em_Gear',np.array([1, 1, 1 ]) )
		config.setdefault('emFixRatio', 1)
		config.setdefault('initEmGear',0)
		config.setdefault('SOC_min',15)
		config.setdefault('SOC_target',95)
		config.setdefault('SOC_init',95)
		config.setdefault('Q_heat',25e6*0.77)  # J/l
		config.setdefault('battery_capacity',50)
		config.setdefault('nCellsHVBattery',50)
		self.batteryEnergy = config['battery_capacity']*3600*config['nCellsHVBattery']
		
		config.setdefault('batteryModelMat','SOC_HV_51Amh.mat')
		dataBatteryMat = io.loadmat(config['batteryModelMat'])
		self.BatterySOCToV = interpolate.interp1d(dataBatteryMat['SOC'].ravel(),dataBatteryMat['HV'].ravel()*config['nCellsHVBattery'])
		
		# Drive Cycle
		self.velProfileFile = velProfileFile
		for key,value in config.items():
			setattr(self,key,value)
		self.UpdateCycleDynamics()
		pass
	
	
	def UpdateCycleDynamics(self):
		tmpData = np.loadtxt(self.velProfileFile)
		self.time, self.vehSpeed = tmpData[:,0], tmpData[:,1]
		self.timeStep = np.diff(self.time[:2])[0]#  Assuming constant timeStep
		self.x = np.cumsum(self.vehSpeed*self.timeStep)
		self.acc = np.gradient(self.vehSpeed,self.time)
		self.dragForce = .5*self.p*self.cd*self.area*(self.vehSpeed**2)
		self.gravityForce = self.mass*self.f_r*np.cos(self.theta)+self.mass*self.g*np.sin(self.theta)
		self.demTrq = self.radius*(self.dragForce + self.mass*self.acc + self.gravityForce)
		self.whlSpeed = UT.Kmph2RPS(self.vehSpeed,self.radius)
		pass
		
	
	def RunSimulation(self):
		# start
		self.ii = 0
		nPoints = self.time.size
		# Signals required and initialization
		listSignalsRequired = ['engGear','emGear','soc','normaLizedSOC','emShiftGear','engShiftGear','emRPM',
								'emTorque','engRPM','engTorque','emEff','engEff','volt','fuelFlow','regenerationMode','alpha','beta',
								'normalizedSOCLimit','normalizedSOCTarget','mode','elecCurrent','powerMech','powerEm','engPower','energyEm']
		for key in listSignalsRequired:
			setattr(self,key,np.zeros((nPoints,)))
		self.engGear[0] = self.initEngGear
		self.emGear[0] = self.initEmGear
		self.soc[0] = self.SOC_init
		self.outputNN = np.zeros((nPoints,self.nOutputs))
		self.completed = False
		try:
			while self.vehSpeed[self.ii]>=0 and self.ii<nPoints-1:
				self.ii += 1
				#if not(self.vehSpeed[self.ii]==0):
				self.normalizedSOCLimit[self.ii] = self.soc[self.ii-1]/self.SOC_min
				self.normalizedSOCTarget[self.ii] = self.soc[self.ii-1]/self.SOC_target
				inputDataNN = np.array([self.normalizedSOCLimit[self.ii],self.normalizedSOCTarget[self.ii],self.whlSpeed[self.ii]/10,self.demTrq[self.ii]/150,self.emGear[self.ii-1],self.engGear[self.ii-1]]).reshape((1,self.nInputs)).copy()
				self.outputNN[self.ii,:] = np.array(self.ControllerNN.predict(inputDataNN)).ravel()
				self.emShiftGear[self.ii], self.engShiftGear[self.ii], self.mode[self.ii], self.alpha[self.ii], self.beta[self.ii] = self.OutputNNToGear(self.outputNN[self.ii,:])
				self.Engine_Mode(0)
				self.EV_Mode(0)
				self.UpdateGearArray()
				#print('The em gear array is',self.emGear,'Engine:',self.engGear)
				if self.mode[self.ii]==0:
					self.EV_Mode(self.demTrq[self.ii])
				elif self.mode[self.ii]==1:                # Change later
					self.Engine_Mode(self.demTrq[self.ii])
				elif self.mode[self.ii]==2:
					self.Enginemode_With_Charging()
				elif self.mode[self.ii]==3:
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
		except:
			print('Uncompleted simulation. Total distance: ', self.x[self.ii])
		if self.completed:
			print('completed simulation',self.x[-1], self.soc[-1])
		pass
		
	def CheckConstraints(self):
		if self.soc[self.ii]< self.SOC_min:
			sys.exit()
		pass
	
	@staticmethod
	def OutputNNToGear(outData):
		#print('Output is :',outData)
		
		emShiftGear = np.tanh(outData[0])*2    
		engShiftGear = np.tanh(outData[1])
		if emShiftGear < -0.5:
			emShiftGear = -2
		elif emShiftGear <= 0.5:
			emShiftGear = 0
		else:
			emShiftGear = 2
			
		if engShiftGear < -0.5:
			engShiftGear = -1
		elif engShiftGear <= 0.5:
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

		elif outputMode > -0.5 and outputMode<= 0:

			mode = 1
		elif  outputMode > 0 and outputMode<=.5:

			mode = 2
		else:
			mode = 3
			
		return emShiftGear,engShiftGear,mode,outAlpha,outBeta
	
	def UpdateGearArray(self):
		gearEngChange = self.engGear[self.ii-1] + self.engShiftGear[self.ii]
		self.engGear[self.ii] = np.clip(gearEngChange,1,self.nGears)
		gearEmChange = (min([self.emGear[self.ii-1] + self.emShiftGear[self.ii], self.engGear[self.ii]]) // 2) * 2 #  engGear[self.ii-1]
		self.emGear[self.ii] = np.clip(gearEmChange,2, 6)
		pass
	
	#no fuel consumption, no battery cahrging, propulsion via even gears
	def EV_Mode(self,demTrq):
		self.emRPM[self.ii] = self.whlSpeed[self.ii]*self.gearRatios[int(self.emGear[self.ii]-1)]*60*self.emFixRatio
		self.emTorque[self.ii] = demTrq/(self.gearRatios[int(self.emGear[self.ii]-1)]*self.emFixRatio)
		self.CalculateSOC(self.emRPM[self.ii],self.emTorque[self.ii])
		pass
		
	def CalculateSOC(self,emRPM,emTorque):
		self.emEff[self.ii] = self.MotorMap(emRPM,self.emTorque[self.ii])
		#print(self.soc[int(self.ii-1)])
		self.volt[self.ii] = self.BatterySOCToV(self.soc[self.ii-1])
		self.powerMech[self.ii] = 2*np.pi*emRPM*emTorque/60
		self.powerEm[self.ii] = self.powerMech[self.ii]/self.emEff[self.ii] #motor efficiency
		self.energyEm[self.ii] = self.timeStep*self.powerEm[self.ii]
		deltaSOC = (self.energyEm[self.ii]/self.batteryEnergy)*100
		self.soc[self.ii] = self.soc[self.ii-1] - deltaSOC
		self.elecCurrent[self.ii] = self.powerEm[self.ii] /self.volt[self.ii]
		if deltaSOC>0:  self.regenerationMode[self.ii] = 1
		pass
	
	def Engine_Mode(self,demTrq): #without charging the battery
		self.engRPM[self.ii] = self.whlSpeed[self.ii]*self.gearRatios[int(self.engGear[self.ii]-1)]*60
		self.engTorque[self.ii] = demTrq/self.gearRatios[int(self.engGear[self.ii]-1)]
		if demTrq>0:
			self.engEff[self.ii] = self.EngineMap(self.engRPM[self.ii],self.engTorque[self.ii])
			self.engPower[self.ii] = 2*np.pi*self.whlSpeed[self.ii]*demTrq
			self.fuelFlow[self.ii] = self.engPower[self.ii]/self.engEff[self.ii]/self.Q_heat
			
			# self.EV_Mode(0) #because soc is consuming in this mode
			'''
			self.engPower[self.ii] = 2*np.pi*self.whlSpeed[self.ii]*demTrq
			self.fuelFlow[self.ii] = self.EngineMap(self.engRPM[self.ii],self.engTorque[self.ii])
			'''
		    # Else braking which is out of the scope. If thermal analysis might be interesting.
		pass

	def Enginemode_With_Charging(self): #hybrid mode, motor acts as generator
		if (self.engGear[self.ii] % 2 == 0) and self.demTrq[self.ii]>0:
			regenerationTrqElMotor = self.alpha[self.ii]*self.demTrq[self.ii]  # Maybe adding
			demTrqEngine = regenerationTrqElMotor + self.demTrq[self.ii]
			self.Engine_Mode(demTrqEngine)
			# Load to Generator
			self.EV_Mode(-regenerationTrqElMotor)
		elif self.demTrq[self.ii]<0:
			self.EV_Mode(self.demTrq[self.ii])
		else:
			self.Engine_Mode(self.demTrq[self.ii])
		pass

	def Power_Mode(self): #engine through all gears and motor through even gears
		demTorqueEng = self.demTrq[self.ii]*self.beta[self.ii]
		demToqueEm = self.demTrq[self.ii]*(1-self.beta[self.ii])
		self.Engine_Mode(demTorqueEng)
		self.EV_Mode(demToqueEm)
		pass
