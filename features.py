import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn import preprocessing
import math

timeseries = np.loadtxt('C:/Users/matthew/Desktop/Natural_Gas_data_N35_reduced_data/processed_data.txt')
numFeatures = 36
featureMatrix = np.zeros((timeseries.shape[1]-1,numFeatures))

time = timeseries[:,0]
tFinal = time[len(time)-1]
timeBetweenSamples = time[1] - time[0]
jFinal = int(math.floor(tFinal/timeBetweenSamples))

for i in range(1,timeseries.shape[1]):
	currentSeries = timeseries[:,i]
	currentSeries = currentSeries.reshape((currentSeries.shape[0],1))
	
	normalizedData = np.zeros((currentSeries.shape[0],currentSeries.shape[1]))
	derivative = np.zeros((currentSeries.shape[0],currentSeries.shape[1]))
	normalizedDerivative = np.zeros((currentSeries.shape[0],currentSeries.shape[1]))
	secondDerivative = np.zeros((currentSeries.shape[0],currentSeries.shape[1]))
	
	# Create derivative vector
	for j in range(5, currentSeries.shape[0]-5):
		derivative[j][0] = (currentSeries[j][0] - currentSeries[j-5][0])/(5*timeBetweenSamples)
		
	# Create second derivative vector
	for j in range(5, derivative.shape[0]-5):
		secondDerivative[j][0] = (derivative[j][0] - derivative[j-5][0])/(5*timeBetweenSamples)
	
	# Find max value
	maxValue = currentSeries[0][0]
	for j in range(currentSeries.shape[0]):
		if (currentSeries[j][0] > maxValue):
			 maxValue = currentSeries[j][0]
			 jmax = j
			 tmax = j*timeBetweenSamples
	featureMatrix[i-1][0] = maxValue
	
	# Find area under curve
	area = 0
	for j in range(currentSeries.shape[0]):
		area = area + timeBetweenSamples * currentSeries[j][0]
	featureMatrix[i-1][1] = area	

	# Normalize
	for j in range(currentSeries.shape[0]):
		normalizedData[j][0] = currentSeries[j][0]/maxValue
	normalizedData = normalizedData.reshape((normalizedData.shape[0],1))
		
	for j in range(5, currentSeries.shape[0]-5):
		normalizedDerivative[j][0] = (normalizedData[j][0] - normalizedData[j-5][0])/(5*timeBetweenSamples)
		
	# Find 95% value
	for j in range(normalizedData.shape[0]):
		if (normalizedData[j][0]>0.949):
			v95 = currentSeries[j][0]
			t95 = j*timeBetweenSamples
			break
			
	# Find 5% value
	for j in range(normalizedData.shape[0]):
		if (normalizedData[j][0]>0.049):
			v5 = currentSeries[j][0]
			t5 = j*timeBetweenSamples
			break
			
	# Find 75% downslope value
	for j in range( jmax,  normalizedData.shape[0]):
		if (normalizedData[j][0] < 0.751):
			norm_t75_down = j*timeBetweenSamples
			break
			
	# Find 10% value
	for j in range(normalizedData.shape[0]):
		if (normalizedData[j][0] > 0.099):
			norm_t10 = j*timeBetweenSamples
			break
			
	# Find 20% value
	for j in range(normalizedData.shape[0]):
		if (normalizedData[j][0] > 0.199):
			norm_t20 = j*timeBetweenSamples
			break
		
	# Find 30% value
	for j in range(normalizedData.shape[0]):
		if (normalizedData[j][0] > 0.299):
			norm_t30 = j*timeBetweenSamples
			break
		
	# Find 40% value
	for j in range(normalizedData.shape[0]):
		if (normalizedData[j][0] > 0.399):
			norm_t40 = j*timeBetweenSamples
			break
		
	# Find 50% value
	for j in range(normalizedData.shape[0]):
		if (normalizedData[j][0] > 0.499):
			norm_t50 = j*timeBetweenSamples
			break
		
	# Find 60% value
	for j in range(normalizedData.shape[0]):
		if (normalizedData[j][0] > 0.599):
			norm_t60 = j*timeBetweenSamples
			break
			
	# Find 70% value
	for j in range(normalizedData.shape[0]):
		if (normalizedData[j][0] > 0.699):
			norm_t70 = j*timeBetweenSamples
			break
		
	# Find 80% value
	for j in range(normalizedData.shape[0]):
		if (normalizedData[j][0] > 0.799):
			norm_t80 = j*timeBetweenSamples
			break
		
	# Find 90% value
	for j in range(normalizedData.shape[0]):
		if (normalizedData[j][0] > 0.899):
			norm_t90 = j*timeBetweenSamples
			break

	# Find 75% value
	for j in range(normalizedData.shape[0]):
		if (normalizedData[j][0] > 0.749):
			norm_t75 = j*timeBetweenSamples
			break
			
	# Find 90% downslope value
	for j in range( jmax,  normalizedData.shape[0]):
		if (normalizedData[j][0] < 0.901):
			norm_t90_down = j*timeBetweenSamples
			break
			
	# Find 80% downslope value
	for j in range( jmax,  normalizedData.shape[0]):
		if (normalizedData[j][0] < 0.801):
			norm_t80_down = j*timeBetweenSamples
			break
			
	# Find 70% downslope value
	for j in range( jmax,  normalizedData.shape[0]):
		if (normalizedData[j][0] < 0.701):
			norm_t70_down = j*timeBetweenSamples
			break
					
	# Find 60% downslope value
	for j in range( jmax,  normalizedData.shape[0]):
		if (normalizedData[j][0] < 0.601):
			norm_t60_down = j*timeBetweenSamples
			break
			
	# Find 50% downslope value
	for j in range( jmax,  normalizedData.shape[0]):
		if (normalizedData[j][0] < 0.501):
			norm_t50_down = j*timeBetweenSamples
			break
	
	# Find slope at 10% of max
	upSlope10 = derivative[math.floor(0.1*jmax)][0]
	
	# Find slope at 20% of max
	upSlope20 = derivative[math.floor(0.2*jmax)][0]
	
	# Find slope at 30% of max
	upSlope30 = derivative[math.floor(0.3*jmax)][0]
	
	# Find slope at 40% of max
	upSlope40 = derivative[math.floor(0.4*jmax)][0]
	
	# Find slope at 50% of max
	upSlope50 = derivative[math.floor(0.5*jmax)][0]
			
	# Find slope at 60% of max
	upSlope60 = derivative[math.floor(0.6*jmax)][0]
	
	# Find slope at 70% of max
	upSlope70 = derivative[math.floor(0.7*jmax)][0]
	
	# Find slope at 80% of max
	upSlope80 = derivative[math.floor(0.8*jmax)][0]
	
	# Find slope at 90% of max
	upSlope90 = derivative[math.floor(0.9*jmax)][0]
	
	# Find slope at 25% of recovery
	downSlope25 = derivative[math.floor(jmax + 0.25*(jFinal-jmax))][0]
	
	# Find slope at 50% of recovery
	downSlope50 = derivative[math.floor(jmax + 0.50*(jFinal-jmax))][0]
	
	# Find slope at 75% of recovery
	downSlope75 = derivative[math.floor(jmax + 0.75*(jFinal-jmax))][0]
	
	# Find nomalized slope at 50% of max
	normalizedDerivative50 = normalizedDerivative[math.floor(0.5*jmax)][0]
	
	vEnd = currentSeries[currentSeries.shape[0]-1][0]	
	normalizedVEnd = normalizedData[normalizedData.shape[0]-1][0]	
	slope95 = (v95-v5)/(t95-t5)
	slopeEnd = (vEnd - v95) / (currentSeries.shape[0]*timeBetweenSamples - t95)
	normalizedSlope95 = 0.9/(t95-t5)
	normalizedSlopeEnd = (normalizedVEnd - 0.95) / (normalizedData.shape[0]*timeBetweenSamples - t95)
	
	featureMatrix[i-1][2] = slope95
	featureMatrix[i-1][3] = slopeEnd
	featureMatrix[i-1][4] = upSlope10
	featureMatrix[i-1][5] = upSlope20
	featureMatrix[i-1][6] = upSlope30
	featureMatrix[i-1][7] = upSlope40
	featureMatrix[i-1][8] = upSlope50
	featureMatrix[i-1][9] = upSlope60
	featureMatrix[i-1][10] = upSlope70
	featureMatrix[i-1][11] = upSlope80
	featureMatrix[i-1][12] = upSlope90
	featureMatrix[i-1][13] = downSlope25
	featureMatrix[i-1][14] = downSlope50
	featureMatrix[i-1][15] = downSlope75
	featureMatrix[i-1][16] = area/maxValue
	featureMatrix[i-1][17] = tmax
	featureMatrix[i-1][18] = normalizedDerivative50
	featureMatrix[i-1][19] = normalizedSlope95
	featureMatrix[i-1][20] = normalizedSlopeEnd
	featureMatrix[i-1][21] = norm_t10
	featureMatrix[i-1][22] = norm_t20
	featureMatrix[i-1][23] = norm_t30
	featureMatrix[i-1][24] = norm_t40
	featureMatrix[i-1][25] = norm_t50
	featureMatrix[i-1][26] = norm_t60
	featureMatrix[i-1][27] = norm_t70
	featureMatrix[i-1][28] = norm_t80
	featureMatrix[i-1][29] = norm_t90
	featureMatrix[i-1][30] = norm_t90_down
	featureMatrix[i-1][31] = norm_t80_down
	featureMatrix[i-1][32] = norm_t75_down
	featureMatrix[i-1][33] = norm_t70_down
	featureMatrix[i-1][34] = norm_t60_down
	featureMatrix[i-1][35] = norm_t50_down

np.savetxt(r'C:/Users/matthew/Desktop/Natural_Gas_data_N35_reduced_data/all_features_unnormalized.csv', featureMatrix, fmt='%.10f', delimiter=',')

scaler = preprocessing.StandardScaler().fit(featureMatrix)
normalizedFeatures = scaler.transform(featureMatrix)

np.savetxt(r'C:/Users/matthew/Desktop/Natural_Gas_data_N35_reduced_data/all_features_normalized.csv', normalizedFeatures, fmt='%.10f', delimiter=',')

