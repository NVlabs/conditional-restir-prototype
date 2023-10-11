import OpenEXR
import numpy as np
import scipy.ndimage.filters
import Imath
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
import csv
import cupy as cp

def crop_image(img):
	#return img[325:625, 1200:1500, :] # Kitchen2
	#return img[25:275, 375:650, :] # Kitchen2
	#return img[75:350, 300:700, :] # VeachAjar
	#return img[100:250, 750:1020, :] # veach_bidir
	#return img[800:1100, 700:1000, :] # veach_bidir
	#return img[250:650, 50:560, :] # Kitchen
	#return img[700:1000, 600:900, :] # Kitchen
	#return img[700:1000, 675:975, :] # Cornell
	#return img[600:1000, 100:600, :] # Cornell
	return img

def get_exr_rgb(path, crop=False):
	I = OpenEXR.InputFile(path)
	dw = I.header()['displayWindow']
	size = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)
	data = [np.fromstring(c, np.float32).reshape(size) for c in I.channels('RGB', Imath.PixelType(Imath.PixelType.FLOAT))]
	img = cp.dstack(data)
	#img = np.dstack(data)
	#img = cp.clip(img, 0, 1)
	#img = cp.where(img<=0.0031308, 12.92*img, 1.055*np.power(img, 1/2.4) - 0.055) # convert color to sRGB
	#return (img*255).astype(cp.uint8)
	if crop:
		return crop_image(img)

	return img

def compute_rmse(ref, img):
	#return np.sqrt(np.mean(np.power(ref - img, 2)))
	return cp.sqrt(cp.mean(cp.power(ref - img, 2)))

def compute_mape(ref, img):
	#return np.mean(np.abs((ref - img) / (ref + 0.01)))*100
	return cp.mean(cp.abs((ref - img) / (ref + 0.01)))*100

sceneName = "Forest"
cropImage = False
modelDir = r"C:\Users\daqil\Downloads" + "\\" + sceneName + "_Environment_Results"#"_IndirectOnly_Results"
refName = sceneName + "_Reference=100000Frames.AccumulatePass.output.exr"
refImg = get_exr_rgb(modelDir + "\\" + refName, cropImage)
numReSTIRInstances = 1
numExperiments = 100
numMutations = 5
shiftStrategies = ["BSDF"]#["HybridRc"]
SSName = ["full path mutation", "path mutation upto reconnection", "reconnection vertex mutation"]
computeCovarianceInsideBox = False

makeVariancePlotsSPP = False
makeCorrelationPlotsSPP = False
makeVariancePlotsMutations = False
makeAcceptanceRatioPlots = False
makeCorrelationPlots = True

plt.imshow(refImg.get())
plt.show()

plt.style.use("seaborn-darkgrid")
plt.rc('font', size=16)

if makeVariancePlotsSPP:
	for shiftStrategy in shiftStrategies:
		modelShiftDir = modelDir + "\\" + shiftStrategy
		dirData = []
		RMSEData = []
		MAPEData = []
		meanTimesData = []
		sppData = []

		for dir_ in os.listdir(modelShiftDir):
			if ".png" in dir_:
				continue
			print(f"Processing directory {dir_}.")

			RMSE = [0.0]*numReSTIRInstances
			MAPE = [0.0]*numReSTIRInstances
			meanTimes = [0.0]*numReSTIRInstances
			spp = [0]*numReSTIRInstances

			for id, ReSTIRN in enumerate(range(numReSTIRInstances)):
				print(id, ReSTIRN)
				folderName = modelShiftDir + "\\" + dir_ + r"\ReSTIRN=" + str(ReSTIRN + 1)

				for fileName in os.listdir(folderName):
					if ".output.exr" in fileName:
						img = get_exr_rgb(folderName + "\\" + fileName, cropImage)
						RMSE[id] += compute_rmse(refImg, img).get()
						MAPE[id] += compute_mape(refImg, img).get()
					elif ".csv" in fileName:
						with open(folderName + "\\" + fileName, 'r') as file:
							t = 0
							count = 0
							reader = csv.reader(file)
							for row in reader:
								t += float(row[0])
								count += 1
							t /= count
							meanTimes[id] += t
				RMSE[id] /= numExperiments
				MAPE[id] /= numExperiments
				meanTimes[id] /= numExperiments
				spp[id] = (ReSTIRN + 1)*numExperiments

			dirData.append(dir_)
			RMSEData.append(RMSE)
			MAPEData.append(MAPE)
			meanTimesData.append(meanTimes)
			sppData.append(spp)

		# plot and save RMSE vs mean times
		plt.figure(figsize=(10, 5))
		for id, dir_ in enumerate(dirData):
			#index1 = dir_.find("NM=")
			#index2 = -1 if index1 == -1 else dir_[index1:].find("_")
			#count = 0 if index1 == -1 else int(dir_[index1+3:index1+index2])
			#if count == 1: continue
			#plt.loglog(meanTimesData[id], RMSEData[id], label=str(count) + " mutations")
			print("Mcap: ", count)
			print("RMSE: ", RMSEData[id])
			#plt.loglog(meanTimesData[id], RMSEData[id], label=dir_)
			index1 = dir_.find("THL=")
			index2 = -1 if index1 == -1 else dir_[index1:].find("_")
			count = 0 if index1 == -1 else int(dir_[index1+4:index1+index2])
			plt.loglog(meanTimesData[id], RMSEData[id], label="M_cap=" + str(count))

		handles, labels = plt.gca().get_legend_handles_labels()
		mCap = []
		for label in labels:
			mCap.append(int(label[label.find("=")+1:]))
		order = np.argsort(mCap)
		plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
		#handles, labels = plt.gca().get_legend_handles_labels()
		#numMutations = []
		#for label in labels:
		#	numMutations.append(int(label[0:label.find(" ")]))
		#order = np.argsort(numMutations)
		#plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
		plt.xlabel("mean time")
		plt.ylabel("RMSE")
		#plt.xlim(xmax=0.1)
		plt.savefig(modelShiftDir + r"\RMSEvsTime" + ("_crop" if cropImage else "") + r".png")

		# plot and save RMSE vs spp
		plt.figure(figsize=(10, 5))
		for id, dir_ in enumerate(dirData):
			#index1 = dir_.find("NM=")
			#index2 = -1 if index1 == -1 else dir_[index1:].find("_")
			#count = 0 if index1 == -1 else int(dir_[index1+3:index1+index2])
			#if count == 1: continue
			#plt.loglog(sppData[id], RMSEData[id], label=str(count) + " mutations")
			index1 = dir_.find("THL=")
			index2 = -1 if index1 == -1 else dir_[index1:].find("_")
			count = 0 if index1 == -1 else int(dir_[index1+4:index1+index2])
			plt.loglog(sppData[id], RMSEData[id], label="M_cap=" + str(count))

		handles, labels = plt.gca().get_legend_handles_labels()
		mCap = []
		for label in labels:
			mCap.append(int(label[label.find("=")+1:]))
		order = np.argsort(mCap)
		plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
		#handles, labels = plt.gca().get_legend_handles_labels()
		#numMutations = []
		#for label in labels:
		#	numMutations.append(int(label[0:label.find(" ")]))
		#order = np.argsort(numMutations)
		#plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
		plt.xlabel("spp")
		plt.ylabel("RMSE")
		#plt.ylim(ymax=0.1)
		plt.savefig(modelShiftDir + r"\RMSEvsSPP" + ("_crop" if cropImage else "") + r".png")

		# plot and save MAPE vs mean times
		plt.figure(figsize=(10, 5))
		for id, dir_ in enumerate(dirData):
			print("MAPE: ", MAPEData[id])
			index1 = dir_.find("THL=")
			index2 = -1 if index1 == -1 else dir_[index1:].find("_")
			count = 0 if index1 == -1 else int(dir_[index1+4:index1+index2])
			plt.loglog(meanTimesData[id], MAPEData[id], label="M_cap=" + str(count))

		handles, labels = plt.gca().get_legend_handles_labels()
		mCap = []
		for label in labels:
			mCap.append(int(label[label.find("=")+1:]))
		order = np.argsort(mCap)
		plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
		plt.xlabel("mean time")
		plt.ylabel("MAPE")
		plt.savefig(modelShiftDir + r"\MAPEvsTime" + ("_crop" if cropImage else "") + r".png")

		# plot and save MAPE vs spp
		plt.figure(figsize=(10, 5))
		for id, dir_ in enumerate(dirData):
			plt.loglog(sppData[id], MAPEData[id], label=dir_)

		plt.legend()
		plt.xlabel("spp")
		plt.ylabel("MAPE")
		plt.savefig(modelShiftDir + r"\MAPEvsSPP" + ("_crop" if cropImage else "") + r".png")

if makeCorrelationPlotsSPP:
	for shiftStrategy in shiftStrategies:
		modelShiftDir = modelDir + "\\" + shiftStrategy
		dirData = []
		correlationData = []
		meanTimesData = []
		sppData = []

		for dir_ in os.listdir(modelShiftDir):
			if ".png" in dir_:
				continue
			print(f"Processing directory {dir_}.")

			correlation = [0.0]*numReSTIRInstances
			meanTimes = [0.0]*numReSTIRInstances
			spp = [0]*numReSTIRInstances

			for id, ReSTIRN in enumerate(range(numReSTIRInstances)):
				print(id, ReSTIRN)
				folderName = modelShiftDir + "\\" + dir_ + r"\ReSTIRN=" + str(ReSTIRN + 1)

				# load images
				images = []
				for fileName in os.listdir(folderName):
					if ".output.exr" in fileName:
						print(f"  Loading image: {fileName}")
						img = get_exr_rgb(folderName + "\\" + fileName, cropImage)
						images.append(img)
					elif ".csv" in fileName:
						with open(folderName + "\\" + fileName, 'r') as file:
							t = 0
							count = 0
							reader = csv.reader(file)
							for row in reader:
								t += float(row[0])
								count += 1
							t /= count
							meanTimes[id] += t

				# send to GPU
				print("Sending images to GPU.")

				# compute means and stddevs
				images_cp = cp.stack([image for image in images], axis=0)
				imageMeans_cp = cp.mean(images_cp, axis=0, keepdims=True)
				#imageMeans_cp = cp.stack([refImg], axis=0)
				imageMeansGrayscale_cp = cp.mean(imageMeans_cp, axis=3, keepdims=True)
				imageStddevs_cp = cp.std(images_cp, axis=0, ddof=1, keepdims=True)
				#imageStddevs_cp = cp.mean(cp.square(images_cp - imageMeans_cp), axis=0, keepdims=True)
				invalid_cp = (imageStddevs_cp == 0.0)
				imageStddevs_cp[invalid_cp] = 1.0

				# normalize pixels for correlation
				images_cp = (images_cp - imageMeans_cp) #/ imageStddevs_cp
				#images_cp = (images_cp - imageMeans_cp) / (0.0001 + imageMeansGrayscale_cp)  # EPSILON = 0.0001

				# compute correlations
				spatialRadii = [8]
				imageCorrelation = [0.0]*len(spatialRadii)

				_memoize = {}
				def evalMeanCorrelation(dy, dx):
					# Evaluates corr(I(x,y), I(x+dx,y+dy).
					# Assumes I has been standardized.

					# Cache results with symmetry: meanCorr(dx, dy) = meanCorr(-dx, -dy)
					if (dx, dy) in _memoize:
						return _memoize[(dx, dy)]
					if (-dx, -dy) in _memoize:
						return _memoize[(-dx, -dy)]

					n, h, w, c = images_cp.shape

					baseCrop = images_cp[:, max(0, -dy):h+min(0, -dy), max(0, -dx):w+min(0, -dx), :]
					offsetCrop = images_cp[:, max(0, dy):h+min(0, dy), max(0, dx):w+min(0, dx), :]
					correlations = cp.sum(baseCrop * offsetCrop, axis=0, keepdims=True) / (n - 1.0)

					# skip pixels with zero covariance
					baseInvalid = invalid_cp[:, max(0, -dy):h+min(0, -dy), max(0, -dx):w+min(0, -dx), :]
					offsetInvalid = invalid_cp[:, max(0, dy):h+min(0, dy), max(0, dx):w+min(0, dx), :]
					correlations[cp.logical_or(baseInvalid, offsetInvalid)] = 0.0

					validCount = c * h * w - cp.sum(cp.logical_or(baseInvalid, offsetInvalid))
					result = cp.sum(correlations) / validCount

					_memoize[(dx, dy)] = result
					return result

				# average correlations around the square edge
				print("folderName: ", folderName)

				for id2, radius in enumerate(spatialRadii): # compute correlation in volume
					meanCorrelation = 1.0
					if radius > 0:
						meanCorrelation = 0.0

						if computeCovarianceInsideBox:
							for rr in range(radius):
								r = rr + 1
								c = 0.0
								for dx in range(-r, r + 1):
									c += evalMeanCorrelation(-r, dx)
								for dx in range(-r, r + 1):
									c += evalMeanCorrelation(r, dx)
								for dy in range(-r + 1, r):
									c += evalMeanCorrelation(dy, -r)
								for dy in range(-r + 1, r):
									c += evalMeanCorrelation(dy, r)

								borderPixels = 4 * (2 * r + 1) - 4
								c /= borderPixels
								meanCorrelation += c

							meanCorrelation = cp.asnumpy(meanCorrelation/radius)
						else:
							for dx in range(-radius, radius + 1):
								meanCorrelation += evalMeanCorrelation(-radius, dx)
							for dx in range(-radius, radius + 1):
								meanCorrelation += evalMeanCorrelation(radius, dx)
							for dy in range(-radius + 1, radius):
								meanCorrelation += evalMeanCorrelation(dy, -radius)
							for dy in range(-radius + 1, radius):
								meanCorrelation += evalMeanCorrelation(dy, radius)

							borderPixels = 4 * (2 * radius + 1) - 4
							meanCorrelation /= borderPixels
							meanCorrelation = cp.asnumpy(meanCorrelation)

					imageCorrelation[id2] = meanCorrelation
					print(f"correlation (r={radius}): {imageCorrelation[id2]}")

				correlation[id] = imageCorrelation[0]
				meanTimes[id] /= numExperiments
				spp[id] = ReSTIRN + 1

			dirData.append(dir_)
			correlationData.append(correlation)
			meanTimesData.append(meanTimes)
			sppData.append(spp)

		# plot and save correlation vs mean times
		plt.figure(figsize=(7, 5))
		for id, dir_ in enumerate(dirData):
			#index1 = dir_.find("SS=")
			index1 = dir_.find("NM=")
			index2 = -1 if index1 == -1 else dir_[index1:].find("_")
			count = 0 if index1 == -1 else int(dir_[index1+3:index1+index2])
			print(meanTimesData[id])
			print(correlationData[id])
			#plt.plot(meanTimesData[id], correlationData[id], label=str(count) + " mutations")
			#plt.plot(meanTimesData[id], correlationData[id], label=dir_[index1+3:])
			#plt.plot(meanTimesData[id], correlationData[id], label=dir_ + "_SR=" + str(spatialRadii[0]))

		handles, labels = plt.gca().get_legend_handles_labels()
		numMutations = []
		for label in labels:
			numMutations.append(int(label[0:label.find(" ")]))
		order = np.argsort(numMutations)
		plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
		#plt.legend()
		plt.title("Experiments per data point: " + str(numExperiments))
		plt.xlabel("Time (seconds)")
		plt.ylabel("Radial correlation (" + str(spatialRadii[0]) + " pixel radius)")
		plt.savefig(modelShiftDir + r"\Correlation" + str(spatialRadii[0]) + r"vsTime" + ("_crop" if cropImage else "") + r".png")

		# plot and save correlation vs spp
		plt.figure(figsize=(7, 5))
		for id, dir_ in enumerate(dirData):
			#index1 = dir_.find("SS=")
			index1 = dir_.find("NM=")
			index2 = -1 if index1 == -1 else dir_[index1:].find("_")
			count = 0 if index1 == -1 else int(dir_[index1+3:index1+index2])
			#plt.plot(sppData[id], correlationData[id], label=str(count) + " mutations")
			#plt.plot(sppData[id], correlationData[id], label=dir_[index1+3:])
			#plt.plot(sppData[id], correlationData[id], label=dir_ + "_SR=" + str(spatialRadii[0]))

		handles, labels = plt.gca().get_legend_handles_labels()
		numMutations = []
		for label in labels:
			numMutations.append(int(label[0:label.find(" ")]))
		order = np.argsort(numMutations)
		plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
		#plt.legend()
		plt.title("Experiments per data point: " + str(numExperiments))
		plt.xlabel("spp")
		plt.ylabel("Radial correlation (" + str(spatialRadii[0]) + " pixel radius)")
		plt.savefig(modelShiftDir + r"\Correlation" + str(spatialRadii[0]) + r"vsSPP" + ("_crop" if cropImage else "") + r".png")

if makeVariancePlotsMutations:
	for shiftStrategy in shiftStrategies:
		modelShiftDir = modelDir + "\\" + shiftStrategy
		RMSE = [0.0]*numMutations
		mutationCount = [0.0]*numMutations
		nCount = 0

		for dir_ in os.listdir(modelShiftDir):
			if ".png" in dir_:
				continue
			print(f"Processing: {dir_}")

			# load images
			images = []
			folderName = modelShiftDir + "\\" + dir_ + r"\ReSTIRN=1"
			index1 = dir_.find("NM=")
			index2 = -1 if index1 == -1 else dir_[index1:].find("_")
			count = 0 if index1 == -1 else float(dir_[index1+3:index1+index2])

			mutationCount[nCount] = count
			for fileName in os.listdir(folderName):
				if ".output.exr" in fileName:
					img = get_exr_rgb(folderName + "\\" + fileName, cropImage)
					RMSE[nCount] += compute_rmse(refImg, img).get()

			RMSE[nCount] /= numExperiments
			nCount = nCount + 1

		RMSESorted = [x for _, x in sorted(zip(mutationCount, RMSE))]
		mutationCountSorted = sorted(mutationCount)

		plt.figure(figsize=(7, 5))
		plt.plot(mutationCountSorted, RMSESorted)
		plt.legend()
		plt.xlabel("Mutations")
		plt.ylabel("RMSE")
		plt.locator_params(axis='x', nbins=6)
		plt.locator_params(axis='y', nbins=6)
		plt.ylim(ymin=0.8*np.min(RMSESorted), ymax=1.2*np.max(RMSESorted))
		plt.savefig(modelShiftDir + r"\RMSEvsMutations" + ("_crop" if cropImage else "") + r".png")

if makeAcceptanceRatioPlots:
	for shiftStrategy in shiftStrategies:
		modelShiftDir = modelDir + "\\" + shiftStrategy
		acceptanceRatio = [0.0]*numMutations
		mutationCount = [0.0]*numMutations
		nCount = 0

		for dir_ in os.listdir(modelShiftDir):
			if ".png" in dir_:
				continue
			print(f"Processing: {dir_}")

			# load images
			images = []
			folderName = modelShiftDir + "\\" + dir_ + r"\ReSTIRN=1"
			index1 = dir_.find("NM=")
			index2 = -1 if index1 == -1 else dir_[index1:].find("_")
			count = 0 if index1 == -1 else float(dir_[index1+3:index1+index2])

			mutationCount[nCount] = count
			for fileName in os.listdir(folderName):
				if ".debug.exr" in fileName:
					img = get_exr_rgb(folderName + "\\" + fileName, cropImage)
					acceptanceRatio[nCount] += cp.mean(img).get()*0.75

			acceptanceRatio[nCount] /= numExperiments
			nCount = nCount + 1

		acceptanceRatioSorted = [x for _, x in sorted(zip(mutationCount, acceptanceRatio))]
		mutationCountSorted = sorted(mutationCount)
		print(acceptanceRatioSorted)

		plt.figure(figsize=(10, 5))
		plt.plot(mutationCountSorted, acceptanceRatioSorted)
		plt.legend()
		plt.xlabel("Mutations")
		plt.ylabel("Acceptance Rate")
		plt.locator_params(axis='x', nbins=6)
		plt.locator_params(axis='y', nbins=6)
		plt.ylim(ymin=0.0, ymax=1.0)
		plt.savefig(modelShiftDir + r"\AcceptancevsMutations" + ("_crop" if cropImage else "") + r".png")

if makeCorrelationPlots:
	for shiftStrategy in shiftStrategies:
		modelShiftDir = modelDir + "\\" + shiftStrategy
		dirData = []
		correlationData = []
		correlationDataBaseline = []
		spatialRadiusData = []

		for dir_ in os.listdir(modelShiftDir):
			if ".png" in dir_:
				continue
			print(f"Processing: {dir_}")

			# load images
			images = []
			folderName = modelShiftDir + "\\" + dir_ + r"\ReSTIRN=1"
			for fileName in os.listdir(folderName):
				if ".output.exr" in fileName:
					print(f"  Loading image: {fileName}")
					img = get_exr_rgb(folderName + "\\" + fileName, cropImage).astype(np.float32)
					images.append(img)

			# send to GPU
			print("Sending images to GPU.")

			# compute means and stddevs
			images_cp = cp.stack([image for image in images], axis=0)
			imageMeans_cp = cp.mean(images_cp, axis=0, keepdims=True)
			imageMeansGrayscale_cp = cp.mean(imageMeans_cp, axis=3, keepdims=True)
			imageStddevs_cp = cp.std(images_cp, axis=0, ddof=1, keepdims=True)
			invalid_cp = (imageStddevs_cp == 0.0)
			imageStddevs_cp[invalid_cp] = 1.0

			# normalize pixels for correlation
			images_cp = (images_cp - imageMeans_cp) / imageStddevs_cp
			#images_cp = (images_cp - imageMeans_cp) / (0.0001 + imageMeansGrayscale_cp)  # EPSILON = 0.0001

			# compute correlations
			spatialRadii = [1, 2, 3, 4, 6, 8, 10, 13, 16, 20, 25, 30, 35, 45, 60]
			imageCorrelation = [0.0]*len(spatialRadii)

			_memoize = {}
			def evalMeanCorrelation(dy, dx):
				# Evaluates corr(I(x,y), I(x+dx,y+dy).
				# Assumes I has been standardized.

				# Cache results with symmetry: meanCorr(dx, dy) = meanCorr(-dx, -dy)
				if (dx, dy) in _memoize:
					return _memoize[(dx, dy)]
				if (-dx, -dy) in _memoize:
					return _memoize[(-dx, -dy)]

				n, h, w, c = images_cp.shape

				baseCrop = images_cp[:, max(0, -dy):h+min(0, -dy), max(0, -dx):w+min(0, -dx), :]
				offsetCrop = images_cp[:, max(0, dy):h+min(0, dy), max(0, dx):w+min(0, dx), :]
				correlations = cp.sum(baseCrop * offsetCrop, axis=0, keepdims=True) / (n - 1.0)

				# skip pixels with zero covariance
				baseInvalid = invalid_cp[:, max(0, -dy):h+min(0, -dy), max(0, -dx):w+min(0, -dx), :]
				offsetInvalid = invalid_cp[:, max(0, dy):h+min(0, dy), max(0, dx):w+min(0, dx), :]
				correlations[cp.logical_or(baseInvalid, offsetInvalid)] = 0.0

				validCount = c * h * w - cp.sum(cp.logical_or(baseInvalid, offsetInvalid))
				result = cp.sum(correlations) / validCount

				_memoize[(dx, dy)] = result

				return result

			# average correlations around the square edge
			print("folderName: ", folderName)

			for id, radius in enumerate(spatialRadii):
				meanCorrelation = 1.0
				if radius > 0:
					meanCorrelation = 0.0
					for dx in range(-radius, radius + 1):
						meanCorrelation += evalMeanCorrelation(-radius, dx)
					for dx in range(-radius, radius + 1):
						meanCorrelation += evalMeanCorrelation(radius, dx)
					for dy in range(-radius + 1, radius):
						meanCorrelation += evalMeanCorrelation(dy, -radius)
					for dy in range(-radius + 1, radius):
						meanCorrelation += evalMeanCorrelation(dy, radius)

					borderPixels = 4 * (2 * radius + 1) - 4
					meanCorrelation /= borderPixels
					meanCorrelation = cp.asnumpy(meanCorrelation)

				imageCorrelation[id] = meanCorrelation
				print(f"correlation (r={radius}): {imageCorrelation[id]}")

			dirData.append(dir_)
			correlationData.append(imageCorrelation)
			spatialRadiusData.append(spatialRadii)

			# plot and save correlation vs spatial radius
			plt.figure(figsize=(7, 5))
			for id, dir_ in enumerate(dirData):
				index1 = dir_.find("NM=")
				index2 = -1 if index1 == -1 else dir_[index1:].find("_")
				count = 0 if index1 == -1 else int(dir_[index1+3:index1+index2])
				p = plt.plot(spatialRadiusData[id], correlationData[id], label=str(count) + " mutations")
				#p = plt.plot(spatialRadiusData[id], np.divide(correlationData[id], correlationDataBaseline[0]), label=dir_)

			handles, labels = plt.gca().get_legend_handles_labels()
			numMutations = []
			for label in labels:
				numMutations.append(int(label[0:label.find(" ")]))
			order = np.argsort(numMutations)
			plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
			plt.xlabel("Spatial radius")
			plt.ylabel("Radial correlation")
			plt.savefig(modelShiftDir + r"\CorrelationvsRadius" + ("_crop" if cropImage else "") + r".png")
