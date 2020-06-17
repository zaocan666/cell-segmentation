import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage import morphology,feature
import matplotlib.pyplot as plt

def instancing_open(pred_bi):
	#pred bi: ndarry [h, w]
	# pred_bi = cv2.fromarray(pred_bi)

	# binary_mask = cv2.medianBlur(pred_bi, 5)
	g=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
	binary_mask=cv2.morphologyEx(pred_bi,cv2.MORPH_OPEN,g)
	# binary_mask=pred_bi

	connectivity = 4
	_, label_img, _, _ = cv2.connectedComponentsWithStats(binary_mask , connectivity , cv2.CV_32S)

	return label_img

def instancing_water(pred_bi):
	pred_bi = pred_bi>0
	distance = ndi.distance_transform_edt(pred_bi)  # ????
	local_maxi = feature.peak_local_max(distance, indices=False, footprint=np.ones((3, 3)),
										labels=pred_bi)  # ????
	markers = ndi.label(local_maxi)[0]  # ?????
	kernel = morphology.disk(8)
	#combine near markers
	markers_bi=morphology.dilation(markers>0, kernel)
	connectivity = 4
	_, markers_lab, _, _=cv2.connectedComponentsWithStats(np.array(markers_bi>0,np.uint8), connectivity , cv2.CV_32S)
	labels = morphology.watershed(-distance, markers_lab, mask=pred_bi)  # ????????????

	# fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
	# axes = axes.ravel()
	# ax0, ax1, ax2, ax3 = axes

	# ax0.imshow(pred_bi, cmap=plt.cm.gray, interpolation='nearest')
	# ax0.set_title("Original")
	# ax1.imshow(-distance, cmap=plt.cm.jet, interpolation='nearest')
	# ax1.set_title("Distance")
	# ax2.imshow(markers_lab, cmap=plt.cm.Spectral, interpolation='nearest')
	# ax2.set_title("Markers")
	# ax3.imshow(labels, cmap=plt.cm.Spectral, interpolation='nearest')
	# ax3.set_title("Segmented")

	# for ax in axes:
	#     ax.axis('off')

	# fig.tight_layout()
	# plt.show()

	return labels

def instancing_water2(pred_bi, dis_thres, max_dil_size):

	pred_bi = pred_bi>0

	distance = ndi.distance_transform_edt(pred_bi)  # ????
	local_maxi = feature.peak_local_max(distance, indices=False, footprint=np.ones((3, 3)),
										labels=pred_bi)  # ????

	distance_threshold = distance>dis_thres
	local_maxi=local_maxi*distance_threshold

	kernel = morphology.disk(max_dil_size)
	localmax_bi=morphology.dilation(local_maxi>0, kernel)
	# markers_lab = ndi.label(localmax_bi)[0]  # ?????
	connectivity = 4
	_, markers_lab, _, _=cv2.connectedComponentsWithStats(np.array(localmax_bi>0,np.uint8), connectivity , cv2.CV_32S)
	#combine near markers
	labels_water = morphology.watershed(-distance, markers_lab, mask=pred_bi)  # ????????????

	pred_labelW = np.array(pred_bi, np.int)-np.array(labels_water>0, np.int)
	_, labels_left, _, _ = cv2.connectedComponentsWithStats(np.array(pred_labelW>0,np.uint8), connectivity , cv2.CV_32S)
	label_base = (labels_left>0)*labels_water.max()
	labels_left += label_base

	labels = labels_left+labels_water

	# fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
	# axes = axes.ravel()
	# ax0, ax1, ax2, ax3 = axes

	# ax0.imshow(pred_bi, cmap=plt.cm.gray, interpolation='nearest')
	# ax0.set_title("Original")
	# ax1.imshow(-distance, cmap=plt.cm.jet, interpolation='nearest')
	# ax1.set_title("Distance")
	# ax2.imshow(markers_lab, cmap=plt.cm.Spectral, interpolation='nearest')
	# ax2.set_title("Markers")
	# ax3.imshow(labels, cmap=plt.cm.Spectral, interpolation='nearest')
	# ax3.set_title("Segmented")

	# for ax in axes:
	#     ax.axis('off')

	# fig.tight_layout()
	# plt.show()

	return labels

def instancing_water3(pred_bi):
	pred_bi_in = pred_bi>0
	connectivity=4
	kernel_pred=morphology.disk(9)
	pred_bi=morphology.opening(pred_bi_in, kernel_pred)

	pred_bi_dia=morphology.dilation(pred_bi>0, kernel_pred)
	pred_minus=(np.array(pred_bi_in,np.int)-np.array(pred_bi,np.int))>0
	pred_bi_elimate=pred_minus*(1-(pred_bi_dia>0))

	pred_bi=(pred_bi+pred_bi_elimate)>0

	# fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
	# axes = axes.ravel()
	# ax0, ax1, ax2, ax3 = axes

	# ax0.imshow(pred_bi_in, cmap=plt.cm.gray, interpolation='nearest')
	# ax1.imshow(pred_bi, cmap=plt.cm.gray, interpolation='nearest')
	# ax2.imshow(pred_minus, cmap=plt.cm.gray, interpolation='nearest')
	# ax3.imshow(pred_bi_elimate, cmap=plt.cm.gray, interpolation='nearest')
	# plt.show()

	pred_bi=morphology.opening(pred_bi, morphology.disk(3)) #elimate small noise

	distance = ndi.distance_transform_edt(pred_bi)  # ????
	local_maxi = feature.peak_local_max(distance, indices=False, footprint=np.ones((3, 3)),
										labels=pred_bi)  # ????

	distance_threshold = distance>15
	local_maxi=local_maxi*distance_threshold

	kernel = morphology.disk(8)
	localmax_bi=morphology.dilation(local_maxi>0, kernel)
	# markers_lab = ndi.label(localmax_bi)[0]  # ?????
	connectivity = 4
	_, markers_lab, _, _=cv2.connectedComponentsWithStats(np.array(localmax_bi>0,np.uint8), connectivity , cv2.CV_32S)
	#combine near markers
	labels_water = morphology.watershed(-distance, markers_lab, mask=pred_bi)  # ????????????

	pred_labelW = np.array(pred_bi, np.int)-np.array(labels_water>0, np.int)
	_, labels_left, _, _ = cv2.connectedComponentsWithStats(np.array(pred_labelW>0,np.uint8), connectivity , cv2.CV_32S)
	label_base = (labels_left>0)*labels_water.max()
	labels_left += label_base

	labels = labels_left+labels_water

	# fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
	# axes = axes.ravel()
	# ax0, ax1, ax2, ax3 = axes

	# ax0.imshow(pred_bi, cmap=plt.cm.gray, interpolation='nearest')
	# ax0.set_title("Original")
	# ax1.imshow(-distance, cmap=plt.cm.jet, interpolation='nearest')
	# ax1.set_title("Distance")
	# ax2.imshow(markers_lab, cmap=plt.cm.Spectral, interpolation='nearest')
	# ax2.set_title("Markers")
	# ax3.imshow(labels, cmap=plt.cm.Spectral, interpolation='nearest')
	# ax3.set_title("Segmented")

	# for ax in axes:
	#     ax.axis('off')

	# fig.tight_layout()
	# plt.show()

	return labels

def instancing(pred_bi, dis_thres=13, max_dil_size=8):
	pred_bi = np.array(pred_bi, np.uint8)
	# return instancing_water3(pred_bi)
	return instancing_water2(pred_bi, dis_thres, max_dil_size)
	# return instancing_open(pred_bi)