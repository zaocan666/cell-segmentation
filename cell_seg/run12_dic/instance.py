import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage import morphology,feature
import matplotlib.pyplot as plt


def instancing_water(pred_bi, pred_contour, img, dis_thres, max_dil_size):
	pred_bi = pred_bi>0
	pred_contour_input = pred_contour>0
	# pred_contour = morphology.dilation(pred_contour_input, morphology.disk(8))
	pred_contour = pred_contour_input

	pred_center = np.array(pred_bi, np.int)-np.array(pred_contour, np.int)
	pred_center = pred_center>0
	# pred_center_ero = morphology.opening(pred_center, morphology.disk(19))
	pred_center_ero = morphology.opening(pred_center, morphology.disk(14))
	_, markers_lab, _, _=cv2.connectedComponentsWithStats(np.array(pred_center_ero,np.uint8), 4 , cv2.CV_32S)

	distance = ndi.distance_transform_edt(markers_lab>0)  # ????
	# distance = ndi.distance_transform_edt(pred_center_ero) 
	distance[pred_contour>0]=1

	water_mask=pred_bi
	# water_mask = (pred_contour_input + markers_lab)>0
	# water_mask = morphology.closing(water_mask, morphology.disk(6))

	labels_water = morphology.watershed(-distance, markers_lab, mask=water_mask)  # ????????????
	# labels_water = morphology.watershed(-distance, markers_lab, mask=distance>0)  # ????????????

	# pred_labelW = np.array(pred_bi, np.int)-np.array(labels_water>0, np.int)
	# pred_labelW = morphology.opening(pred_labelW, morphology.disk(11))
	# _, labels_left, _, _ = cv2.connectedComponentsWithStats(np.array(pred_labelW>0,np.uint8), 4 , cv2.CV_32S)
	# label_base = (labels_left>0)*labels_water.max()
	# labels_left += label_base

	# labels = labels_left+labels_water
	labels = labels_water

	# fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(8, 8))
	# axes = axes.ravel()
	# ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7 = axes

	# ax0.imshow(np.array(img*255, np.int), cmap=plt.cm.gray, interpolation='nearest')
	# ax0.set_title("input")
	# ax1.imshow(pred_bi, cmap=plt.cm.gray, interpolation='nearest')
	# ax1.set_title("Original")
	# ax2.imshow(pred_contour, cmap=plt.cm.gray, interpolation='nearest')
	# ax2.set_title("contour")
	# ax3.imshow(pred_center, cmap=plt.cm.gray, interpolation='nearest')
	# ax3.set_title("center")

	# ax4.imshow(-distance, cmap=plt.cm.jet, interpolation='nearest')
	# ax4.set_title("Distance")
	# ax5.imshow(markers_lab, cmap=plt.cm.Spectral, interpolation='nearest')
	# ax5.set_title("Markers")
	# ax6.imshow(water_mask, cmap=plt.cm.gray, interpolation='nearest')
	# ax6.set_title("water_mask")
	# ax7.imshow(labels, cmap=plt.cm.Spectral, interpolation='nearest')
	# ax7.set_title("Segmented")

	# for ax in axes:
	#     ax.axis('off')

	# fig.tight_layout()
	# plt.show()

	return labels


def instancing_water2(pred_bi, pred_contour, img, dis_thres, max_dil_size):
	pred_bi = pred_bi>0
	pred_contour_input = pred_contour>0
	# pred_contour = morphology.dilation(pred_contour_input, morphology.disk(8))
	pred_contour = pred_contour_input

	pred_center = np.array(pred_bi, np.int)-np.array(pred_contour, np.int)
	pred_center = pred_center>0
	# pred_center_ero = morphology.opening(pred_center, morphology.disk(19))
	pred_center_ero = morphology.opening(pred_center, morphology.disk(14))
	_, markers_lab, _, _=cv2.connectedComponentsWithStats(np.array(pred_center_ero,np.uint8), 4 , cv2.CV_32S)

	distance = ndi.distance_transform_edt(pred_bi>0)  # ????

	water_mask=pred_bi
	# water_mask = (pred_contour_input + markers_lab)>0
	# water_mask = morphology.closing(water_mask, morphology.disk(6))

	labels_water = morphology.watershed(-distance, markers_lab, mask=water_mask)  # ????????????
	# labels_water = morphology.watershed(-distance, markers_lab, mask=distance>0)  # ????????????

	# pred_labelW = np.array(pred_bi, np.int)-np.array(labels_water>0, np.int)
	# pred_labelW = morphology.opening(pred_labelW, morphology.disk(11))
	# _, labels_left, _, _ = cv2.connectedComponentsWithStats(np.array(pred_labelW>0,np.uint8), 4 , cv2.CV_32S)
	# label_base = (labels_left>0)*labels_water.max()
	# labels_left += label_base

	# labels = labels_left+labels_water
	labels = labels_water

	# fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(8, 8))
	# axes = axes.ravel()
	# ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7 = axes

	# ax0.imshow(np.array(img*255, np.int), cmap=plt.cm.gray, interpolation='nearest')
	# ax0.set_title("input")
	# ax1.imshow(pred_bi, cmap=plt.cm.gray, interpolation='nearest')
	# ax1.set_title("Original")
	# ax2.imshow(pred_contour, cmap=plt.cm.gray, interpolation='nearest')
	# ax2.set_title("contour")
	# ax3.imshow(pred_center, cmap=plt.cm.gray, interpolation='nearest')
	# ax3.set_title("center")

	# ax4.imshow(-distance, cmap=plt.cm.jet, interpolation='nearest')
	# ax4.set_title("Distance")
	# ax5.imshow(markers_lab, cmap=plt.cm.Spectral, interpolation='nearest')
	# ax5.set_title("Markers")
	# ax6.imshow(water_mask, cmap=plt.cm.gray, interpolation='nearest')
	# ax6.set_title("water_mask")
	# ax7.imshow(labels, cmap=plt.cm.Spectral, interpolation='nearest')
	# ax7.set_title("Segmented")

	# for ax in axes:
	#     ax.axis('off')

	# fig.tight_layout()
	# plt.show()

	return labels


def instancing(pred_bi, pred_contour, img, dis_thres=13, max_dil_size=8):
	pred_bi = np.array(pred_bi, np.uint8)
	pred_contour = np.array(pred_contour, np.uint8)
	return instancing_water2(pred_bi, pred_contour, img, dis_thres=dis_thres, max_dil_size=max_dil_size)