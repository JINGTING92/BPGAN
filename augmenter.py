import numpy as np
import SimpleITK as sitk
import os
import math
import random
import tensorflow as tf


def threshold_based_crop(image, inside_value=0, outside_value=255):
    """
    Set pixels that are in [min_intensity,otsu_threshold] to inside_value, values above otsu_threshold are
    set to outside_value. The anatomy has higher intensity values than the background, so it is outside.
    :param image:
    :return:
    """

    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
    label_shape_filter.Execute(sitk.OtsuThreshold(image, inside_value, outside_value))
    bounding_box = label_shape_filter.GetBoundingBox(outside_value)

    # The bounding box's first "dim" entries are the starting index and last "dim" entries the size
    return sitk.RegionOfInterest(image, bounding_box[int(len(bounding_box)/2):], bounding_box[0:int(len(bounding_box)/2)])



def augment_images_intensity(image_list, output_prefix, output_suffix):

    '''
    Generate intensity modified images from the originals.
    Args:
        image_list (iterable containing SimpleITK images): The images which we whose intensities we modify.
        output_prefix (string): output file name prefix (file name: output_prefixi_FilterName.output_suffix).
        output_suffix (string): output file name suffix (file name: output_prefixi_FilterName.output_suffix).

    Example in use: intensity_augmened_image = augment_images_intensity(data, os.path.join(OUTPUT_DIR, 'intensity_aug'), 'mha')
    '''

    # Create a list of intensity modifying filters, which we apply to the given images
    filter_list = []

    # Smoothing filters

    filter_list.append(sitk.SmoothingRecursiveGaussianImageFilter())
    filter_list[-1].SetSigma(2.0)

    filter_list.append(sitk.DiscreteGaussianImageFilter())
    filter_list[-1].SetVariance(4.0)

    filter_list.append(sitk.BilateralImageFilter())
    filter_list[-1].SetDomainSigma(4.0)
    filter_list[-1].SetRangeSigma(8.0)

    filter_list.append(sitk.MedianImageFilter())
    filter_list[-1].SetRadius(8)

    # Noise filters using default settings

    # Filter control via SetMean, SetStandardDeviation.
    filter_list.append(sitk.AdditiveGaussianNoiseImageFilter())

    # Filter control via SetProbability
    filter_list.append(sitk.SaltAndPepperNoiseImageFilter())

    # Filter control via SetScale
    filter_list.append(sitk.ShotNoiseImageFilter())

    # Filter control via SetStandardDeviation
    filter_list.append(sitk.SpeckleNoiseImageFilter())

    filter_list.append(sitk.AdaptiveHistogramEqualizationImageFilter())
    filter_list[-1].SetAlpha(1.0)
    filter_list[-1].SetBeta(0.0)

    filter_list.append(sitk.AdaptiveHistogramEqualizationImageFilter())
    filter_list[-1].SetAlpha(0.0)
    filter_list[-1].SetBeta(1.0)

    aug_image_lists = []  # Used only for display purposes in this notebook.
    for i, img in enumerate(image_list):
        aug_image_lists.append([f.Execute(img) for f in filter_list])
        for aug_image, f in zip(aug_image_lists[-1], filter_list):
            sitk.WriteImage(aug_image, output_prefix + str(i) + '_' +
                            f.GetName() + '.' + output_suffix)
    return aug_image_lists


def mult_and_add_intensity_fields(original_image):

    '''
    Modify the intensities using multiplicative and additive Gaussian bias fields.
    '''

    # Gaussian image with same meta-information as original (size, spacing, direction cosine)
    # Sigma is half the image's physical size and mean is the center of the image.
    g_mult = sitk.GaussianSource(original_image.GetPixelIDValue(),
                                 original_image.GetSize(),
                                 [(sz - 1) * spc / 2.0 for sz, spc in
                                  zip(original_image.GetSize(), original_image.GetSpacing())],
                                 original_image.TransformContinuousIndexToPhysicalPoint(
                                     np.array(original_image.GetSize()) / 2.0),
                                 25,
                                 original_image.GetOrigin(),
                                 original_image.GetSpacing(),
                                 original_image.GetDirection())

    # Gaussian image with same meta-information as original (size, spacing, direction cosine)
    # Sigma is 1/8 the image's physical size and mean is at 1/16 of the size
    g_add = sitk.GaussianSource(original_image.GetPixelIDValue(),
                                original_image.GetSize(),
                                [(sz - 1) * spc / 8.0 for sz, spc in
                                 zip(original_image.GetSize(), original_image.GetSpacing())],
                                original_image.TransformContinuousIndexToPhysicalPoint(
                                    np.array(original_image.GetSize()) / 16.0),
                                25,
                                original_image.GetOrigin(),
                                original_image.GetSpacing(),
                                original_image.GetDirection())

    return g_mult * original_image + g_add


class Normalization(object):

    """
    Normalize an image to [0, 255]
    """

    def __init__(self):
        self.name = 'Normalization'

    def __call__(self, sample):

        rescalerFilter = sitk.RescaleIntensityImageFilter()
        rescalerFilter.SetOutputMinimum(0)
        rescalerFilter.SetOutputMaximum(255)
        image, label = sample['image'], sample['label']
        image = rescalerFilter.Execute(image)

        return {'image': image, 'label': label}


class StatisticalNormalization(object):

    """
    Normalize an image by mapping intensity with intensity distribution
    """

    def __init__(self, sigma):
        self.name = 'StatisticalNormalization'
        assert isinstance(sigma, float)
        self.sigma = sigma

    def __call__(self, sample):

        image, label = sample['image'], sample['label']
        statisticsFilter = sitk.StatisticsImageFilter()
        statisticsFilter.Execute(image)

        intensityWindowingFilter = sitk.IntensityWindowingImageFilter()
        intensityWindowingFilter.SetOutputMaximum(255)
        intensityWindowingFilter.SetOutputMinimum(0)
        intensityWindowingFilter.SetWindowMaximum(statisticsFilter.GetMean() + self.sigma * statisticsFilter.GetSigma())
        intensityWindowingFilter.SetWindowMinimum(statisticsFilter.GetMean() - self.sigma * statisticsFilter.GetSigma())

        image = intensityWindowingFilter.Execute(image)

        return {'image': image, 'label': label}


class Reorient(object):
    """
    (Beta) Function to orient image in specific axes order
    The elements of the order array must be an permutation of the numbers from 0 to 2.
    """

    def __init__(self, order):
        self.name = 'Reoreient'
        assert isinstance(order, (int, tuple))
        assert len(order) == 3
        self.order = order

    def __call__(self, sample):
        reorientFilter = sitk.PermuteAxesImageFilter()
        reorientFilter.SetOrder(self.order)
        image = reorientFilter.Execute(sample['image'])
        label = reorientFilter.Execute(sample['label'])

        return {'image': image, 'label': label}


class Invert(object):
    """
    Invert the image intensity from 0-255
    """

    def __init__(self):
        self.name = 'Invert'

    def __call__(self, sample):
        invertFilter = sitk.InvertIntensityImageFilter()
        image = invertFilter.Execute(sample['image'], 255)
        label = sample['label']

        return {'image': image, 'label': label}


class Resample(object):
    """
    Resample the volume in a sample to a given voxel size
      Args:
          voxel_size (float or tuple): Desired output size.
          If float, output volume is isotropic.
          If tuple, output voxel size is matched with voxel size
          Currently only support linear interpolation method
    """

    def __init__(self, voxel_size):
        self.name = 'Resample'

        assert isinstance(voxel_size, (float, tuple))
        if isinstance(voxel_size, float):
            self.voxel_size = (voxel_size, voxel_size, voxel_size)
        else:
            assert len(voxel_size) == 3
            self.voxel_size = voxel_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        old_spacing = image.GetSpacing()
        old_size = image.GetSize()

        new_spacing = self.voxel_size

        new_size = []
        for i in range(3):
            new_size.append(int(math.ceil(old_spacing[i] * old_size[i] / new_spacing[i])))
        new_size = tuple(new_size)

        resampler = sitk.ResampleImageFilter()
        resampler.SetInterpolator(2)
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetSize(new_size)

        # resample on image
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetOutputDirection(image.GetDirection())
        # print("Resampling image...")
        image = resampler.Execute(image)

        # resample on segmentation
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetOutputOrigin(label.GetOrigin())
        resampler.SetOutputDirection(label.GetDirection())
        # print("Resampling segmentation...")
        label = resampler.Execute(label)

        return {'image': image, 'label': label}


class Padding(object):
    """
    Add padding to the image if size is smaller than patch size
      Args:
          output_size (tuple or int): Desired output size. If int, a cubic volume is formed
      """

    def __init__(self, output_size):
        self.name = 'Padding'

        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size, output_size)
        else:
            assert len(output_size) == 3
            self.output_size = output_size

        assert all(i > 0 for i in list(self.output_size))

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        size_old = image.GetSize()

        if (size_old[0] >= self.output_size[0]) and (size_old[1] >= self.output_size[1]) and (
                size_old[2] >= self.output_size[2]):
            return sample
        else:
            self.output_size = list(self.output_size)
            if size_old[0] > self.output_size[0]:
                self.output_size[0] = size_old[0]
            if size_old[1] > self.output_size[1]:
                self.output_size[1] = size_old[1]
            if size_old[2] > self.output_size[2]:
                self.output_size[2] = size_old[2]

            self.output_size = tuple(self.output_size)

            resampler = sitk.ResampleImageFilter()
            resampler.SetOutputSpacing(image.GetSpacing())
            resampler.SetSize(self.output_size)

            # resample on image
            resampler.SetInterpolator(2)
            resampler.SetOutputOrigin(image.GetOrigin())
            resampler.SetOutputDirection(image.GetDirection())
            image = resampler.Execute(image)

            # resample on label
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            resampler.SetOutputOrigin(label.GetOrigin())
            resampler.SetOutputDirection(label.GetDirection())

            label = resampler.Execute(label)

            return {'image': image, 'label': label}


class RandomCrop(object):
    """
    Crop randomly the image in a sample. This is usually used for data augmentation.
      Drop ratio is implemented for randomly dropout crops with empty label. (Default to be 0.2)
      This transformation only applicable in train mode
    Args:
      output_size (tuple or int): Desired output size. If int, cubic crop is made.
    """

    def __init__(self, output_size, drop_ratio=0.1, min_pixel=1):

        self.name = 'Random Crop'

        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size, output_size)
        else:
            assert len(output_size) == 3
            self.output_size = output_size

        assert isinstance(drop_ratio, float)
        if drop_ratio >= 0 and drop_ratio <= 1:
            self.drop_ratio = drop_ratio
        else:
            raise RuntimeError('Drop ratio should be between 0 and 1')

        assert isinstance(min_pixel, int)
        if min_pixel >= 0:
            self.min_pixel = min_pixel
        else:
            raise RuntimeError('Min label pixel count should be integer larger than 0')

    def __call__(self, sample):

        image, label = sample['image'], sample['label']
        size_old = image.GetSize()
        size_new = self.output_size

        contain_label = False

        roiFilter = sitk.RegionOfInterestImageFilter()
        roiFilter.SetSize([size_new[0], size_new[1], size_new[2]])

        # statFilter = sitk.StatisticsImageFilter()
        # statFilter.Execute(label)
        # print(statFilter.GetMaximum(), statFilter.GetSum())

        while not contain_label:
            # get the start crop coordinate in ijk
            if size_old[0] <= size_new[0]:
                start_i = 0
            else:
                start_i = np.random.randint(0, size_old[0] - size_new[0])

            if size_old[1] <= size_new[1]:
                start_j = 0
            else:
                start_j = np.random.randint(0, size_old[1] - size_new[1])

            if size_old[2] <= size_new[2]:
                start_k = 0
            else:
                start_k = np.random.randint(0, size_old[2] - size_new[2])

            roiFilter.SetIndex([start_i, start_j, start_k])

            label_crop = roiFilter.Execute(label)
            statFilter = sitk.StatisticsImageFilter()
            statFilter.Execute(label_crop)

            # will iterate until a sub volume containing label is extracted
            # pixel_count = seg_crop.GetHeight()*seg_crop.GetWidth()*seg_crop.GetDepth()
            # if statFilter.GetSum()/pixel_count<self.min_ratio:
            if statFilter.GetSum() < self.min_pixel:
                contain_label = self.drop(self.drop_ratio)  # has some probabilty to contain patch with empty label
            else:
                contain_label = True

        image_crop = roiFilter.Execute(image)

        return {'image': image_crop, 'label': label_crop}

    def drop(self, probability):

        return random.random() <= probability


class RandomNoise(object):
    """
    Randomly noise to the image in a sample. This is usually used for data augmentation.
    """

    def __init__(self):
        self.name = 'Random Noise'

    def __call__(self, sample):
        self.noiseFilter = sitk.AdditiveGaussianNoiseImageFilter()
        self.noiseFilter.SetMean(0)
        self.noiseFilter.SetStandardDeviation(0.1)

        # print("Normalizing image...")
        image, label = sample['image'], sample['label']
        image = self.noiseFilter.Execute(image)

        return {'image': image, 'label': label}
