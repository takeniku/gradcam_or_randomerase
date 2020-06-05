import torchvision
import random
import math
from my_module import dataset


class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al.
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    '''

    def __init__(self, probability=0.5, mean=[1, 1, 1], gd_th=0, sl=0.02, sh=0.4, r1=0.3):
        self.probability = probability
        self.mean = mean
        self.gd_th = gd_th
        self.sl = sl
        self.sh = sh
        self.r1 = r1

# #################################################mask####################################
    def mask_generator(self, img, grad):

        for bc in range(img.size()[0]):
            # if random.uniform(0, 1) > self.probability:
            #     continue
            grad[bc, 1, :, :] = grad[bc, 0, :, :] <= self.gd_th
            grad[bc, 2, :, :] = grad[bc, 0, :, :] > self.gd_th
            # white_noise
            img[bc, 0, :, :] = grad[bc, 1, :, :] * \
                img[bc, 0, :, :] + grad[bc, 2, :, :] * 1
            img[bc, 1, :, :] = grad[bc, 1, :, :] * \
                img[bc, 1, :, :] + grad[bc, 2, :, :] * 1
            img[bc, 2, :, :] = grad[bc, 1, :, :] * \
                img[bc, 2, :, :] + grad[bc, 2, :, :] * 1
            # green_noise
            # img[bc, 0, :, :] = grad[bc, 1, :, :]*img[bc, 0, :, :] + grad[bc, 2, :, :]*0
            # img[bc, 1, :, :] = grad[bc, 1, :, :]*img[bc, 1, :, :] + grad[bc, 2, :, :]*1
            # img[bc, 2, :, :] = grad[bc, 1, :, :]*img[bc, 2, :, :] + grad[bc, 2, :, :]*0
        return img

# randomerase#########################################3
    def randomerase_generator(self, img):
        for bc in range(img.size()[0]):
            if random.uniform(0, 1) > self.probability:
                return img

            for attempt in range(100):
                area = img.size()[2] * img.size()[3]
                target_area = random.uniform(self.sl, self.sh) * area
                aspect_ratio = random.uniform(self.r1, 1 / self.r1)

                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))
                # print('w', img.shape)
                # print('img.size()[2]', img.size()[1])
                if w < img.size()[2] and h < img.size()[3]:
                    x1 = random.randint(0, img.size()[3] - h)
                    y1 = random.randint(0, img.size()[2] - w)
                    if img.size()[1] == 3:
                        img[bc, 0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                        img[bc, 1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                        img[bc, 2, x1:x1 + h, y1:y1 + w] = self.mean[2]

                    else:
                        img[bc, 0, x1:x1 + h, y1:y1 + w] = self.mean[0]

                    return img

        return img

    def show_ex(self, img, grad):
        img = self.mask_generator(img, grad)
        img = img.cpu()
        # img_np = img.numpy()
        img = next(iter(img))
        img = torchvision.utils.make_grid(img)
        dataset.imshow(img)
