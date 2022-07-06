import math
import numpy as np
import imageio

from glob import glob
import os
from PIL import Image
import PIL.ImageOps
import cairocffi as cairo
from PIL import Image
from util_files.data.transforms.degradation_models import DegradationGenerator, all_degradations
from util_files.color_utils import rgb_to_gray, img_8bit_to_float, gray_float_to_8bit


class Synthetic:
    def __init__(self, MAX_X=840, MAX_Y=840, border=50):
        self.MAX_X = MAX_X
        self.MAX_Y = MAX_Y
        self.border = border

    def triangle(self, ctx):
        x = []
        y = []
        for it in range(3):
            x.append(np.random.randint(self.border, self.MAX_X - self.border))
            y.append(np.random.randint(self.border, self.MAX_Y - self.border))
        ctx.move_to(x[0], y[0])
        ctx.line_to(x[1], y[1])
        ctx.line_to(x[2], y[2])
        ctx.line_to(x[0], y[0])
        ctx.stroke()

        ctx.close_path()

    def bowtie(self, ctx):
        x, y = [], []
        for it in range(4):
            x.append(np.random.randint(self.border, self.MAX_X - self.border))
            y.append(np.random.randint(self.border, self.MAX_Y - self.border))
        ctx.move_to(x[3], y[3])
        ctx.line_to(x[0], y[0])
        ctx.line_to(x[1], y[1])
        ctx.line_to(x[2], y[2])
        ctx.set_line_join(cairo.LINE_JOIN_MITER)

        ctx.stroke()

        ctx.close_path()

    def line(self, ctx):
        x = []
        y = []
        for it in range(2):
            x.append(np.random.randint(self.border, self.MAX_X - self.border))
            y.append(np.random.randint(self.border, self.MAX_Y - self.border))
        ctx.move_to(x[0], y[0])
        ctx.line_to(x[1], y[1])
        ctx.stroke()
        ctx.close_path()

    def rectangle(self, ctx):
        x = []
        y = []
        for it in range(2):
            if (it == 1):
                x.append(np.random.randint(0, self.MAX_X - x[0] - self.border))
                y.append(np.random.randint(0, self.MAX_Y - y[0] - self.border))
            else:
                x.append(np.random.randint(self.border, self.MAX_X - self.border))
                y.append(np.random.randint(self.border, self.MAX_Y - self.border))
        ctx.move_to(0, 0)
        ctx.rectangle(x[0], y[0], x[1], y[1])
        ctx.set_line_join(cairo.LINE_JOIN_MITER)
        ctx.stroke()
        ctx.close_path()

    def circle(self, ctx):
        ctx.save()
        x = []
        y = []
        for it in range(3):
            if (it == 1):
                x.append(np.random.randint(0, self.MAX_X - x[0] - self.border))
                y.append(np.random.randint(0, self.MAX_Y - y[0] - self.border))
            else:
                x.append(np.random.randint(self.border, self.MAX_X - self.border))
                y.append(np.random.randint(self.border, self.MAX_Y - self.border))

        #         ctx.scale(np.random.random(), np.random.random())
        ctx.arc(x[0], y[0], min(x[1], x[0], y[0]), 0, 2 * math.pi)
        ctx.stroke()
        ctx.close_path()
        ctx.restore()

    def arc(self, ctx):
        ctx.save()

        x = []
        y = []
        for it in range(3):
            if (it == 1):
                x.append(np.random.randint(0, self.MAX_X - x[0] - self.border))
                y.append(np.random.randint(0, self.MAX_Y - y[0] - self.border))
            else:
                x.append(np.random.randint(self.border, self.MAX_X - self.border))
                y.append(np.random.randint(self.border, self.MAX_Y - self.border))

        #         ctx.scale(np.random.random(), np.random.random())
        ctx.arc(x[0], y[0], min(x[1], x[0], y[0]), math.pi / np.random.randint(1, 100),
                (1 + np.random.random()) * math.pi)
        ctx.stroke()
        ctx.close_path()
        ctx.restore()

    def curve(self, ctx):

        x, y = [], []
        for it in range(4):
            x.append(np.random.randint(self.border, self.MAX_X - self.border))
            y.append(np.random.randint(self.border, self.MAX_Y - self.border))
        ctx.move_to(x[3], y[3])
        ctx.curve_to(x[0], y[0], x[1], y[1], x[2], y[2])
        ctx.stroke()
        ctx.close_path()

    def circle_fill(self, ctx):
        ctx.save()
        x = []
        y = []
        for it in range(3):
            x.append(np.random.randint(0, self.MAX_X))
            y.append(np.random.randint(0, self.MAX_Y))

        ctx.scale(np.random.random(), np.random.random())
        ctx.arc(x[0], y[0], np.random.randint(1, 50), 0, 2 * math.pi)
        ctx.fill()
        ctx.close_path()
        ctx.restore()

    def radial(self, cr):
        cr.save()
        x = []
        y = []

        for it in range(3):
            x.append(np.random.randint(self.border, self.MAX_X - self.border))
            y.append(np.random.randint(self.border, self.MAX_Y - self.border))
        cr.translate(x[0], y[0])
        ran = np.random.randint(10, 40)
        ran_1 = np.random.randint(10, 20)
        r2 = cairo.RadialGradient(0, 0, ran_1, 0, 0, ran)
        r2.add_color_stop_rgba(0.8, 0, 0, 0, 1)
        r2.add_color_stop_rgba(0, 1, 1, 1, 1)

        cr.set_source(r2)
        cr.arc(0, 0, ran, 0, math.pi * 2)
        cr.fill()
        cr.restore()

    def MergeImages(self, img_path, name, prev_degr,backgrouds_path='../../dataset/Background/*.png'):
        im1 = Image.open(img_path + name + '_h_gt.png').convert('RGB')
        if (not prev_degr):
            png_image = np.array(im1)
            d = DegradationGenerator(degradations_list=['noisy_binary_blur'],
                                     max_num_degradations=1)
            png_image = rgb_to_gray(png_image)
            png_image = img_8bit_to_float(png_image)
            png_image = d.do_degrade(png_image)
            png_image = gray_float_to_8bit(png_image)
            im1 = Image.fromarray(png_image).convert('RGB')

        backgrounds = glob(backgrouds_path)
        r = np.random.randint(0, len(backgrounds))
        im2 = Image.open(backgrounds[r])

        im2 = im2.convert('RGB')
        im2 = im2.resize((self.MAX_X, self.MAX_Y))
        im1arr = np.array(im1)
        im2arr = np.array(im2)
        im2arr = im2arr[0:self.MAX_X, 0:self.MAX_Y, :]
        im1arrF = im1arr.astype('float')
        im2arrF = im2arr.astype('float')
        additionF = (im1arrF + im2arrF) / 2
        addition = additionF.astype('uint8')
        img = Image.fromarray(addition)

        scale_method = [PIL.Image.NEAREST, PIL.Image.BOX, PIL.Image.BILINEAR, PIL.Image.HAMMING, PIL.Image.BICUBIC,
                        PIL.Image.LANCZOS]
        small = img.resize((np.random.randint(300, 800), np.random.randint(300, 800)),
                           scale_method[np.random.randint(0, 6)])
        same = small.resize((self.MAX_X, self.MAX_Y), scale_method[np.random.randint(0, 6)])
        if prev_degr:
            same *= np.repeat(np.random.normal(loc=1, scale=0.008, size=(self.MAX_X, self.MAX_Y))[:, :, np.newaxis], 3,
                              axis=2)
            same = same.astype('uint8')
        imageio.imwrite(img_path + name + '.png', same)

    def syn_degradate(self, img_path, name, prev_degr):
        png_image = np.array(Image.open(img_path + name + '_nh_gt.png'))
        if (prev_degr):
            all_degradations = [
                'distort',
                'gaussian_blur',
                'random_blotches']
        else:
            all_degradations = ['kanungo',
                                'distort',
                                'gaussian_blur',
                                'binary_blur',
                                'random_blotches']

        m_num = np.random.uniform()
        if (m_num > 0.9):
            m_num = 3
        else:
            m_num = 2

        d = DegradationGenerator(degradations_list=all_degradations,
                                 max_num_degradations=m_num)
        png_image = rgb_to_gray(png_image)
        png_image = img_8bit_to_float(png_image)
        png_image = d.do_degrade(png_image)
        png_image = gray_float_to_8bit(png_image)

        imageio.imwrite(img_path + name + '_h_gt.png', Image.fromarray(png_image).convert('RGB'))

    def get_image(self, img_path='../data/Synthetic/', name='1',backgrouds_path='/data/Background/*.png'):
        if not os.path.exists(img_path + '/svg/'):
            os.makedirs(img_path + '/svg/')
        ps = cairo.SVGSurface(img_path + 'svg' + name + ".svg", self.MAX_X, self.MAX_Y)
        ctx = cairo.Context(ps)
        ctx.save()
        ctx.set_source_rgb(1, 1, 1)
        ctx.paint()
        ctx.restore()
        ctx.move_to(0, 0)
        prev_degr = False
        for it in range(24):
            if np.random.random() < 0.5:
                r = np.random.randint(0, 6)
                if (np.random.random() < 0.05):
                    wdth = 1
                else:
                    wdth = np.random.randint(2, 7)
                ctx.set_line_width(wdth)
                if (wdth < 2):
                    prev_degr = True
                ctx.set_source_rgba(0, 0, 0, 1)
                if (r == 0):
                    self.triangle(ctx)
                elif (r == 1):
                    self.rectangle(ctx)
                elif (r == 2):
                    self.bowtie(ctx)
                elif (r == 3):
                    self.circle(ctx)
                elif (r == 4):
                    self.arc(ctx)
                elif (r == 5):
                    self.line(ctx)
                else:
                    self.curve(ctx)
        ctx.set_operator(cairo.OPERATOR_LIGHTEN)
        ps.write_to_png(img_path + name + '_nh_gt.png')
        self.syn_degradate(img_path, name, prev_degr)
        self.MergeImages(img_path, name, prev_degr,backgrouds_path)
