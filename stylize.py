import PIL
from PIL import Image
import numpy as np
from keras import backend
from keras.applications.vgg16 import VGG16
from scipy.optimize import fmin_l_bfgs_b
from keras.models import Model
import time

def stylize_image(content_path, style_paths):
    width = 512
    height = 512

    # open image and resize for output image
    content_image = Image.open(content_path)
    content_image = content_image.resize((width, height), PIL.Image.ANTIALIAS)

    # we create a list of style image arrays
    style_image_array = []
    for p in style_paths:
        style_image = Image.open(p)
        style_image = style_image.resize((width, height), PIL.Image.ANTIALIAS)
        style_array = np.asarray(style_image, dtype=np.float64)
        style_array = np.expand_dims(style_array, axis=0)
        style_array[:, :, :, 0] -= 103.939
        style_array[:, :, :, 1] -= 116.779
        style_array[:, :, :, 2] -= 123.68
        style_array = style_array[:, :, :, ::-1]
        style_image_array.append(style_array)

    # convert to numpy array
    content_array = np.asarray(content_image, dtype=np.float64)
    content_array = np.expand_dims(content_array, axis=0)

    # preprocess array for VGG ImageNet
    content_array[:, :, :, 0] -= 103.939
    content_array[:, :, :, 1] -= 116.779
    content_array[:, :, :, 2] -= 123.68

    # flip from RGB to BGR, since the CNN uses this format
    content_array = content_array[:, :, :, ::-1]

    # set up tensorflow for extracting layer features
    tf_content = backend.variable(content_array)
    tf_style = [backend.variable(a) for a in style_image_array]
    tf_combo = backend.placeholder((1, width, height, 3))
    feed_tensor = backend.concatenate([tf_content] + tf_style + [tf_combo], axis=0)
    model = VGG16(input_tensor=feed_tensor, weights='imagenet', include_top=False)
    loss = backend.variable(0.0)

    # these are the feature layers we need to get the output from, naming convention as defined in keras
    # layer selection is done based on the paper
    # these are defined only for visualizing layer outputs
    layers = dict([(layer.name, layer.output) for layer in model.layers])

    def content_loss(combo, content):
        return backend.sum(backend.square(combo - content))

    content_weight = 0.025
    style_weight = 5.0
    total_variation_weight = 1.0

    # extract content features for generating content loss
    content_layer_features = layers['block2_conv2']
    content_features = content_layer_features[0, :, :, :]
    combo_features = content_layer_features[-1, :, :, :]

    # update loss function to include content loss
    print('Updating loss function to include content loss')
    loss = loss + content_weight * content_loss(combo_features, content_features)

    # function returns gram matrix for a feature vector
    # gram matrix is simply outer product
    def gram_matrix(x):
        vectorized_x = backend.batch_flatten(backend.permute_dimensions(x, (2, 0, 1)))
        gram = backend.dot(vectorized_x, backend.transpose(vectorized_x))
        return gram

    def style_loss(style, combination):
        S = gram_matrix(style)
        C = gram_matrix(combination)
        channels = 3
        size = height * width
        return backend.sum(backend.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

    # update loss function to include style loss for multiple images
    style_layers = ['block1_conv2', 'block2_conv2',
                    'block3_conv3', 'block4_conv3',
                    'block5_conv3']

    print('Updating loss function to include style loss')
    for layer_name in style_layers:
        for i, s in enumerate(style_image_array):
            layer_features = layers[layer_name]
            style_features = layer_features[i+1, :, :, :]
            combination_features = layer_features[-1, :, :, :]
            sloss = style_loss(style_features, combination_features)
            loss += (style_weight / (len(style_image_array) * len(style_layers))) * sloss

    # update loss function to include style loss for each of the style images
    # see wikipedia article on total variation denoising
    def total_variational_loss(x):
        diff1 = backend.square(x[:, :height-1, :width-1, :] - x[:, 1:, :width-1, :])
        diff2 = backend.square(x[:, :height-1, :width-1, :] - x[:, :height-1, 1:, :])
        return backend.sum(backend.pow(diff1, 2) + backend.pow(diff2, 2))

    print('Updating loss function to include variational loss')
    loss = loss + total_variation_weight * total_variational_loss(tf_combo)
    grads = backend.gradients(loss, tf_combo)
    outputs = [loss]
    outputs += grads
    f_outputs = backend.function([tf_combo], outputs)

    def eval_loss_and_grads(x):
        x = x.reshape((1, height, width, 3))
        outs = f_outputs([x])
        loss_value = outs[0]
        grad_values = outs[1].flatten().astype(np.float64)
        return loss_value, grad_values

    # This is done only to prevent recomputing gradient and total loss repeatedly
    class LossGradient:
        def __init__(self):
            self.loss_value = None
            self.grads_values = None

        def loss(self, x):
            assert self.loss_value is None
            loss_value, grad_values = eval_loss_and_grads(x)
            self.loss_value = loss_value
            self.grad_values = grad_values
            return self.loss_value

        def grads(self, x):
            assert self.loss_value is not None
            grad_values = np.copy(self.grad_values)
            self.loss_value = None
            self.grad_values = None
            return grad_values

    lg = LossGradient()
    combo_array = np.random.uniform(0, 255, (1, height, width, 3)) - 128.

    print('Running eval iterations')
    for i in range(10):
        print('Start of iteration', i)
        start_time = time.time()

        temp = np.copy(combo_array)
        temp = temp.reshape((width, height, 3))
        temp = temp[:, :, ::-1]
        temp[:, :, 0] += 103.939
        temp[:, :, 1] += 116.779
        temp[:, :, 2] += 123.68
        temp = np.clip(temp, 0, 255).astype(np.uint8)
        temp = Image.fromarray(temp)
        temp.save("outputs/blend" + str(i) + ".jpg")

        combo_array, min_val, info = fmin_l_bfgs_b(lg.loss, combo_array.flatten(), fprime=lg.grads, maxfun=20)
        print('Current loss value:', min_val)
        end_time = time.time()
        print('Iteration {} completed in {}s'.format(i, end_time - start_time))

    # flip output into RGB format again
    combo_array = combo_array.reshape((width, height, 3))
    combo_array = combo_array[:, :, ::-1]
    combo_array[:, :, 0] += 103.939
    combo_array[:, :, 1] += 116.779
    combo_array[:, :, 2] += 123.68
    combo_array = np.clip(combo_array, 0, 255).astype(np.uint8)
    combined_image = Image.fromarray(combo_array)

    return combined_image

if __name__ == "__main__":
    content_image = 'content_images/us.jpg'
    style_image = ['style_images/picasso.jpg', 'style_images/style1.jpg']

    combined_image = stylize_image(content_image, style_image)
    combined_image.show()
    combined_image.save('outputs/output3.jpg')
