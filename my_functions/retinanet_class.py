import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

# %% [markdown]
# ## Building Neural Network
# %% [markdown]
# ### Backbone (ResNet50) --> AvgPooling

# %%
from my_functions.useful_functions import insert_layer

def layer_factory(layer):
    kwargs=layer.get_config()
    return keras.layers.AveragePooling2D(**kwargs)

def get_backbone(bbone_path, input_shape=[None, None, 3]):
    if not os.path.exists(bbone_path):
        print("\nLoading ResNet50 and transforming it to AveragePooling...")
        backbone = keras.applications.ResNet50(
            include_top=False, input_shape=input_shape
        )
        backbone = insert_layer(backbone, ['.*_pool.*'], layer_factory)
        print("\nSaving AveragePool backbone to ["+str(bbone_path)+"]")
        backbone.save(bbone_path)
    else:
        print("Loading AveragePool backbone from ["+str(bbone_path)+"]")

    backbone = tf.keras.models.load_model(bbone_path)

    c3_output, c4_output, c5_output = [
        backbone.get_layer(layer_name).output
        for layer_name in ["conv3_block4_out", "conv4_block6_out", "conv5_block3_out"]
    ]
    return keras.Model(
        inputs=[backbone.inputs], outputs=[c3_output, c4_output, c5_output]
    )

# %% [markdown]
# ### Feature Pyramid Network (FPN)

# %%
class FeaturePyramid(keras.Model):
    """Builds the Feature Pyramid with the feature maps from the backbone.

    Attributes:
      num_classes: Number of classes in the dataset.
      backbone: The backbone to build the feature pyramid from.
        Currently supports ResNet50 only.
    """

    def __init__(self, backbone=None, input_shape=[None, None, 3], **kwargs):
        super(FeaturePyramid, self).__init__(name="FeaturePyramid", **kwargs)
        #self.input_layer = keras.Input(shape=input_shape)
        self.backbone = backbone if backbone else get_backbone(input_shape)
        self.conv_c3_1x1 = keras.layers.Conv2D(256, 1, 1, "same")
        self.conv_c4_1x1 = keras.layers.Conv2D(256, 1, 1, "same")
        self.conv_c5_1x1 = keras.layers.Conv2D(256, 1, 1, "same")

        self.upsample_2x_p5 = keras.layers.UpSampling2D(2)
        self.add_p4_p5 = keras.layers.Add()
        self.upsample_2x_p4 = keras.layers.UpSampling2D(2)
        self.add_p3_p4 = keras.layers.Add()

        self.conv_c3_3x3 = keras.layers.Conv2D(256, 3, 1, "same")
        self.conv_c4_3x3 = keras.layers.Conv2D(256, 3, 1, "same")
        self.conv_c5_3x3 = keras.layers.Conv2D(256, 3, 1, "same")
        self.conv_c6_3x3 = keras.layers.Conv2D(256, 3, 2, "same")

        self.relu_p6 = keras.layers.ReLU()

        self.conv_c7_3x3 = keras.layers.Conv2D(256, 3, 2, "same")


    def call(self, images, training=False):
        #images = self.input_layer(images)
        c3_output, c4_output, c5_output = self.backbone(images, training=training)
        p3_output = self.conv_c3_1x1(c3_output)
        p4_output = self.conv_c4_1x1(c4_output)
        p5_output = self.conv_c5_1x1(c5_output)


        p4_output = self.add_p4_p5([p4_output, self.upsample_2x_p5(p5_output)])
        p3_output = self.add_p3_p4([p3_output, self.upsample_2x_p4(p4_output)])

        p3_output = self.conv_c3_3x3(p3_output)
        p4_output = self.conv_c4_3x3(p4_output)
        p5_output = self.conv_c5_3x3(p5_output)
        p6_output = self.conv_c6_3x3(c5_output)
        p7_output = self.conv_c7_3x3(self.relu_p6(p6_output))
        
        return p3_output, p4_output, p5_output, p6_output, p7_output

# %% [markdown]
# ### Build subnetworks (classification and box regression heads)

# %%
def build_head(output_filters, bias_init, name):
    """Builds the class/box predictions head.

    Arguments:
      output_filters: Number of convolution filters in the final layer.
      bias_init: Bias Initializer for the final convolution layer.

    Returns:
      A keras sequential model representing either the classification
        or the box regression head depending on `output_filters`.
    """
    head = keras.Sequential([keras.Input(shape=[None, None, 256])], name=name)
    kernel_init = tf.initializers.RandomNormal(0.0, 0.01)
    for _ in range(4):
        head.add(keras.layers.Conv2D(256, 3, padding="same", kernel_initializer=kernel_init))
        head.add(keras.layers.ReLU())
    head.add(keras.layers.Conv2D(output_filters,3,1,padding="same",kernel_initializer=kernel_init,bias_initializer=bias_init))
    return head

# %% [markdown]
# ### Build Whetstone RetinaNet

# %%
# Build Whetstone model
from snntoolbox.simulation.backends import custom_layers

class RetinaNet(keras.Model):
    """A subclassed Keras model implementing the RetinaNet architecture.

    Attributes:
      num_classes: Number of classes in the dataset.
      backbone: The backbone to build the feature pyramid from.
        Currently supports ResNet50 only.
    """

    def __init__(self, num_classes, backbone=None, input_shape=[None, None, 3], **kwargs):
        super(RetinaNet, self).__init__(name="RetinaNet", **kwargs)
        #self.input_layer = keras.Input(shape=input_shape)
        self.fpn = FeaturePyramid(backbone)
        self.num_classes = num_classes

        prior_probability = tf.constant_initializer(-np.log((1 - 0.01) / 0.01))
        self.cls_head = build_head(9 * num_classes, prior_probability, "ClassHead")
        self.box_head = build_head(9 * 4, "zeros", "BoxHead")

        self.cls_resh = keras.layers.Reshape([1, -1, self.num_classes])
        self.box_resh = keras.layers.Reshape([1, -1, 4])
        # self.cls_resh = custom_layers.NormReshape([1, -1, self.num_classes])
        # self.box_resh = custom_layers.NormReshape([1, -1, 4])

        self.cls_cat = keras.layers.Concatenate(axis=2)
        self.box_cat = keras.layers.Concatenate(axis=2)
        self.cat = keras.layers.Concatenate(axis=-1)



    def call(self, image, training=False):
        #image = self.input_layer(image)
        features = self.fpn(image, training=training)
        N = tf.shape(image)[0]
        cls_outputs = []
        box_outputs = []
        for feature in features:
            box_outputs.append(self.box_resh(self.box_head(feature))) 
            cls_outputs.append(self.cls_resh(self.cls_head(feature)))

        cls_outputs = self.cls_cat(cls_outputs)
        box_outputs = self.box_cat(box_outputs)

        output = self.cat([box_outputs, cls_outputs])
        # output = tf.reshape(output,[1,-1,84])

        return output