import tensorflow.keras as keras
from tensorflow.keras.models import Model
import tensorflow as tf
import site
import sys
import numpy as np
from importlib import import_module

import re

from functools import partial


def get_inbound_layers(layer):
    try:
        # noinspection PyProtectedMember
        inbound_layers = layer._inbound_nodes[0].inbound_layers
    except AttributeError:  # For Keras backward-compatibility.
        inbound_layers = layer.inbound_nodes[0].inbound_layers
    if not isinstance(inbound_layers, (list, tuple)):
        inbound_layers = [inbound_layers]
    return inbound_layers

def get_outbound_layers(layer):
    try:
        # noinspection PyProtectedMember
        outbound_nodes = layer._outbound_nodes
    except AttributeError:  # For Keras backward-compatibility.
        outbound_nodes = layer.outbound_nodes
    return [on.outbound_layer for on in outbound_nodes]

def get_layer_type(layer):
    return layer.__class__.__name__

def get_activation(layer):
    return layer.activation.__name__



def print_NetworkStructure(model):
    print('---------------------------------------------------\n', model.name, ' structure:\n---------------------------------------------------')
    NN = model.layers

    nl = 0
    for i,section in enumerate(NN): 
        connections = get_inbound_layers(section)
        connected_to = []
        for connection in connections:
            connected_to.append(connection.name)   
        try:
            subnet = section.layers
            print(i, ' - ', section.name, ' - ', connected_to, ': ')
            for j,layer in enumerate(subnet):
                    connections = get_inbound_layers(layer)
                    connected_to = []
                    for connection in connections:
                        connected_to.append(connection.name)
                    try:
                        subnet2 = layer.layers
                        print('  .', j, ' - ', layer.name, ' - ', connected_to, ': ')
                        for k,layer2 in enumerate(subnet2):
                            try:
                                act = layer2.activation.__name__
                            except:
                                act = ''
                            connections = get_inbound_layers(layer2)
                            connected_to = []
                            for connection in connections:
                                connected_to.append(connection.name)
                            print('    ..', k, ' - ', layer2.name, ' - ', connected_to, '    ',act)
                            nl = nl+1                 
                    except:
                        try:
                            act = layer.activation.__name__
                        except:
                            act = ''
                        print('  .', j, ' - ', layer.name, ' - ', connected_to, '    ',act)
                        nl = nl+1
        except:
            try:
                act = section.activation.__name__
            except:
                act = ''
            print(i, ' - ', section.name, ' - ', connected_to, '    ',act)
            nl = nl+1

    print('\n\nNumber of layers = ', nl)


def fix_shape(image, tar_shape = [896,1152]):
    
    image = np.array(image)
    image_shape = np.array(image.shape[:2])
    

    ratio = tar_shape[0] / image_shape[0]
    if ratio*image_shape[1] > tar_shape[1]:
      ratio = tar_shape[1] / image_shape[1]
    image_shape = (ratio * np.array(image_shape)).astype(int)
    image = tf.image.resize(image, tf.cast(image_shape, dtype=tf.int32))
    image = tf.image.pad_to_bounding_box(
        image, 0, 0, tar_shape[0], tar_shape[1]
    )
    return image, image_shape, ratio


def convert_to_WS(model, position='replace'):

    layer_regex = '.*relu.*'
    # Auxiliary dictionary to describe the network graph
    network_dict = {'input_layers_of': {}, 'new_output_tensor_of': {}}

    # Set the input layers of each layer
    for layer in model.layers:
        for node in layer._outbound_nodes:
            layer_name = node.outbound_layer.name
            if layer_name not in network_dict['input_layers_of']:
                network_dict['input_layers_of'].update(
                        {layer_name: [layer.name]})
            elif layer.name not in network_dict['input_layers_of'][layer_name]:
                network_dict['input_layers_of'][layer_name].append(layer.name)

    # Set the output tensor of the input layer
    m_input = model.input
    if len(m_input)==1:
        m_input=m_input[0]
    network_dict['new_output_tensor_of'].update(
            {model.layers[0].name: m_input})

    # Iterate over all layers after the input
    model_outputs = []
    for layer in model.layers[1:]:

        # Determine input tensors
        layer_input = [network_dict['new_output_tensor_of'][layer_aux] 
                for layer_aux in network_dict['input_layers_of'][layer.name]]
        if len(layer_input)==1:
            layer_input = layer_input[0]

        # Insert layer if name matches the regular expression
        if re.match(layer_regex, layer.name):
            if position == 'replace':
                x = layer_input
            elif position == 'after':
                x = layer(layer_input)
            elif position == 'before':
                pass
            else:
                raise ValueError('position must be: before, after or replace')

            new_layer = Spiking_BRelu(name=layer.name)
            # if insert_layer_name:
            #     new_layer.name = insert_layer_name
            # else:
            #     new_layer.name = '{}_{}'.format(layer.name, 
            #                                     new_layer.name)
            x = new_layer(x)
            # print('New layer: {} Old layer: {} Type: {}'.format(new_layer.name,
            #                                                 layer.name, position))
            if position == 'before':
                x = layer(x)
        else:
            x = layer(layer_input)

        # Set new output tensor (the original one, or the one of the inserted
        # layer)
        network_dict['new_output_tensor_of'].update({layer.name: x})

        # Save tensor in output list if it is output in initial model
        if layer_name in model.output_names:
            model_outputs.append(x)

    return Model(inputs=model.inputs, outputs=model_outputs)



def insert_backbone(model, layer_regex, insert_layer_factory, in_idx, out_layers, input_shape, position='replace'):

    # Auxiliary dictionary to describe the network graph
    network_dict = {'input_layers_of': {}, 'new_output_tensor_of': {}}

    # Set the input layers of each layer
    for layer in model.layers:
        for node in layer._outbound_nodes:
            layer_name = node.outbound_layer.name
            if layer_name not in network_dict['input_layers_of']:
                network_dict['input_layers_of'].update(
                        {layer_name: [layer.name]})
            elif layer.name not in network_dict['input_layers_of'][layer_name]:
                network_dict['input_layers_of'][layer_name].append(layer.name)

    # Set the output tensor of the input layer
    m_input = tf.keras.Input(shape=input_shape)

    # network_dict['new_output_tensor_of'].update(
    #         {model.layers[0].name: m_input})

    # Iterate over all layers after the input
    model_outputs = []
    subs_layers = []
    for ix,layer in enumerate(model.layers):
        if ix==0: 
            layer_input = m_input
        else:
            # Determine input tensors
            layer_input = [network_dict['new_output_tensor_of'][layer_aux] 
                for layer_aux in network_dict['input_layers_of'][layer.name]]
        try:
            if len(layer_input)==1:
                layer_input = layer_input[0]
        except:
            pass

        # Insert layer if name matches the regular expression
        if re.match(layer_regex, layer.name):
            subs_layers.append(layer.name)
            if position == 'replace':
                x = layer_input
            elif position == 'after':
                x = layer(layer_input)
            elif position == 'before':
                pass
            else:
                raise ValueError('position must be: before, after or replace')

            new_layer = insert_layer_factory()
            x = new_layer(x)
            print('Layer: {} Transformed from {} to {}'.format(layer.name,
                                                        layer.__class__.__name__, 
                                                        new_layer.__class__.__name__))
            if position == 'before':
                x = layer(x)
        else:
            for layer_name in network_dict['input_layers_of'][layer.name]:
                if layer_name in subs_layers:
                    layer_input = layer_input[in_idx.index(ix)]
            x = layer(layer_input)

        # Set new output tensor (the original one, or the one of the inserted
        # layer)
        network_dict['new_output_tensor_of'].update({layer.name: x})

        # Save tensor in output list if it is output in initial model
        if layer in out_layers:
            model_outputs.append(x)

    return tf.keras.Model(inputs=m_input, outputs=model_outputs)


def target_subnet(model, subnet_idx):
    """
    model = Keras model
    subnet_idx = indexes of the subnet you want to target.
        ex: subnet_idx = [1,0] -> targets model.layers[1].layers[0]
    """
    subnet = model
    for idx in subnet_idx:
        subnet = subnet.layers[idx]
    return subnet

def insert_layer(model, layer_regex, insert_layer_factory, position='replace'):

    # Auxiliary dictionary to describe the network graph
    network_dict = {'input_layers_of': {}, 'new_output_tensor_of': {}}

    # Set the input layers of each layer
    for layer in model.layers:
        for node in layer._outbound_nodes:
            layer_name = node.outbound_layer.name
            if layer_name not in network_dict['input_layers_of']:
                network_dict['input_layers_of'].update(
                        {layer_name: [layer.name]})
            elif layer.name not in network_dict['input_layers_of'][layer_name]:
                network_dict['input_layers_of'][layer_name].append(layer.name)

    # Set the output tensor of the input layer
    try:
        m_input = model.input
    except:
        m_input = model.layers[0].input
    try:
        if len(m_input)==1:
            m_input=m_input[0]
    except:
        pass
    network_dict['new_output_tensor_of'].update(
            {model.layers[0].name: m_input})

    # Iterate over all layers after the input
    model_outputs = []
    for layer in model.layers[1:]:

        # Determine input tensors
        layer_input = [network_dict['new_output_tensor_of'][layer_aux] 
                for layer_aux in network_dict['input_layers_of'][layer.name]]
        if len(layer_input)==1:
            layer_input = layer_input[0]

        flag=False
        for keyword in layer_regex:
            if re.match(keyword, layer.name):
                flag=True

        # Insert layer if name matches the regular expression
        if flag:
            if position == 'replace':
                x = layer_input
            elif position == 'after':
                x = layer(layer_input)
            elif position == 'before':
                pass
            else:
                raise ValueError('position must be: before, after or replace')
            
            new_layer = insert_layer_factory(layer)
            x = new_layer(x)

            try:
                act = " ("+str(get_activation(layer))+")"
            except:
                act = ""
            print('Layer: {}[{}] ----> {}[{}]'.format(layer.name,
                                                    layer.__class__.__name__+act,
                                                    new_layer.name, 
                                                    new_layer.__class__.__name__))
            if position == 'before':
                x = layer(x)
        else:
            x = layer(layer_input)

        # Set new output tensor (the original one, or the one of the inserted
        # layer)
        network_dict['new_output_tensor_of'].update({layer.name: x})

        # Save tensor in output list if it is output in initial model
        if layer_name in model.output_names:
            model_outputs.append(x)

    return Model(inputs=model.inputs, outputs=model_outputs)

def insert_layer_old(model, layer_regex, insert_layer_factory, position='replace'):

    # Auxiliary dictionary to describe the network graph
    network_dict = {'input_layers_of': {}, 'new_output_tensor_of': {}}

    # Set the input layers of each layer
    for layer in model.layers:
        for node in layer._outbound_nodes:
            layer_name = node.outbound_layer.name
            if layer_name not in network_dict['input_layers_of']:
                network_dict['input_layers_of'].update(
                        {layer_name: [layer.name]})
            elif layer.name not in network_dict['input_layers_of'][layer_name]:
                network_dict['input_layers_of'][layer_name].append(layer.name)

    # Set the output tensor of the input layer
    m_input = model.input
    try:
        if len(m_input)==1:
            m_input=m_input[0]
    except:
        pass
    network_dict['new_output_tensor_of'].update(
            {model.layers[0].name: m_input})

    # Iterate over all layers after the input
    model_outputs = []
    for layer in model.layers[1:]:

        # Determine input tensors
        layer_input = [network_dict['new_output_tensor_of'][layer_aux] 
                for layer_aux in network_dict['input_layers_of'][layer.name]]
        if len(layer_input)==1:
            layer_input = layer_input[0]

        flag=False
        for keyword in layer_regex:
            if re.match(keyword, layer.name):
                flag=True

        # Insert layer if name matches the regular expression
        if flag:
            if position == 'replace':
                x = layer_input
            elif position == 'after':
                x = layer(layer_input)
            elif position == 'before':
                pass
            else:
                raise ValueError('position must be: before, after or replace')
            
            new_layer = insert_layer_factory(layer)
            x = new_layer(x)

            try:
                act = " ("+str(get_activation(layer))+")"
            except:
                act = ""
            print('Layer: {}[{}] ----> {}[{}]'.format(layer.name,
                                                    layer.__class__.__name__+act,
                                                    new_layer.name, 
                                                    new_layer.__class__.__name__))
            if position == 'before':
                x = layer(x)
        else:
            x = layer(layer_input)

        # Set new output tensor (the original one, or the one of the inserted
        # layer)
        network_dict['new_output_tensor_of'].update({layer.name: x})

        # Save tensor in output list if it is output in initial model
        if layer_name in model.output_names:
            model_outputs.append(x)

    return Model(inputs=model.inputs, outputs=model_outputs)


def decode_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32)
    return image

def read_tfrecord(example, labeled=True):
    sample = {}
    tfrecord_format = (
        {
            'image/height': tf.io.FixedLenFeature([], tf.int64),
            'image/width': tf.io.FixedLenFeature([], tf.int64),
            'image/source_id': tf.io.FixedLenFeature([], tf.string),
            'image/encoded': tf.io.FixedLenFeature([], tf.string),
            'image/format': tf.io.FixedLenFeature([], tf.string),
            'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
            'image/object/class/text': tf.io.VarLenFeature(tf.string),
            'image/object/class/label': tf.io.VarLenFeature(tf.int64)
        }
        if labeled
        else {
            'image/height': tf.io.FixedLenFeature([], tf.int64),
            'image/width': tf.io.FixedLenFeature([], tf.int64),
            'image/source_id': tf.io.FixedLenFeature([], tf.string),
            'image/encoded': tf.io.FixedLenFeature([], tf.string),
            'image/format': tf.io.FixedLenFeature([], tf.string),
            }
    )
    example = tf.io.parse_single_example(example, tfrecord_format)
    sample["image"] = tf.image.decode_jpeg(example['image/encoded'], channels=3)
    if labeled:
        sample["objects"]={}
        sample["objects"]["bbox"] = tf.transpose(tf.convert_to_tensor([
            tf.sparse.to_dense(example['image/object/bbox/xmin']),
            tf.sparse.to_dense(example['image/object/bbox/ymin']),
            tf.sparse.to_dense(example['image/object/bbox/xmax']),
            tf.sparse.to_dense(example['image/object/bbox/ymax'])
        ]))
        sample["objects"]["label"] = tf.sparse.to_dense(example['image/object/class/label'])
    return sample

def load_dataset(filenames, labeled=True, stream_unordered=True):
    from functools import partial
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False  # disable order, increase speed
    dataset = tf.data.TFRecordDataset(
        filenames
    )  # automatically interleaves reads from multiple files
    if stream_unordered:
        dataset = dataset.with_options(
            ignore_order
        )  # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(
        partial(read_tfrecord, labeled=labeled), num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    # returns a dataset of (image, label) pairs if labeled=True or just images if labeled=False
    return dataset

