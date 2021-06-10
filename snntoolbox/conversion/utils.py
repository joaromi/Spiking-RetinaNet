# -*- coding: utf-8 -*-
"""

This module performs modifications on the network parameters during conversion
from analog to spiking.

.. autosummary::
    :nosignatures:

    normalize_parameters

@author: rbodo
"""

import os

import json
from collections import OrderedDict
from tensorflow.keras.models import Model
import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm


def layer_norm(model, config, norm_set=None, num_samples=None, divisions=1, free_GB=2, layers_to_plot=[], equiv_layers=[], **kwargs):

    from snntoolbox.parsing.utils import get_inbound_layers_with_params

    print("Normalizing parameters...")

    norm_dir = kwargs[str('path')] if 'path' in kwargs else \
        os.path.join(config.get('paths', 'path_wd'),
                     'layer_norm')

    activ_dir = os.path.join(norm_dir, 'activations')
    norm_activ_dir = os.path.join(norm_dir, 'norm_activations')
    if not os.path.exists(activ_dir):
        os.makedirs(activ_dir)
    # Store original weights for later plotting
    if not os.path.isfile(os.path.join(activ_dir, 'weights.npz')):
        weights = {}
        for layer in model.layers:
            w = layer.get_weights()
            if len(w) > 0:
                weights[layer.name] = w[0]
        np.savez_compressed(os.path.join(activ_dir, 'weights.npz'), **weights)

    batch_size = config.getint('simulation', 'batch_size')

    # Either load scale factors from disk, or get normalization data set to
    # calculate them.
    filepath = os.path.join(norm_dir, config.get('normalization',
                                                'percentile') + '.json')
    samples_in_division = int(num_samples/divisions) 
    if 'scale_facs' in kwargs:
        scale_facs = kwargs[str('scale_facs')]
        print('(*) Scale factors input as **kwargs to the function')
    elif os.path.isfile(filepath):
        with open(filepath) as f:
            scale_facs = json.load(f)
        print('(*) Scale factors loaded from previous run: [', filepath, ']')
    elif norm_set is not None: 
        print('(*) Computing scale factors from the model. This might take a while...')    
        print("Using {} samples for normalization: {} packs of {} samples".format(num_samples,
                                                                            divisions,
                                                                            samples_in_division))
        sizes = [
            samples_in_division * float(np.array(layer.output_shape[1:]).prod()) * float(32 /(8 * 1e9)) \
                 for layer in model.layers if len(layer.weights) > 0 ]
        size_str = ['{:.2f}'.format(s) for s in sizes]
        print('INFO: Size of layer activations: ', size_str ,'GB\n')       
        req_space = max(sizes)
        print("Required {:.2f} GB of free space for the largest activation. \n".format(req_space))
        print("In total, {:.2f} GB of information flow. \n".format(sum(sizes)))
        if req_space > free_GB:
            import warnings
            warnings.warn("Required space is larger than specified free space of "+str(free_GB)+
            "GB. Reduce size of data set or increase available space.", ResourceWarning)
            print('[Skipping normalization]')
            return
        scale_facs = OrderedDict({model.layers[0].name: 1})
    else:
        import warnings
        warnings.warn("Scale factors or normalization data set could not be "
                      "loaded. Proceeding without normalization.",
                      RuntimeWarning)
        return

    # If scale factors have not been computed in a previous run, do so now.
    if len(scale_facs) == 1:
        i = 0
        for layer in model.layers:
            # Skip if layer has no parameters
            if len(layer.weights) == 0:
                continue
            perc = get_percentile(config, i)
            accum_samples=0
            layer_lambdas = []
            x = []
            for sample in norm_set.take(num_samples):
                x.append(np.array(sample[0], dtype=np.float32))
                accum_samples+=1
                if accum_samples>=samples_in_division:
                    accum_samples = 0
                    x = np.array(x)[:,0]
                    tf.keras.backend.clear_session()
                    activations = np.array(Model(model.input, layer.output).predict(x, batch_size))
                    nonzero_activations = activations[np.nonzero(activations)]
                    x = []
                    if layer.name in layers_to_plot and not layer_lambdas:    
                        print("Writing activations to disk...")
                        np.savez_compressed(os.path.join(activ_dir, layer.name), activations)
                    del activations
                    layer_lambdas.append(get_scale_fac(nonzero_activations, perc))
                    del nonzero_activations
            layer_lambdas = np.array(layer_lambdas)
            scale_facs[layer.name] = np.average(layer_lambdas)
            print("Layer "+str(layer.name)+" - Scale factor: {:.2f} ± {:.2f}.".format(scale_facs[layer.name],
                                                    max(abs(scale_facs[layer.name]-layer_lambdas))))
            i += 1
        del x
        # Write scale factors to disk
        filepath = os.path.join(norm_dir, config.get('normalization',
                                                     'percentile') + '.json')
        from snntoolbox.utils.utils import confirm_overwrite
        if config.get('output', 'overwrite') or confirm_overwrite(filepath):
            with open(filepath, str('w')) as f:
                json.dump(scale_facs, f)

    # Fix RNet heads, they need to be equally weighted
    if equiv_layers:
        for layer in model.layers:
            for ix,group in enumerate(equiv_layers):
                if layer.name in group:
                    refs = equiv_layers.pop(ix)
                    scale_fac = 0
                    for ref in refs: scale_fac += scale_facs[ref] # average
                    scale_fac /= len(refs)
                    for ref in refs: scale_facs[ref] = scale_fac
                    del refs, scale_fac
        
        filepath = os.path.join(norm_dir, config.get('normalization',
                                                     'percentile') + '_mod.json')
        with open(filepath, str('w')) as f:
            json.dump(scale_facs, f)


    # Apply scale factors to normalize the parameters.
    for layer in model.layers:
        # Skip if layer has no parameters
        if len(layer.weights) == 0:
            continue

        # Scale parameters
        parameters = layer.get_weights()
        if layer.activation.__name__ == 'softmax':
            # When using a certain percentile or even the max, the scaling
            # factor can be extremely low in case of many output classes
            # (e.g. 0.01 for ImageNet). This amplifies weights and biases
            # greatly. But large biases cause large offsets in the beginning
            # of the simulation (spike input absent).
            scale_fac = 1.0
            print("Using scale factor {:.2f} for softmax layer.".format(
                scale_fac))
        else:
            scale_fac = scale_facs[layer.name]
        inbound = get_inbound_layers_with_params(layer)
        if len(inbound) == 0:  # Input layer
            parameters_norm = [
                parameters[0] * scale_facs[model.layers[0].name] / scale_fac,
                parameters[1] / scale_fac]
        elif len(inbound) == 1:
            parameters_norm = [
                parameters[0] * scale_facs[inbound[0].name] / scale_fac,
                parameters[1] / scale_fac]
        else:
            # In case of this layer receiving input from several layers, we can
            # apply scale factor to bias as usual, but need to rescale weights
            # according to their respective input.
            parameters_norm = [parameters[0], parameters[1] / scale_fac]
            if parameters[0].ndim == 4:
                # In conv layers, just need to split up along channel dim.
                offset = 0  # Index offset at input filter dimension
                for inb in inbound:
                    f_out = inb.filters  # Num output features of inbound layer
                    f_in = range(offset, offset + f_out)
                    parameters_norm[0][:, :, f_in, :] *= \
                        scale_facs[inb.name] / scale_fac
                    offset += f_out
            else:
                # Fully-connected layers need more consideration, because they
                # could receive input from several conv layers that are
                # concatenated and then flattened. The neuron position in the
                # flattened layer depend on the image_data_format.
                raise NotImplementedError

        # Check if the layer happens to be Sparse
        # if the layer is sparse, add the mask to the list of parameters
        if len(parameters) == 3:
            parameters_norm.append(parameters[-1])
        # Update model with modified parameters
        layer.set_weights(parameters_norm)

    # Plot distributions of weights and activations before and after norm.
    if 'normalization_activations' in eval(config.get('output', 'plot_vars')) and layers_to_plot:
        from snntoolbox.simulation.plotting import plot_hist
        from snntoolbox.simulation.plotting import plot_max_activ_hist

        # All layers in one plot. Assumes model.get_weights() returns
        # [w, b, w, b, ...].
        # from snntoolbox.simulation.plotting import plot_weight_distribution
        # plot_weight_distribution(norm_dir, model)

        print("Plotting distributions of weights and activations before and "
              "after normalizing...")

        # Load original parsed model to get parameters before normalization
        weights = np.load(os.path.join(activ_dir, 'weights.npz'))

        x=[]
        for sample in norm_set.take(samples_in_division):
            x.append(np.array(sample[0], dtype=np.float32))
        x = np.array(x)[:,0]
                
        for idx, layer in enumerate(model.layers):
            # Skip if layer has no parameters
            if len(layer.weights) == 0 or layer.name not in layers_to_plot:
                continue

            label = str(idx) + layer.__class__.__name__ \
                if config.getboolean('output', 'use_simple_labels') \
                else layer.name
            parameters = weights[layer.name]
            parameters_norm = layer.get_weights()[0]
            weight_dict = {'weights': parameters.flatten(),
                           'weights_norm': parameters_norm.flatten()}
            plot_hist(weight_dict, 'Weight', label, norm_dir)

            # Load activations of model before normalization
            try:
                activations = np.load(os.path.join(activ_dir, layer.name + '.npz'))['arr_0']
            except IOError:
                    print('Error when loading activations from [',
                            os.path.join(activ_dir, layer.name + '.npz'),
                            ']. \n...Skipping layer...')
                    continue

            if activations is None:
                continue

            # Compute activations with modified parameters
            nonzero_activations = activations[np.nonzero(activations)]
            tf.keras.backend.clear_session()
            activations_norm = np.array(Model(model.input, layer.output).predict(x, batch_size))
            activation_dict = {'Activations': nonzero_activations,
                               'Activations_norm':
                               activations_norm[np.nonzero(activations_norm)]}
            scale_fac = scale_facs[layer.name]
            plot_hist(activation_dict, 'Activation', label, norm_dir,
                      scale_fac)
            ax = tuple(np.arange(len(layer.output_shape))[1:])
            plot_max_activ_hist(
                {'Activations_max': np.max(activations, axis=ax)},
                'Maximum Activation', label, norm_dir, scale_fac)
            np.savez_compressed(os.path.join(norm_activ_dir, layer.name), activations_norm)
    print('')


def channel_norm(model, config, norm_set=None, num_samples=None, divisions=1, free_GB=2, layers_to_plot=[], equiv_layers=[], **kwargs):

    from snntoolbox.parsing.utils import get_inbound_layers_with_params

    print("Normalizing parameters...")

    norm_dir = kwargs[str('path')] if 'path' in kwargs else \
        os.path.join(config.get('paths', 'path_wd'),
                     'channel_norm')

    activ_dir = os.path.join(norm_dir, 'activations')
    norm_activ_dir = os.path.join(norm_dir, 'norm_activations')
    if not os.path.exists(activ_dir):
        os.makedirs(activ_dir)
    # Store original weights for later plotting
    if not os.path.isfile(os.path.join(activ_dir, 'weights.npz')):
        weights = {}
        for layer in model.layers:
            w = layer.get_weights()
            if len(w) > 0:
                weights[layer.name] = w[0]
        np.savez_compressed(os.path.join(activ_dir, 'weights.npz'), **weights)

    batch_size = config.getint('simulation', 'batch_size')

    # Either load scale factors from disk, or get normalization data set to
    # calculate them.
    filepath = os.path.join(norm_dir, config.get('normalization',
                                                'percentile') + '.json')
    
    samples_in_division = int(num_samples/divisions) 
    if 'scale_facs' in kwargs:
        scale_facs = kwargs[str('scale_facs')]
        print('(*) Scale factors input as **kwargs to the function')
    elif os.path.isfile(filepath):
        with open(filepath) as f:
            scale_facs = json.load(f)
        print('(*) Scale factors loaded from previous run: [', filepath, ']')
    elif norm_set is not None: 
        print('(*) Computing scale factors from the model. This might take a while...')    
        print("Using {} samples for normalization: {} packs of {} samples".format(num_samples,
                                                                            divisions,
                                                                            samples_in_division))
        sizes = [
            samples_in_division * float(np.array(layer.output_shape[1:]).prod()) * float(32 /(8 * 1e9)) \
                 for layer in model.layers if len(layer.weights) > 0 ]
        size_str = ['{:.2f}'.format(s) for s in sizes]
        print('INFO: Size of layer activations: ', size_str ,'GB\n')       
        req_space = max(sizes)
        print("Required {:.2f} GB of free space for the largest activation. \n".format(req_space))
        print("In total, {:.2f} GB of information flow. \n".format(sum(sizes)))
        if req_space > free_GB:
            import warnings
            warnings.warn("Required space is larger than specified free space of "+str(free_GB)+
            "GB. Reduce size of data set or increase available space.", ResourceWarning)
            print('[Skipping normalization]')
            return
        scale_facs = OrderedDict({model.layers[0].name: 1})
    else:
        import warnings
        warnings.warn("Scale factors or normalization data set could not be "
                      "loaded. Proceeding without normalization.",
                      RuntimeWarning)
        return

    # If scale factors have not been computed in a previous run, do so now.
    if len(scale_facs) == 1:
        i = 0
        for layer in model.layers:
            # Skip if layer has no parameters
            if len(layer.weights) == 0:
                continue
            perc = get_percentile(config, i)
            accum_samples=0
            layer_lambdas = []
            x = []
            for sample in norm_set.take(num_samples):
                x.append(np.array(sample[0], dtype=np.float32))
                accum_samples+=1
                if accum_samples>=samples_in_division:
                    accum_samples = 0
                    x = np.array(x)[:,0]
                    tf.keras.backend.clear_session()
                    activations = np.array(Model(model.input, layer.output).predict(x, batch_size))
                    x = []
                    if layer.name in layers_to_plot and not layer_lambdas:    
                        print("Writing "+layer.name+"'s activations to disk...")
                        np.savez_compressed(os.path.join(activ_dir, layer.name), activations)
                    layer_lambdas.append(get_scale_fac_channel(activations, perc))
                    del activations
            layer_lambdas = np.array(layer_lambdas)
            scale_facs[layer.name] = np.average(layer_lambdas, axis=0).tolist()
            print("[✓]  Layer "+str(layer.name))
            i += 1
        del x
        # Write scale factors to disk
        filepath = os.path.join(norm_dir, config.get('normalization',
                                                     'percentile') + '.json')
        from snntoolbox.utils.utils import confirm_overwrite
        if config.get('output', 'overwrite') or confirm_overwrite(filepath):
            with open(filepath, str('w')) as f:
                json.dump(scale_facs, f)

    # Fix RNet heads, they need to be equally weighted
    if equiv_layers:
        for layer in model.layers:
            for ix,group in enumerate(equiv_layers):
                if layer.name in group:
                    refs = equiv_layers.pop(ix)
                    scale_fac = np.zeros(len(scale_facs[refs[0]]))
                    for ref in refs: scale_fac += np.array(scale_facs[ref]) # average
                    scale_fac /= float(len(refs))
                    for ref in refs: scale_facs[ref] = scale_fac.tolist()
                    del refs, scale_fac
        
        filepath = os.path.join(norm_dir, config.get('normalization',
                                                     'percentile') + '_mod.json')
        with open(filepath, str('w')) as f:
            json.dump(scale_facs, f)


    # Apply scale factors to normalize the parameters.
    for layer in model.layers:
        # Skip if layer has no parameters
        if len(layer.weights) == 0:
            continue

        # Scale parameters
        parameters = layer.get_weights()
        scale_fac = np.array(scale_facs[layer.name])
        inbound = get_inbound_layers_with_params(layer)

        if parameters[0].ndim != 4:
            # Fully-connected layers need more consideration, because they
            # could receive input from several conv layers that are
            # concatenated and then flattened. The neuron position in the
            # flattened layer depend on the image_data_format.
            raise NotImplementedError

        in_ch = parameters[0].shape[-2]
        out_ch = parameters[0].shape[-1]

        parameters_norm = parameters

        parameters_norm[1] = parameters[1]/scale_fac
        if len(inbound) == 0:  # Input layer
            parameters_norm[0] /= scale_fac
        elif len(inbound) == 1:
            for i in range(in_ch):
                parameters_norm[0][:, :, i] *= scale_facs[inbound[0].name][i]/scale_fac
        else:
            offset = 0  # Index offset at input filter dimension
            for inb in inbound:
                f_out = inb.filters  # Num output features of inbound layer
                for i in range(f_out):
                    parameters_norm[0][:, :, i+offset] *= scale_facs[inb.name][i]/scale_fac
                offset += f_out
        
        # Check if the layer happens to be Sparse
        # if the layer is sparse, add the mask to the list of parameters
        if len(parameters) == 3:
            parameters_norm.append(parameters[-1])
        # Update model with modified parameters
        layer.set_weights(parameters_norm)

    # Plot distributions of weights and activations before and after norm.
    if 'normalization_activations' in eval(config.get('output', 'plot_vars')) and layers_to_plot:
        from snntoolbox.simulation.plotting import plot_hist
        from snntoolbox.simulation.plotting import plot_max_activ_hist

        # All layers in one plot. Assumes model.get_weights() returns
        # [w, b, w, b, ...].
        # from snntoolbox.simulation.plotting import plot_weight_distribution
        # plot_weight_distribution(norm_dir, model)

        print("Plotting distributions of weights and activations before and "
              "after normalizing...")

        # Load original parsed model to get parameters before normalization
        weights = np.load(os.path.join(activ_dir, 'weights.npz'))

        x=[]
        for sample in norm_set.take(samples_in_division):
            x.append(np.array(sample[0], dtype=np.float32))
        x = np.array(x)[:,0]
                
        for idx, layer in enumerate(model.layers):
            # Skip if layer has no parameters
            if len(layer.weights) == 0 or layer.name not in layers_to_plot:
                continue

            label = str(idx) + layer.__class__.__name__ \
                if config.getboolean('output', 'use_simple_labels') \
                else layer.name
            parameters = weights[layer.name]
            parameters_norm = layer.get_weights()[0]
            weight_dict = {'weights': parameters.flatten(),
                           'weights_norm': parameters_norm.flatten()}
            plot_hist(weight_dict, 'Weight', label, norm_dir)

            # Load activations of model before normalization
            try:
                activations = np.load(os.path.join(activ_dir, layer.name + '.npz'))['arr_0']
            except IOError:
                    print('Error when loading activations from [',
                            os.path.join(activ_dir, layer.name + '.npz'),
                            ']. \n...Skipping layer...')
                    continue

            if activations is None:
                continue

            # Compute activations with modified parameters
            nonzero_activations = activations[np.nonzero(activations)]
            tf.keras.backend.clear_session()
            activations_norm = np.array(Model(model.input, layer.output).predict(x, batch_size))
            activation_dict = {'Activations': nonzero_activations,
                               'Activations_norm':
                               activations_norm[np.nonzero(activations_norm)]}
            scale_fac = scale_facs[layer.name]
            plot_hist(activation_dict, 'Activation', label, norm_dir,
                      scale_fac)
            ax = tuple(np.arange(len(layer.output_shape))[1:])
            plot_max_activ_hist(
                {'Activations_max': np.max(activations, axis=ax)},
                'Maximum Activation', label, norm_dir, scale_fac)
            np.savez_compressed(os.path.join(norm_activ_dir, layer.name), activations_norm)
    print('')



def layer_norm_J(model, config, norm_set=None, num_samples=None, divisions=1, free_GB=2, layers_to_plot=[], equiv_layers=[], **kwargs):

    from snntoolbox.parsing.utils import get_inbound_layers_with_params

    print("Normalizing parameters...")

    norm_dir = kwargs[str('path')] if 'path' in kwargs else \
        os.path.join(config.get('paths', 'path_wd'),
                     'layer_norm_J')

    activ_dir = os.path.join(norm_dir, 'activations')
    norm_activ_dir = os.path.join(norm_dir, 'norm_activations')
    if not os.path.exists(activ_dir):
        os.makedirs(activ_dir)
    # Store original weights for later plotting
    if not os.path.isfile(os.path.join(activ_dir, 'weights.npz')):
        weights = {}
        for layer in model.layers:
            w = layer.get_weights()
            if len(w) > 0:
                weights[layer.name] = w[0]
        np.savez_compressed(os.path.join(activ_dir, 'weights.npz'), **weights)

    batch_size = config.getint('simulation', 'batch_size')

    # Either load scale factors from disk, or get normalization data set to
    # calculate them.
    filepath = os.path.join(norm_dir, config.get('normalization',
                                                'percentile') + '.json')
    samples_in_division = int(num_samples/divisions) 
    if 'scale_facs' in kwargs:
        scale_facs = kwargs[str('scale_facs')]
        print('(*) Scale factors input as **kwargs to the function')
    elif os.path.isfile(filepath):
        with open(filepath) as f:
            scale_facs = json.load(f)
        print('(*) Scale factors loaded from previous run: [', filepath, ']')
    elif norm_set is not None: 
        print('(*) Computing scale factors from the model. This might take a while...')    
        print("Using {} samples for normalization: {} packs of {} samples".format(num_samples,
                                                                            divisions,
                                                                            samples_in_division))
        sizes = [
            samples_in_division * float(np.array(layer.output_shape[1:]).prod()) * float(32 /(8 * 1e9)) \
                 for layer in model.layers if len(layer.weights) > 0 ]
        size_str = ['{:.2f}'.format(s) for s in sizes]
        print('INFO: Size of layer activations: ', size_str ,'GB\n')       
        req_space = max(sizes)
        print("Required {:.2f} GB of free space for the largest activation. \n".format(req_space))
        print("In total, {:.2f} GB of information flow. \n".format(sum(sizes)))
        if req_space > free_GB:
            import warnings
            warnings.warn("Required space is larger than specified free space of "+str(free_GB)+
            "GB. Reduce size of data set or increase available space.", ResourceWarning)
            print('[Skipping normalization]')
            return
        scale_facs = OrderedDict({model.layers[0].name: 1})
    else:
        import warnings
        warnings.warn("Scale factors or normalization data set could not be "
                      "loaded. Proceeding without normalization.",
                      RuntimeWarning)
        return

    # If scale factors have not been computed in a previous run, do so now.
    if len(scale_facs) == 1:
        i = 0
        for layer in model.layers:
            # Skip if layer has no parameters
            if len(layer.weights) == 0:
                continue
            perc = get_percentile(config, i)
            accum_samples=0
            layer_lambdas = []
            layer_shifts = []
            x = []
            for sample in norm_set.take(num_samples):
                x.append(np.array(sample[0], dtype=np.float32))
                accum_samples+=1
                if accum_samples>=samples_in_division:
                    accum_samples = 0
                    x = np.array(x)[:,0]
                    tf.keras.backend.clear_session()
                    activations = np.array(Model(model.input, layer.output).predict(x, batch_size))
                    nonzero_activations = activations[np.nonzero(activations)]
                    x = []
                    if layer.name in layers_to_plot and not layer_lambdas:    
                        print("Writing activations to disk...")
                        np.savez_compressed(os.path.join(activ_dir, layer.name), activations)
                    del activations
                    layer_lambdas.append(get_scale_fac(nonzero_activations, perc))
                    layer_shifts.append(get_shift(nonzero_activations))
                    del nonzero_activations
            layer_lambdas = np.array(layer_lambdas, dtype='float64')
            layer_shifts = np.array(layer_shifts, dtype='float64')
            scale_facs[layer.name] = [np.average(layer_lambdas), np.amin(layer_shifts)]
            print("Layer "+str(layer.name)+" - lmbda: {:.2f} ± {:.2f},  shift: {:.2f} ± {:.2f}.".format(
                                                scale_facs[layer.name][0],
                                                max(abs(scale_facs[layer.name][0]-layer_lambdas)),
                                                scale_facs[layer.name][1],
                                                max(abs(scale_facs[layer.name][1]-layer_shifts))
                                                ))
            i += 1
        del x
        # Write scale factors to disk
        filepath = os.path.join(norm_dir, config.get('normalization',
                                                     'percentile') + '.json')
        from snntoolbox.utils.utils import confirm_overwrite
        if config.get('output', 'overwrite') or confirm_overwrite(filepath):
            with open(filepath, str('w')) as f:
                json.dump(scale_facs, f)

    # Fix RNet heads, they need to be equally weighted
    if equiv_layers:
        for layer in model.layers:
            for ix,group in enumerate(equiv_layers):
                if layer.name in group:
                    refs = equiv_layers.pop(ix)
                    lmbda = 0
                    shift = 1e100
                    for ref in refs: 
                        lmbda += scale_facs[ref][0]
                        shift = min(shift,scale_facs[ref][1]) 
                    lmbda /= len(refs)
                    for ref in refs: scale_facs[ref] = [lmbda,shift]
                    del refs, lmbda, shift
        
        filepath = os.path.join(norm_dir, config.get('normalization',
                                                     'percentile') + '_mod.json')
        with open(filepath, str('w')) as f:
            json.dump(scale_facs, f)


    # Apply scale factors to normalize the parameters.
    for layer in model.layers:
        
        if layer.__class__.__name__ == 'NormReshape':
            inbound = get_inbound_layers_with_params(layer)
            weights = layer.get_weights()
            layer.set_weights([
                weights[0]*scale_facs[inbound[0].name][0],
                weights[1]+scale_facs[inbound[0].name][1]
            ])
            continue


        # Skip if layer has no parameters
        if len(layer.weights) == 0:
            continue

        # Scale parameters
        parameters = [np.array(w, dtype='float64') for w in layer.get_weights()]
        lmbda = np.array(scale_facs[layer.name][0], dtype='float64')
        shift = np.array(scale_facs[layer.name][1], dtype='float64')
        inbound = get_inbound_layers_with_params(layer)
        if len(inbound) == 0:  # Input layer
            parameters_norm = [
                parameters[0]/(lmbda-shift),
                (parameters[1]-shift)/(lmbda-shift)
                ]
        elif len(inbound) == 1:
            ws = np.sum(parameters[0],(0,1,2))*scale_facs[inbound[0].name][1]
            parameters_norm = [
                parameters[0] * (scale_facs[inbound[0].name][0]-scale_facs[inbound[0].name][1])/(lmbda-shift),
                (parameters[1]+ws-shift)/(lmbda-shift)
                ]
        else:
            # In case of this layer receiving input from several layers, we can
            # apply scale factor to bias as usual, but need to rescale weights
            # according to their respective input.
            parameters_norm = parameters
            # In conv layers, just need to split up along channel dim.
            offset = 0  # Index offset at input filter dimension
            ws = np.zeros(parameters[1].shape)
            for i,inb in enumerate(inbound):
                f_out = inb.filters  # Num output features of inbound layer
                f_in = range(offset, offset + f_out)
                parameters_norm[0][:,:,f_in] *= (scale_facs[inb.name][0]-scale_facs[inb.name][1])/(lmbda-shift)
                parameters_norm[i+2] = parameters_norm[i+2] + scale_facs[inb.name][1]/(scale_facs[inb.name][0]-scale_facs[inb.name][1])
                offset += f_out
            parameters_norm[1] = (parameters[1]-shift)/(lmbda-shift)

        # Check if the layer happens to be Sparse
        # if the layer is sparse, add the mask to the list of parameters
        if len(parameters) == 3:
            parameters_norm.append(parameters[-1])
        # Update model with modified parameters
        layer.set_weights(parameters_norm)

    # Plot distributions of weights and activations before and after norm.
    if 'normalization_activations' in eval(config.get('output', 'plot_vars')) and layers_to_plot:
        from snntoolbox.simulation.plotting import plot_hist
        from snntoolbox.simulation.plotting import plot_max_activ_hist

        # All layers in one plot. Assumes model.get_weights() returns
        # [w, b, w, b, ...].
        # from snntoolbox.simulation.plotting import plot_weight_distribution
        # plot_weight_distribution(norm_dir, model)

        print("Plotting distributions of weights and activations before and "
              "after normalizing...")

        # Load original parsed model to get parameters before normalization
        weights = np.load(os.path.join(activ_dir, 'weights.npz'))

        x=[]
        for sample in norm_set.take(samples_in_division):
            x.append(np.array(sample[0], dtype=np.float32))
        x = np.array(x)[:,0]
                
        for idx, layer in enumerate(model.layers):
            # Skip if layer has no parameters
            if len(layer.weights) == 0 or layer.name not in layers_to_plot:
                continue

            label = str(idx) + layer.__class__.__name__ \
                if config.getboolean('output', 'use_simple_labels') \
                else layer.name
            parameters = weights[layer.name]
            parameters_norm = layer.get_weights()[0]
            weight_dict = {'weights': parameters.flatten(),
                           'weights_norm': parameters_norm.flatten()}
            plot_hist(weight_dict, 'Weight', label, norm_dir)

            # Load activations of model before normalization
            try:
                activations = np.load(os.path.join(activ_dir, layer.name + '.npz'))['arr_0']
            except IOError:
                    print('Error when loading activations from [',
                            os.path.join(activ_dir, layer.name + '.npz'),
                            ']. \n...Skipping layer...')
                    continue

            if activations is None:
                continue

            # Compute activations with modified parameters
            nonzero_activations = activations[np.nonzero(activations)]
            tf.keras.backend.clear_session()
            activations_norm = np.array(Model(model.input, layer.output).predict(x, batch_size))
            activation_dict = {'Activations': nonzero_activations,
                               'Activations_norm':
                               activations_norm[np.nonzero(activations_norm)]}
            scale_fac = scale_facs[layer.name]
            plot_hist(activation_dict, 'Activation', label, norm_dir,
                      scale_fac)
            ax = tuple(np.arange(len(layer.output_shape))[1:])
            plot_max_activ_hist(
                {'Activations_max': np.max(activations, axis=ax)},
                'Maximum Activation', label, norm_dir, scale_fac)
            np.savez_compressed(os.path.join(norm_activ_dir, layer.name), activations_norm)
    print('')



def channel_norm_J(model, config, norm_set=None, divisions=1, free_GB=2, layers_to_plot=[], 
        equiv_layers=[], perform_layer_norm_to_these=[], trim_these = None, scaling_coef=1, **kwargs):

    from snntoolbox.parsing.utils import get_inbound_layers_with_params

    print("Normalizing parameters with channel norm...")

    norm_dir = kwargs[str('path')] if 'path' in kwargs else \
        os.path.join(config.get('paths', 'path_wd'),
                     'channel_norm_J')

    activ_dir = os.path.join(norm_dir, 'activations')
    norm_activ_dir = os.path.join(norm_dir, 'norm_activations')
    if not os.path.exists(activ_dir):
        os.makedirs(activ_dir)
    # Store original weights for later plotting
    if not os.path.isfile(os.path.join(activ_dir, 'weights.npz')):
        weights = {}
        for layer in model.layers:
            w = layer.get_weights()
            if len(w) > 0:
                weights[layer.name] = w[0]
        np.savez_compressed(os.path.join(activ_dir, 'weights.npz'), **weights)

    num_samples = config.getint('normalization', 'num_to_norm')
    batch_size = config.getint('simulation', 'batch_size')
    norm_method = config.getint('normalization', 'method', fallback=0)

    # Either load scale factors from disk, or get normalization data set to
    # calculate them.
    filepath = os.path.join(norm_dir, config.get('normalization',
                                                'percentile') + '.json')
    samples_in_division = int(num_samples/divisions) 
    if 'scale_facs' in kwargs:
        scale_facs = kwargs[str('scale_facs')]
        print('(*) Scale factors input as **kwargs to the function')
    elif os.path.isfile(filepath):
        with open(filepath) as f:
            scale_facs = json.load(f)
        print('(*) Scale factors loaded from previous run: [', filepath, ']')
    elif norm_set is not None: 
        print('(*) Computing scale factors from the model. This might take a while...')
        if norm_method == 1:
            print("Using {} samples for normalization: METHOD 1".format(num_samples))
            req_space = 0
            for layer in model.layers: 
                if len(layer.weights) > 0:
                    req_space += (float(np.array(layer.output_shape[1:]).prod()) + 2.0) * float(2*32 /(8 * 1e9))
            print("Required {:.2f} GB of free space for each sample iteration. \n".format(req_space))
        else:    
            print("Using {} samples for normalization: {} packs of {} samples".format(num_samples,
                                                                                divisions,
                                                                                samples_in_division))
            sizes = [
                samples_in_division * float(np.array(layer.output_shape[1:]).prod()) * float(32 /(8 * 1e9)) \
                    for layer in model.layers if len(layer.weights) > 0 ]
            size_str = ['{:.2f}'.format(s) for s in sizes]
            print('INFO: Size of layer activations: ', size_str ,'GB\n')       
            req_space = max(sizes)
            print("Required {:.2f} GB of free space for the largest activation. \n".format(req_space))
            print("In total, {:.2f} GB of information flow. \n".format(sum(sizes)))
        if req_space > free_GB:
            import warnings
            warnings.warn("Required space is larger than specified free space of "+str(free_GB)+
            "GB. Reduce sizes of data set or increase available space.", ResourceWarning)
            print('[Skipping normalization]')
            return
        scale_facs = OrderedDict({model.layers[0].name: 1})
    else:
        import warnings
        warnings.warn("Scale factors or normalization data set could not be "
                      "loaded. Proceeding without normalization.",
                      RuntimeWarning)
        return

    # If scale factors have not been computed in a previous run, do so now.
    if len(scale_facs) == 1:
        if norm_method == 1:
            save_interval = config.getint('normalization','save_interval', fallback=None)
            inps = model.input                                                           # input placeholder
            outs = [layer.output for layer in model.layers if len(layer.weights)>0]  # all layer outputs
            lnames = [layer.name for layer in model.layers if len(layer.weights)>0]   
            norm_model = Model(inputs = inps, outputs = outs) 

            n = np.array([o.shape[-1] for o in outs])
            lbdas = [np.ones(o)*-1000 for o in n]
            shifts = [np.ones(o)*1000 for o in n]
            save_counter = 0
            i=0
            for sample in tqdm(norm_set.take(num_samples)):
                activations = norm_model.predict(sample)
                for j,act in enumerate(activations):
                    perc = get_percentile(config, j)
                    new_scale_facts = np.percentile(act, [100-perc, perc], axis=(0,1,2))
                    shifts[j] = np.fmin(shifts[j], new_scale_facts[0])
                    lbdas[j] = np.fmax(lbdas[j], new_scale_facts[1])
                save_counter+=1
                if save_counter >= save_interval:
                    print('[OK] Saving scale_facts at sample ', i, '...')
                    save_counter=0
                    for j,nme in enumerate(lnames):
                        scale_facs[nme] = [lbdas[j].tolist(), shifts[j].tolist()] 
                    filepath = os.path.join(norm_dir, config.get('normalization','percentile') + 's'+str(i)+'.json')
                    with open(filepath, str('w')) as f:
                        json.dump(scale_facs, f)
                i+=1
            for j,nme in enumerate(lnames):
                scale_facs[nme] = [lbdas[j].tolist(), shifts[j].tolist()]    
        else:
            i = 0
            for layer in model.layers:
                # Skip if layer has no parameters
                if len(layer.weights) == 0:
                    continue
                perc = get_percentile(config, i)
                accum_samples=0
                layer_lambdas = []
                layer_shifts = []
                x = []
                for sample in norm_set.take(num_samples):
                    x.append(np.array(sample[0], dtype=np.float32))
                    accum_samples+=1
                    if accum_samples>=samples_in_division:
                        accum_samples = 0
                        x = np.array(x)[:,0]
                        tf.keras.backend.clear_session()
                        activations = np.array(Model(model.input, layer.output).predict(x, batch_size))
                        nonzero_activations = activations[np.nonzero(activations)]
                        x = []
                        if layer.name in layers_to_plot and not layer_lambdas:    
                            print("Writing "+layer.name+"'s activations to disk...")
                            np.savez_compressed(os.path.join(activ_dir, layer.name), activations)
                        layer_lambdas.append(get_scale_fac_channel(activations, perc))
                        layer_shifts.append(get_shift_channel(activations))
                        del activations
                layer_lambdas = np.array(layer_lambdas)
                layer_shifts = np.array(layer_shifts)
                scale_facs[layer.name] = [
                    np.average(layer_lambdas, axis=0).tolist(),
                    np.average(layer_shifts, axis=0).tolist()
                ]
                print("[OK]  Layer "+str(layer.name))
                i += 1
            del x
        # Write scale factors to disk
        filepath = os.path.join(norm_dir, config.get('normalization',
                                                    'percentile') + '.json')
        from snntoolbox.utils.utils import confirm_overwrite
        if config.get('output', 'overwrite') or confirm_overwrite(filepath):
            with open(filepath, str('w')) as f:
                json.dump(scale_facs, f)

    # MODIFICATIONS TO OBTAINED WEIGHTS
    # 1 Fix RNet heads, they need to be equally weighted
    if equiv_layers:
        for ix,group in enumerate(equiv_layers):
            refs = equiv_layers.pop(ix)
            n_channels = len(scale_facs[refs[0]][0])
            lbdas = np.zeros(n_channels)
            shifts = np.zeros(n_channels)
            for ref in refs: # average and min
                lbdas  = np.fmax(lbdas,  np.array(scale_facs[ref][0])) 
                shifts = np.fmin(shifts, np.array(scale_facs[ref][1])) 
            for ref in refs: scale_facs[ref] = [lbdas.tolist(), shifts.tolist()]
            del refs, lbdas, shifts

    # 2 Perform layer_norm to some layers
    if perform_layer_norm_to_these:
        if perform_layer_norm_to_these in ['all', 'All']:
            perform_layer_norm_to_these = [lr.name for lr in model.layers if len(lr.weights) != 0]
        for lr in perform_layer_norm_to_these:
            lbdas = np.array(scale_facs[lr][0])
            shifts = np.array(scale_facs[lr][1])
            lbdas = np.array([np.amax(lbdas)]*len(lbdas))
            shifts = np.array([np.amin(shifts)]*len(shifts))
            scale_facs[lr] = [lbdas.tolist(), shifts.tolist()]
            del lbdas, shifts

    # 3 Manually scale all scale facts
    for layer in model.layers: 
        if len(layer.weights) != 0:
            scale_facs[layer.name] = [
                (np.array(scale_facs[layer.name][0])*scaling_coef).tolist(),
                (np.array(scale_facs[layer.name][1])*scaling_coef).tolist()
            ]
    
    # 4 Manually trim some scale facts
    if trim_these is not None:
        for layer_name in trim_these['layers']:
            scale_facs[layer_name] = [
                np.fmin(np.array(scale_facs[layer_name][0]), trim_these['limits'][1]).tolist(),
                np.fmax(np.array(scale_facs[layer_name][1]), trim_these['limits'][0]).tolist()
            ]

        
    filepath = os.path.join(norm_dir, config.get('normalization',
                                                    'percentile') + '_mod.json')
    with open(filepath, str('w')) as f:
        json.dump(scale_facs, f)


    # Apply scale factors to normalize the parameters.
    for layer in model.layers:
        
        if layer.__class__.__name__ == 'NormReshape':
            continue

        # Skip if layer has no parameters
        if len(layer.weights) == 0:
            continue

        # Scale parameters
        parameters = [np.array(tf.convert_to_tensor(w), dtype='float64') for w in layer.get_weights()]
        lbdas = np.array(scale_facs[layer.name][0], dtype='float64')
        shifts = np.array(scale_facs[layer.name][1], dtype='float64')
        denom = lbdas-shifts
        denom = np.array([aux if aux!=0 else 1 for aux in [l-s for l,s in zip(lbdas,shifts)]], dtype='float64')
        inbound = get_inbound_layers_with_params(layer)

        in_ch = parameters[0].shape[-2]
        out_ch = parameters[0].shape[-1]

        if len(inbound) == 0:  # Input layer
            parameters_norm = [
                parameters[0]/denom,
                (parameters[1]-shifts)/denom
                ]
        elif len(inbound) == 1:
            lbdas0  = np.array(scale_facs[inbound[0].name][0], dtype='float64')
            shifts0 = np.array(scale_facs[inbound[0].name][1], dtype='float64')
            denom0 = lbdas0-shifts0
            denom0 = np.array([aux if aux!=0 else 1 for aux in [l-s for l,s in zip(lbdas0,shifts0)]], dtype='float64')
            parameters_norm = parameters.copy()

            if layer.__class__.__name__ == 'NormConv2D':
                parameters_norm[0] = parameters[0]/denom
                parameters_norm[1] = (parameters[1]-shifts)/denom
                parameters_norm[2] = denom0
                parameters_norm[3] = shifts0
            else:
                for i in range(len(lbdas0)):
                    den0 = (lbdas0[i]-shifts0[i])
                    den0 = den0 if den0!=0 else 1
                    parameters_norm[1] += np.sum(parameters[0][:,:,i],(0,1))*shifts0[i]
                    parameters_norm[0][:,:,i] = parameters[0][:,:,i]*den0/denom
                parameters_norm[1] = (parameters_norm[1]-shifts)/denom

        else:
            parameters_norm = parameters.copy()
            offset = 0  # Index offset at input filter dimension
            for i,inb in enumerate(inbound):
                lbdas0  = np.array(scale_facs[inb.name][0], dtype='float64')
                shifts0 = np.array(scale_facs[inb.name][1], dtype='float64')
                f_out = len(lbdas0)  # Num output features of inbound layer
                unshift = np.zeros(f_out)
                for j in range(f_out):
                    den0 = (lbdas0[j]-shifts0[j])
                    den0 = den0 if den0!=0 else 1
                    parameters_norm[0][:,:,j+offset] *= den0/denom
                    unshift[j] = parameters_norm[i+2][j] + shifts0[j]/den0
                parameters_norm[i+2] = unshift
                offset += f_out
            parameters_norm[1] = (parameters[1]-shifts)/denom

        # Check if the layer happens to be Sparse
        # if the layer is sparse, add the mask to the list of parameters
        if len(parameters) == 3:
            parameters_norm.append(parameters[-1])
        # Update model with modified parameters
        layer.set_weights(parameters_norm)

    print('')


def normalize_parameters(model, config, free_GB=2, **kwargs):
    """Normalize the parameters of a network.

    The parameters of each layer are normalized with respect to the maximum
    activation, or the ``n``-th percentile of activations.

    Generates plots of the activity- and weight-distribution before and after
    normalization. Note that plotting the activity-distribution can be very
    time- and memory-consuming for larger networks.
    """

    from snntoolbox.parsing.utils import get_inbound_layers_with_params

    print("Normalizing parameters...")

    norm_dir = kwargs[str('path')] if 'path' in kwargs else \
        os.path.join(config.get('paths', 'log_dir_of_current_run'),
                     'normalization')

    activ_dir = os.path.join(norm_dir, 'activations')
    if not os.path.exists(activ_dir):
        os.makedirs(activ_dir)
    # Store original weights for later plotting
    if not os.path.isfile(os.path.join(activ_dir, 'weights.npz')):
        weights = {}
        for layer in model.layers:
            w = layer.get_weights()
            if len(w) > 0:
                weights[layer.name] = w[0]
        np.savez_compressed(os.path.join(activ_dir, 'weights.npz'), **weights)

    batch_size = config.getint('simulation', 'batch_size')

    # Either load scale factors from disk, or get normalization data set to
    # calculate them.
    filepath = os.path.join(norm_dir, config.get('normalization',
                                                'percentile') + '.json')
    x_norm = None
    if 'scale_facs' in kwargs:
        scale_facs = kwargs[str('scale_facs')]
    elif 'x_norm' in kwargs or 'dataflow' in kwargs:
        if 'x_norm' in kwargs:
            x_norm = kwargs[str('x_norm')]
        elif 'dataflow' in kwargs:
            x_norm = []
            dataflow = kwargs[str('dataflow')]
            num_samples_norm = config.getint('normalization', 'num_samples',
                                             fallback='')
            if num_samples_norm == '':
                num_samples_norm = len(dataflow) * dataflow.batch_size
            while len(x_norm) * batch_size < num_samples_norm:
                x = dataflow.next()
                if isinstance(x, tuple):  # Remove class label if present.
                    x = x[0]
                x_norm.append(x)
            x_norm = np.concatenate(x_norm)
        print("Using {} samples for normalization.".format(len(x_norm)))
        sizes = [
            len(x_norm) * float(np.array(layer.output_shape[1:]).prod()) * float(32 /(8 * 1e9)) \
                 for layer in model.layers if len(layer.weights) > 0 ]
        size_str = ['{:.2f}'.format(s) for s in sizes]
        print('INFO: Size of layer activations: ', size_str ,'GB\n')       
        req_space = max(sizes)
        print("Required {:.2f} GB of free space for the largest activation. \n".format(req_space))
        print("In total, {:.2f} GB of information flow. \n".format(sum(sizes)))
        if req_space > free_GB:
            import warnings
            warnings.warn("Required space is larger than specified free space of "+str(free_GB)+
            "GB. Reduce size of data set or increase available space.", ResourceWarning)
            print('[Skipping normalization]')
            return
        scale_facs = OrderedDict({model.layers[0].name: 1})
    else:
        import warnings
        warnings.warn("Scale factors or normalization data set could not be "
                      "loaded. Proceeding without normalization.",
                      RuntimeWarning)
        return

    # If scale factors have not been computed in a previous run, do so now.
    if len(scale_facs) == 1:
        i = 0
        sparsity = []
        for layer in model.layers:
            # Skip if layer has no parameters
            if len(layer.weights) == 0:
                continue

            activations = try_reload_activations(layer, model, x_norm,
                                                 batch_size, activ_dir)
            nonzero_activations = activations[np.nonzero(activations)]
            sparsity.append(1 - nonzero_activations.size / activations.size)
            del activations
            perc = get_percentile(config, i)
            scale_facs[layer.name] = get_scale_fac(nonzero_activations, perc)
            print("Scale factor: {:.2f}.".format(scale_facs[layer.name]))
            # Since we have calculated output activations here, check at this
            # point if the output is mostly negative, in which case we should
            # stick to softmax. Otherwise ReLU is preferred.
            # Todo: Determine the input to the activation by replacing the
            # combined output layer by two distinct layers ``Dense`` and
            # ``Activation``!
            # if layer.activation == 'softmax' and settings['softmax_to_relu']:
            #     softmax_inputs = ...
            #     if np.median(softmax_inputs) < 0:
            #         print("WARNING: You allowed the toolbox to replace "
            #               "softmax by ReLU activations. However, more than "
            #               "half of the activations are negative, which "
            #               "could reduce accuracy. Consider setting "
            #               "settings['softmax_to_relu'] = False.")
            #         settings['softmax_to_relu'] = False
            i += 1
        # Write scale factors to disk
        from snntoolbox.utils.utils import confirm_overwrite
        if config.get('output', 'overwrite') or confirm_overwrite(filepath):
            with open(filepath, str('w')) as f:
                json.dump(scale_facs, f)
        np.savez_compressed(os.path.join(norm_dir, 'activations', 'sparsity'),
                            sparsity=sparsity)

    # Apply scale factors to normalize the parameters.
    for layer in model.layers:
        # Skip if layer has no parameters
        if len(layer.weights) == 0:
            continue

        # Scale parameters
        parameters = layer.get_weights()
        if layer.activation.__name__ == 'softmax':
            # When using a certain percentile or even the max, the scaling
            # factor can be extremely low in case of many output classes
            # (e.g. 0.01 for ImageNet). This amplifies weights and biases
            # greatly. But large biases cause large offsets in the beginning
            # of the simulation (spike input absent).
            scale_fac = 1.0
            print("Using scale factor {:.2f} for softmax layer.".format(
                scale_fac))
        else:
            scale_fac = scale_facs[layer.name]
        inbound = get_inbound_layers_with_params(layer)
        if len(inbound) == 0:  # Input layer
            parameters_norm = [
                parameters[0] * scale_facs[model.layers[0].name] / scale_fac,
                parameters[1] / scale_fac]
        elif len(inbound) == 1:
            parameters_norm = [
                parameters[0] * scale_facs[inbound[0].name] / scale_fac,
                parameters[1] / scale_fac]
        else:
            # In case of this layer receiving input from several layers, we can
            # apply scale factor to bias as usual, but need to rescale weights
            # according to their respective input.
            parameters_norm = [parameters[0], parameters[1] / scale_fac]
            if parameters[0].ndim == 4:
                # In conv layers, just need to split up along channel dim.
                offset = 0  # Index offset at input filter dimension
                for inb in inbound:
                    f_out = inb.filters  # Num output features of inbound layer
                    f_in = range(offset, offset + f_out)
                    parameters_norm[0][:, :, f_in, :] *= \
                        scale_facs[inb.name] / scale_fac
                    offset += f_out
            else:
                # Fully-connected layers need more consideration, because they
                # could receive input from several conv layers that are
                # concatenated and then flattened. The neuron position in the
                # flattened layer depend on the image_data_format.
                raise NotImplementedError

        # Check if the layer happens to be Sparse
        # if the layer is sparse, add the mask to the list of parameters
        if len(parameters) == 3:
            parameters_norm.append(parameters[-1])
        # Update model with modified parameters
        layer.set_weights(parameters_norm)

    # Plot distributions of weights and activations before and after norm.
    if 'normalization_activations' in eval(config.get('output', 'plot_vars')):
        from snntoolbox.simulation.plotting import plot_hist
        from snntoolbox.simulation.plotting import plot_max_activ_hist

        # All layers in one plot. Assumes model.get_weights() returns
        # [w, b, w, b, ...].
        # from snntoolbox.simulation.plotting import plot_weight_distribution
        # plot_weight_distribution(norm_dir, model)

        print("Plotting distributions of weights and activations before and "
              "after normalizing...")

        # Load original parsed model to get parameters before normalization
        weights = np.load(os.path.join(activ_dir, 'weights.npz'))
        for idx, layer in enumerate(model.layers):
            # Skip if layer has no parameters
            if len(layer.weights) == 0:
                continue

            label = str(idx) + layer.__class__.__name__ \
                if config.getboolean('output', 'use_simple_labels') \
                else layer.name
            parameters = weights[layer.name]
            parameters_norm = layer.get_weights()[0]
            weight_dict = {'weights': parameters.flatten(),
                           'weights_norm': parameters_norm.flatten()}
            plot_hist(weight_dict, 'Weight', label, norm_dir)

            # Load activations of model before normalization
            activations = try_reload_activations(layer, model, x_norm,
                                                 batch_size, activ_dir)

            if activations is None or x_norm is None:
                continue

            # Compute activations with modified parameters
            nonzero_activations = activations[np.nonzero(activations)]
            activations_norm = get_activations_layer(model.input, layer.output,
                                                     x_norm, batch_size)
            activation_dict = {'Activations': nonzero_activations,
                               'Activations_norm':
                               activations_norm[np.nonzero(activations_norm)]}
            scale_fac = scale_facs[layer.name]
            plot_hist(activation_dict, 'Activation', label, norm_dir,
                      scale_fac)
            ax = tuple(np.arange(len(layer.output_shape))[1:])
            plot_max_activ_hist(
                {'Activations_max': np.max(activations, axis=ax)},
                'Maximum Activation', label, norm_dir, scale_fac)
    print('')


def get_scale_fac(activations, percentile):
    """
    Determine the activation value at ``percentile`` of the layer distribution.

    Parameters
    ----------

    activations: np.array
        The activations of cells in a specific layer, flattened to 1-d.

    percentile: int
        Percentile at which to determine activation.

    Returns
    -------

    scale_fac: float
        Maximum (or percentile) of activations in this layer.
        Parameters of the respective layer are scaled by this value.
    """

    return np.percentile(activations, percentile) if activations.size else 1


def get_shift(activations):
    return np.amin(activations)


def get_scale_fac_channel(activations, percentile):
    """
    Determine the activation value at ``percentile`` of the layer distribution per channel.

    Parameters
    ----------

    activations: np.array
        The activations of cells in a specific layer.

    percentile: int
        Percentile at which to determine activation.

    Returns
    -------

    scale_fac: float
        Maximum (or percentile) of activations in each channel of this layer.
        Parameters of the respective layer are scaled by this value.
    """

    return np.percentile(activations, percentile, axis=(0,1,2))

def get_shift_channel(activations):

    return np.minimum(0, np.amin(activations, axis=(0,1,2)))


def get_percentile(config, layer_idx=None):
    """Get percentile at which to draw the maximum activation of a layer.

    Parameters
    ----------

    config: configparser.ConfigParser
        Settings.

    layer_idx: Optional[int]
        Layer index.

    Returns
    -------

    : int
        Percentile.

    """

    perc = config.getfloat('normalization', 'percentile')

    if config.getboolean('normalization', 'normalization_schedule'):
        assert layer_idx >= 0, "Layer index needed for normalization schedule."
        perc = apply_normalization_schedule(perc, layer_idx)

    return perc


def apply_normalization_schedule(perc, layer_idx):
    """Transform percentile according to some rule, depending on layer index.

    Parameters
    ----------

    perc: float
        Original percentile.

    layer_idx: int
        Layer index, used to decrease the scale factor in higher layers, to
        maintain high spike rates.

    Returns
    -------

    : int
        Modified percentile.

    """

    return int(perc - layer_idx * 0.02)


def get_activations_layer(layer_in, layer_out, x, batch_size=None):
    """
    Get activations of a specific layer, iterating batch-wise over the complete
    data set.

    Parameters
    ----------

    layer_in: keras.layers.Layer
        The input to the network.

    layer_out: keras.layers.Layer
        The layer for which we want to get the activations.

    x: np.array
        The samples to compute activations for. With data of the form
        (channels, num_rows, num_cols), x_train has dimension
        (batch_size, channels*num_rows*num_cols) for a multi-layer perceptron,
        and (batch_size, channels, num_rows, num_cols) for a convolutional net.

    batch_size: Optional[int]
        Batch size

    Returns
    -------

    activations: ndarray
        The activations of cells in a specific layer. Has the same shape as
        ``layer_out``.
    """
    
    if batch_size is None:
        batch_size = 10

    if len(x) % batch_size != 0:
        x = x[: -(len(x) % batch_size)]

    return Model(layer_in, layer_out).predict(x, batch_size)


def get_activations_batch(ann, x_batch):
    """Compute layer activations of an ANN.

    Parameters
    ----------

    ann: keras.models.Model
        Needed to compute activations.

    x_batch: np.array
        The input samples to use for determining the layer activations. With
        data of the form (channels, num_rows, num_cols), X has dimension
        (batch_size, channels*num_rows*num_cols) for a multi-layer perceptron,
        and (batch_size, channels, num_rows, num_cols) for a convolutional net.

    Returns
    -------

    activations_batch: list[tuple[np.array, str]]
        Each tuple ``(activations, label)`` represents a layer in the ANN for
        which an activation can be calculated (e.g. ``Dense``,
        ``Conv2D``).
        ``activations`` containing the activations of a layer. It has the same
        shape as the original layer, e.g.
        (batch_size, n_features, n_rows, n_cols) for a convolution layer.
        ``label`` is a string specifying the layer type, e.g. ``'Dense'``.
    """

    activations_batch = []
    for layer in ann.layers:
        # Todo: This list should be replaced by
        #       ``not in eval(config.get('restrictions', 'spiking_layers')``
        if layer.__class__.__name__ in ['Input', 'InputLayer', 'Flatten',
                                        'Concatenate', 'ZeroPadding2D',
                                        'Reshape']:
            continue
        activations = Model(ann.input, layer.output).predict_on_batch(x_batch)
        activations_batch.append((activations, layer.name))
    return activations_batch


def try_reload_activations(layer, model, x_norm, batch_size, activ_dir):
    try:
        activations = np.load(os.path.join(activ_dir,
                                           layer.name + '.npz'))['arr_0']
    except IOError:
        if x_norm is None:
            return

        print("Calculating activations of layer {} ...".format(layer.name))
        activations = get_activations_layer(model.input, layer.output, x_norm,
                                            batch_size)
        print("Writing activations to disk...")
        np.savez_compressed(os.path.join(activ_dir, layer.name), activations)
    else:
        print("Loading activations stored during a previous run.")
    return np.array(activations)
