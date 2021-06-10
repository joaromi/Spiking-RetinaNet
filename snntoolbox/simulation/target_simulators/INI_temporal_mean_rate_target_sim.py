# -*- coding: utf-8 -*-
"""INI simulator with temporal mean rate code.

@author: rbodo
"""

import os
import sys

from tensorflow import keras
import numpy as np

from snntoolbox.parsing.utils import get_inbound_layers_with_params
from snntoolbox.simulation.utils import AbstractSNN, remove_name_counter
from tqdm.auto import tqdm
import tensorflow as tf

remove_classifier = False


class SNN(AbstractSNN):
    """
    The compiled spiking neural network, using layers derived from
    Keras base classes (see
    `snntoolbox.simulation.backends.inisim.temporal_mean_rate_tensorflow`).

    Aims at simulating the network on a self-implemented Integrate-and-Fire
    simulator using a timestepped approach.

    Attributes
    ----------

    snn: keras.models.Model
        Keras model. This is the output format of the compiled spiking model
        because INI simulator runs networks of layers that are derived from
        Keras layer base classes.
    """

    def __init__(self, config, queue=None):

        AbstractSNN.__init__(self, config, queue)

        self.snn = None
        self._spiking_layers = {}
        self._input_images = None
        self._binary_activation = None

    @property
    def is_parallelizable(self):
        return True

    def add_input_layer(self, input_shape):
        self._input_images = keras.layers.Input(batch_shape=input_shape)
        self._spiking_layers[self.parsed_model.layers[0].name] = \
            self._input_images

    def add_layer(self, layer):
        from snntoolbox.parsing.utils import get_type
        spike_layer_name = getattr(self.sim, 'Spike' + get_type(layer))
        # noinspection PyProtectedMember
        inbound = layer._inbound_nodes[0].inbound_layers
        if not isinstance(inbound, (list, tuple)):
            inbound = [inbound]
        inbound = [self._spiking_layers[inb.name] for inb in inbound]
        if len(inbound) == 1:
            inbound = inbound[0]
        layer_kwargs = layer.get_config()
        layer_kwargs['config'] = self.config

        # Check if layer uses binary activations. In that case, we will want to
        # tell the following to MaxPool layer because then we can use a
        # cheaper operation.
        if 'Conv' in layer.name and 'binary' in layer.activation.__name__:
            self._binary_activation = layer.activation.__name__

        if 'MaxPool' in layer.name and self._binary_activation is not None:
            layer_kwargs['activation'] = self._binary_activation
            self._binary_activation = None

        # Replace activation from kwargs by 'linear' before initializing
        # superclass, because the relu activation is applied by the spike-
        # generation mechanism automatically. In some cases (quantized
        # activation), we need to apply the activation manually. This
        # information is taken from the 'activation' key during conversion.
        activation_str = str(layer_kwargs.pop(str('activation'), None))

        spike_layer = spike_layer_name(**layer_kwargs)
        spike_layer.activation_str = activation_str
        spike_layer.is_first_spiking = \
            len(get_inbound_layers_with_params(layer)) == 0
        self._spiking_layers[layer.name] = spike_layer(inbound)

    def build_dense(self, layer):
        pass

    def build_convolution(self, layer):
        pass

    def build_NormAdd(self,layer):
        pass

    def build_pooling(self, layer):
        pass

    def compile_RNet(self, loss_fn, optimizer):

        self.snn = keras.models.Model(
            self._input_images,
            self._spiking_layers[self.parsed_model.layers[-1].name])
        self.snn.compile(loss=loss_fn, optimizer=optimizer)

        # Tensorflow 2 lists all variables as weights, including our state
        # variables (membrane potential etc). So a simple
        # snn.set_weights(parsed_model.get_weights()) does not work any more.
        # Need to extract the actual weights here.

        parameter_map = {remove_name_counter(p.name): v for p, v in
                         zip(self.parsed_model.weights,
                             self.parsed_model.get_weights()) if p.name[0].isnumeric()}
        
        # parameter_map_SNN = {remove_name_counter(p.name): v for p, v in
        #                  zip(self.snn.weights,
        #                      self.snn.get_weights()) if p.name[0].isnumeric()}
        count = 0
        for p in tqdm(self.snn.weights):
            name = remove_name_counter(p.name)
            if name in parameter_map:
                keras.backend.set_value(p, parameter_map[name])
                count += 1
        assert count == len(parameter_map), "Not all weights have been " \
                                            "transferred from ANN to SNN."

        factor = self._dt
        for layer in tqdm(self.snn.layers):
            if layer.__class__.__name__ == 'SpikeNormAdd':
                #print("Adapting shifts of "+layer.name)
                weights = self.parsed_model.get_layer(layer.name).get_weights()
                weights[1:] = [ arr*factor for arr in weights[1:] ]
                layer.set_weights(weights)
            elif layer.__class__.__name__ == 'SpikeNormReshape':
                layer.set_dt(factor)
            else:
                if hasattr(layer, 'norm'):
                    #print("Adapting shift of "+layer.name)
                    shift = keras.backend.get_value(layer.norm[1]) * factor
                    keras.backend.set_value(layer.norm[1], shift)
                if hasattr(layer, 'bias'):
                    # Adjust biases to time resolution of simulator.
                    #print("Adapting bias of "+layer.name)
                    bias = keras.backend.get_value(layer.bias) * factor
                    keras.backend.set_value(layer.bias, bias)
                    if self.config.getboolean('cell', 'bias_relaxation'):
                        keras.backend.set_value(
                            layer.b0, keras.backend.get_value(layer.bias))

    def compile(self):

        self.snn = keras.models.Model(
            self._input_images,
            self._spiking_layers[self.parsed_model.layers[-1].name])
        self.snn.compile('sgd', 'categorical_crossentropy', ['accuracy'])

        # Tensorflow 2 lists all variables as weights, including our state
        # variables (membrane potential etc). So a simple
        # snn.set_weights(parsed_model.get_weights()) does not work any more.
        # Need to extract the actual weights here.

        parameter_map = {remove_name_counter(p.name): v for p, v in
                         zip(self.parsed_model.weights,
                             self.parsed_model.get_weights())}
        count = 0
        for p in self.snn.weights:
            name = remove_name_counter(p.name)
            if name in parameter_map:
                keras.backend.set_value(p, parameter_map[name])
                count += 1
        assert count == len(parameter_map), "Not all weights have been " \
                                            "transferred from ANN to SNN."

        for layer in self.snn.layers:
            if hasattr(layer, 'bias'):
                # Adjust biases to time resolution of simulator.
                bias = keras.backend.get_value(layer.bias) * self._dt
                keras.backend.set_value(layer.bias, bias)
                if self.config.getboolean('cell', 'bias_relaxation'):
                    keras.backend.set_value(
                        layer.b0, keras.backend.get_value(layer.bias))


    def simulate_RNet_compare(self, x=None, y_parsed=None):
        num_detections = self.parsed_model.output_shape[-2]
        input_b_l = x * self._dt
        num_timesteps = self._get_timestep_at_spikecount(input_b_l)
        output_b_l_t = np.zeros((self.batch_size, num_detections, 4+self.num_classes))
        err = [None]*num_timesteps

        self._input_spikecount = 0
        for sim_step_int in tqdm(range(num_timesteps)):
            sim_step = (sim_step_int + 1) * self._dt
            self.set_time(sim_step)

            out_spikes = self.snn.predict_on_batch(input_b_l)
            output_b_l_t += (out_spikes[0] > 0)
            out = np.expand_dims(output_b_l_t/sim_step, 0)
            errs = np.abs(out-y_parsed)
            err[sim_step_int] = [
                [np.average(errs[:,:,:,:4]), np.amax(errs[:,:,:,:4])],
                [np.average(errs[:,:,:,4:]), np.amax(errs[:,:,:,4:])]
            ]
                
        return out, np.array(err), errs

  
    def analyze_model(self, x, y=None, layers_to_check=[], duration=None, previous_out = None, 
        return_the_rate=False, no_display=False, ignore_transient=False, transient_dur=None):
        x *= self._dt

        if duration is None:
            num_timesteps = self._get_timestep_at_spikecount(x)
        else:
            num_timesteps = int(duration / self._dt)
        
        if previous_out is None:
            output_b_l_t = [np.zeros(layer.output_shape) for layer in self.snn.layers if layer.name in layers_to_check] 
        else: 
            output_b_l_t = previous_out[0]
            if output_b_l_t is None:
                output_b_l_t = [np.zeros(layer.output_shape) for layer in self.snn.layers if layer.name in layers_to_check]
            t0 = previous_out[1]

        loss = [None]*num_timesteps 

        tf.keras.backend.clear_session() 
        model = tf.keras.Model(
            inputs = self.snn.input, 
            outputs = [layer.output for layer in self.snn.layers if layer.name in layers_to_check]
        ) 
        if not ignore_transient: transient_dur=0.0
        elif not transient_dur: transient_dur=float(len(model.layers))     
        
        for sim_step_int in tqdm(range(num_timesteps), disable=no_display):
            sim_step = (sim_step_int + 1) * self._dt + t0
            self.set_time(sim_step)
            out_spikes = model.predict_on_batch(x) 
            if sim_step>transient_dur: output_b_l_t = [acc+(spikes>0) for acc,spikes in zip(output_b_l_t,out_spikes)]
            T = max(1, sim_step-transient_dur)
            if y is not None:
                loss[sim_step_int] = self.snn.loss(y, output_b_l_t[-1]/float(T)).numpy()

        
        if return_the_rate:
            return [(acc/sim_step).astype('float32') for acc in output_b_l_t]
        else:
            return (output_b_l_t, sim_step, np.array(loss), transient_dur)



    def predict(self, x, ignore_transient=False, transient_dur=None):
        """ x --> Input to the network. """
        input_b_l = x * self._dt # Input scaled to the chosen dt
        num_timesteps = self._get_timestep_at_spikecount(input_b_l)
        output_b_l_t = np.zeros(self.snn.layers[-1].output_shape)

        if not ignore_transient: transient_dur=0.0
        elif not transient_dur: transient_dur=float(len(model.layers))

        for sim_step_int in range(num_timesteps): # Computation of the spikes in each timestep
            sim_step = (sim_step_int + 1) * self._dt
            self.set_time(sim_step)

            out_spikes = self.snn.predict(input_b_l)
            if sim_step>transient_dur: output_b_l_t += out_spikes[0] > 0 # Accumulation of the generated spikes

        T = max(1, sim_step-transient_dur)
        return output_b_l_t / T  
        """ Return the spiking rate """

    def predict_in_phases(self, x, output_b_l_t=None, t0=0.0, duration=None, ignore_transient=False, transient_dur=None):
        """ x --> Input to the network. """
        input_b_l = x * self._dt # Input scaled to the chosen dt
        if duration is None: num_timesteps = self._get_timestep_at_spikecount(x)
        else:                num_timesteps = int(duration / self._dt)
        if output_b_l_t is None: output_b_l_t = np.zeros(self.snn.layers[-1].output_shape, dtype='float32')

        if not ignore_transient: transient_dur=0.0
        elif not transient_dur: transient_dur=float(len(self.snn.layers))
        display('[{} --> {}]'.format(t0, t0+duration))

        for sim_step_int in tqdm(range(num_timesteps)): # Computation of the spikes in each timestep
            sim_step = float(sim_step_int + 1) * self._dt + t0
            self.set_time(sim_step)

            out_spikes = self.snn.predict(input_b_l)
            if sim_step>transient_dur: output_b_l_t += out_spikes[0] > 0 # Accumulation of the generated spikes

        T = max(1.0, sim_step-transient_dur)
        return output_b_l_t, T, sim_step
        """ Return the spiking rate """

            


    def simulate(self, **kwargs):

        from snntoolbox.utils.utils import echo
        from snntoolbox.simulation.utils import get_layer_synaptic_operations

        input_b_l = kwargs[str('x_b_l')] * self._dt

        # Optionally stop simulation of current batch when number of input
        # spikes exceeds a given limit.
        num_timesteps = self._get_timestep_at_spikecount(input_b_l)

        output_b_l_t = np.zeros((self.batch_size, self.num_classes,
                                 self._num_timesteps))

        print("Current accuracy of batch:")

        # Loop through simulation time.
        self._input_spikecount = 0
        for sim_step_int in range(num_timesteps):
            sim_step = (sim_step_int + 1) * self._dt
            self.set_time(sim_step)

            # Generate new input in case it changes with each simulation step.
            if self._poisson_input:
                input_b_l = self.get_poisson_frame_batch(kwargs[str('x_b_l')])
            elif self._is_aedat_input:
                input_b_l = kwargs[str('dvs_gen')].next_eventframe_batch()

            if self._is_early_stopping and np.count_nonzero(input_b_l) == 0:
                print("\nInput empty: Finishing simulation {} steps early."
                      "".format(self._num_timesteps - sim_step_int))
                break

            # Main step: Propagate input through network and record output
            # spikes.
            out_spikes = self.snn.predict_on_batch(input_b_l)

            # Add current spikes to previous spikes.
            if remove_classifier:  # Need to flatten output.
                output_b_l_t[:, :, sim_step_int] = np.argmax(np.reshape(
                    out_spikes > 0, (out_spikes.shape[0], -1)), 1)
            else:
                output_b_l_t[:, :, sim_step_int] = out_spikes > 0

            # Record neuron variables.
            i = j = 0
            for layer in self.snn.layers:
                # Excludes Input, Flatten, Concatenate, etc:
                if hasattr(layer, 'spiketrain') \
                        and layer.spiketrain is not None:
                    spiketrains_b_l = keras.backend.get_value(layer.spiketrain)
                    if self.spiketrains_n_b_l_t is not None:
                        self.spiketrains_n_b_l_t[i][0][
                            Ellipsis, sim_step_int] = spiketrains_b_l
                    if self.synaptic_operations_b_t is not None:
                        self.synaptic_operations_b_t[:, sim_step_int] += \
                            get_layer_synaptic_operations(spiketrains_b_l,
                                                          self.fanout[i + 1])
                    if self.neuron_operations_b_t is not None:
                        self.neuron_operations_b_t[:, sim_step_int] += \
                            self.num_neurons_with_bias[i + 1]
                    i += 1
                if hasattr(layer, 'mem') and self.mem_n_b_l_t is not None:
                    self.mem_n_b_l_t[j][0][Ellipsis, sim_step_int] = \
                        keras.backend.get_value(layer.mem)
                    j += 1

            if 'input_b_l_t' in self._log_keys:
                self.input_b_l_t[Ellipsis, sim_step_int] = input_b_l
            if self._poisson_input or self._is_aedat_input:
                if self.synaptic_operations_b_t is not None:
                    self.synaptic_operations_b_t[:, sim_step_int] += \
                        get_layer_synaptic_operations(input_b_l,
                                                      self.fanout[0])
            else:
                if self.neuron_operations_b_t is not None:
                    if sim_step_int == 0:
                        self.neuron_operations_b_t[:, 0] += self.fanin[1] * \
                            self.num_neurons[1] * np.ones(self.batch_size) * 2

            spike_sums_b_l = np.sum(output_b_l_t, 2)
            undecided_b = np.sum(spike_sums_b_l, 1) == 0
            guesses_b = np.argmax(spike_sums_b_l, 1)
            none_class_b = -1 * np.ones(self.batch_size)
            clean_guesses_b = np.where(undecided_b, none_class_b, guesses_b)
            current_acc = np.mean(kwargs[str('truth_b')] == clean_guesses_b)
            if self.config.getint('output', 'verbose') > 0 \
                    and sim_step % 1 == 0:
                echo('{:.2%}_'.format(current_acc))
            else:
                sys.stdout.write('\r{:>7.2%}'.format(current_acc))
                sys.stdout.flush()

        if self._is_aedat_input:
            remaining_events = \
                kwargs[str('dvs_gen')].remaining_events_of_current_batch()
        elif self._poisson_input and self._num_poisson_events_per_sample > 0:
            remaining_events = self._num_poisson_events_per_sample - \
                self._input_spikecount
        else:
            remaining_events = 0
        if remaining_events > 0:
            print("\nSNN Toolbox WARNING: Simulation of current batch "
                  "finished, but {} input events were not processed. Consider "
                  "increasing the simulation time.".format(remaining_events))

        return np.cumsum(output_b_l_t, 2)

    def reset(self, sample_idx):

        for layer in self.snn.layers[1:]:  # Skip input layer
            layer.reset(sample_idx)

    def end_sim(self):
        pass

    def save(self, path, filename):

        filepath = str(os.path.join(path, filename + '.h5'))
        print("Saving model to {}...\n".format(filepath))
        self.snn.save(filepath, self.config.getboolean('output', 'overwrite'))

    def load(self, path, filename):

        from snntoolbox.simulation.backends.inisim.temporal_mean_rate_tensorflow \
            import custom_layers

        filepath = os.path.join(path, filename + '.h5')

        try:
            self.snn = keras.models.load_model(filepath, custom_layers)
        except KeyError:
            raise NotImplementedError(
                "Loading SNN for INIsim is not supported yet.")
            # Loading does not work anymore because the configparser object
            # needed by the custom layers is not stored when saving the model.
            # Could be implemented by overriding Keras' save / load methods,
            # but since converting even large Keras models from scratch is so
            # fast, there's really no need.

    def get_poisson_frame_batch(self, x_b_l):
        """Get a batch of Poisson input spikes.

        Parameters
        ----------

        x_b_l: ndarray
            The input frame. Shape: (`batch_size`, ``layer_shape``).

        Returns
        -------

        input_b_l: ndarray
            Array of Poisson input spikes, with same shape as ``x_b_l``.

        """

        if self._input_spikecount < self._num_poisson_events_per_sample \
                or self._num_poisson_events_per_sample < 0:
            spike_snapshot = np.random.random_sample(x_b_l.shape) \
                             * self.rescale_fac * np.max(x_b_l)
            input_b_l = (spike_snapshot <= np.abs(x_b_l)).astype('float32')
            self._input_spikecount += \
                int(np.count_nonzero(input_b_l) / self.batch_size)
            # For BinaryNets, with input that is not normalized and
            # not all positive, we stimulate with spikes of the same
            # size as the maximum activation, and the same sign as
            # the corresponding activation. Is there a better
            # solution?
            input_b_l *= np.max(x_b_l) * np.sign(x_b_l)
        else:  # No more input spikes if _input_spikecount exceeded limit.
            input_b_l = np.zeros(x_b_l.shape)

        return input_b_l

    def set_time(self, t):
        """Set the simulation time variable of all layers in the network.

        Parameters
        ----------

        t: float
            Current simulation time.
        """

        for layer in self.snn.layers[1:]:
            if layer.get_time() is not None:  # Has time attribute
                layer.set_time(np.float32(t))

    def set_spiketrain_stats_input(self):
        # Added this here because PyCharm complains about not all abstract
        # methods being implemented (even though this is not abstract).
        AbstractSNN.set_spiketrain_stats_input(self)

    def get_spiketrains_input(self):
        # Added this here because PyCharm complains about not all abstract
        # methods being implemented (even though this is not abstract).
        AbstractSNN.get_spiketrains_input(self)

    def scale_first_layer_parameters(self, t, input_b_l, tau=1):
        w, b = self.snn.layers[0].get_weights()
        alpha = (self._duration + tau) / (t + tau)
        beta = b + tau * (self._duration - t) / (t + tau) * w * input_b_l
        keras.backend.set_value(self.snn.layers[0].kernel, alpha * w)
        keras.backend.set_value(self.snn.layers[0].bias, beta)

    def _get_timestep_at_spikecount(self, x):
        """Compute timestep at which a given number of input spikes is reached.

        If the user hasn't set the ``max_num_input_spikes`` parameter in the
        config file, the simulation duration will not change.

        Otherwise, we compute the number of steps required to reach the desired
        number of spikes, which can be used to limit the simulation duration.

        Currently only works with input in the form of constant bias currents,
        not DVS or Poisson input.

        Only supports reset by subtraction for now.
        """

        max_spikecount = self.config.getint('input', 'max_num_input_spikes',
                                            fallback='')
        if max_spikecount == '':
            return self._num_timesteps

        if self._is_aedat_input or self._poisson_input or \
                self.config.get('cell', 'reset') != 'Reset by subtraction':
            # raise NotImplementedError
            return self._num_timesteps

        # Transform sample-wise to batch-wise spikecount limit.
        max_spikecount_norm = max_spikecount * self.batch_size

        x_accum = np.zeros_like(x)
        t = 0
        while True:
            x_accum += x
            # Neglect threshold here (always 1 in input layer)
            spikecount = np.sum(np.floor(x_accum))  # / v_thresh
            if spikecount > max_spikecount_norm:
                print(t)
                return min(t, self._num_timesteps)
            t += 1
