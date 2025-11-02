# from https://github.com/Alec-Wright/CoreAudioML

# import torch
import torch.nn as nn

# import CoreAudioML.miscfuncs as miscfuncs
# import math
# from contextlib import nullcontext

# def wrapperkwargs(func, kwargs):
#     return func(**kwargs)


def wrapperargs(func, args):
    return func(*args)


# A simple RNN class that consists of a single recurrent unit of type LSTM, GRU or Elman, followed by a fully connected
# layer


class SimpleRNN(nn.Module):
    def __init__(self, input_size=1, output_size=1, unit_type="LSTM", hidden_size=32, skip=1, bias_fl=True, num_layers=1):
        super(SimpleRNN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        # Create dictionary of possible block types
        self.rec = wrapperargs(getattr(nn, unit_type), [input_size, hidden_size, num_layers])
        self.lin = nn.Linear(hidden_size, output_size, bias=bias_fl)
        self.bias_fl = bias_fl
        self.skip = skip
        self.save_state = True
        self.hidden = None

    def forward(self, x):
        if self.skip:
            # save the residual for the skip connection
            res = x[:, :, 0 : self.skip]
            x, self.hidden = self.rec(x, self.hidden)
            return self.lin(x) + res
        else:
            x, self.hidden = self.rec(x, self.hidden)
            return self.lin(x)

    # detach hidden state, this resets gradient tracking on the hidden state
    def detach_hidden(self):
        if self.hidden.__class__ == tuple:
            self.hidden = tuple([h.clone().detach() for h in self.hidden])
        else:
            self.hidden = self.hidden.clone().detach()

    # changes the hidden state to None, causing pytorch to create an all-zero hidden state when the rec unit is called
    def reset_hidden(self):
        self.hidden = None

    # This functions saves the model and all its paraemters to a json file, so it can be loaded by a JUCE plugin
    # def save_model(self, file_name, direc=""):
    #     if direc:
    #         miscfuncs.dir_check(direc)
    #     model_data = {
    #         "model_data": {
    #             "model": "SimpleRNN",
    #             "input_size": self.rec.input_size,
    #             "skip": self.skip,
    #             "output_size": self.lin.out_features,
    #             "unit_type": self.rec._get_name(),
    #             "num_layers": self.rec.num_layers,
    #             "hidden_size": self.rec.hidden_size,
    #             "bias_fl": self.bias_fl,
    #         }
    #     }

    #     if self.save_state:
    #         model_state = self.state_dict()
    #         for each in model_state:
    #             model_state[each] = model_state[each].tolist()
    #         model_data["state_dict"] = model_state

    #     miscfuncs.json_save(model_data, file_name, direc)

    # train_epoch runs one epoch of training
    # def train_epoch(self, input_data, target_data, loss_fcn, optim, bs, init_len=200, up_fr=1000):
    #     # shuffle the segments at the start of the epoch
    #     shuffle = torch.randperm(input_data.shape[1])

    #     # Iterate over the batches
    #     ep_loss = 0
    #     for batch_i in range(math.ceil(shuffle.shape[0] / bs)):
    #         # Load batch of shuffled segments
    #         input_batch = input_data[:, shuffle[batch_i * bs : (batch_i + 1) * bs], :]
    #         target_batch = target_data[:, shuffle[batch_i * bs : (batch_i + 1) * bs], :]

    #         # Initialise network hidden state by processing some samples then zero the gradient buffers
    #         self(input_batch[0:init_len, :, :])
    #         self.zero_grad()

    #         # Choose the starting index for processing the rest of the batch sequence, in chunks of args.up_fr
    #         start_i = init_len
    #         batch_loss = 0
    #         # Iterate over the remaining samples in the mini batch
    #         for k in range(math.ceil((input_batch.shape[0] - init_len) / up_fr)):
    #             # Process input batch with neural network
    #             output = self(input_batch[start_i : start_i + up_fr, :, :])

    #             # Calculate loss and update network parameters
    #             loss = loss_fcn(output, target_batch[start_i : start_i + up_fr, :, :])
    #             loss.backward()
    #             optim.step()

    #             # Set the network hidden state, to detach it from the computation graph
    #             self.detach_hidden()
    #             self.zero_grad()

    #             # Update the start index for the next iteration and add the loss to the batch_loss total
    #             start_i += up_fr
    #             batch_loss += loss

    #         # Add the average batch loss to the epoch loss and reset the hidden states to zeros
    #         ep_loss += batch_loss / (k + 1)
    #         self.reset_hidden()
    #     return ep_loss / (batch_i + 1)

    # only proc processes a the input data and calculates the loss, optionally grad can be tracked or not
    # def process_data(self, input_data, target_data, loss_fcn, chunk, grad=False):
    #     with torch.no_grad() if not grad else nullcontext():
    #         output = torch.empty_like(target_data)
    #         for l in range(int(output.size()[0] / chunk)):
    #             output[l * chunk : (l + 1) * chunk] = self(input_data[l * chunk : (l + 1) * chunk])
    #             self.detach_hidden()
    #         # If the data set doesn't divide evenly into the chunk length, process the remainder
    #         if not (output.size()[0] / chunk).is_integer():
    #             output[(l + 1) * chunk : -1] = self(input_data[(l + 1) * chunk : -1])
    #         self.reset_hidden()
    #         loss = loss_fcn(output, target_data)
    #     return output, loss
