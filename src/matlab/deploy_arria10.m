% This script deploy a Neural Network in a Intel Arria 10 SoC, using 
% arria10soc_single bitstream.
%
% Required products and support packages to run this script.
% MATLAB
% Deep Learning HDL Toolbox
% Deep Learning Toolbox Converter for TensorFlow Models
% Deep Learning HDL Toolbox Support Package for Intel FPGA and SoC Devices

% https://mathworks.com/help/deep-learning-hdl/ug/get-started-with-deepl-learning-fpga-deployment-to-intel-arria10-soc.html#ImagePredictionUsingDIGITSDeployedToIntelArria10SoCExample-2

% matlab2025b -nodesktop -nosplash -nodesktop -r "deploy_arria10.m"

close all, clc, clear all

optimize_dlprocessor_to_nn = true;
precision = 'single'; % 'sinole' or 'int0'

addpath(genpath('boards'))

%% Load model 
model_path='saved_models/mnist_fc/model.h5'
net = importKerasNetwork(model_path);
info = analyzeNetwork(net)
% Classification output
correctClasses = string(0:9);
% Get original layers
layers = net.Layers;
newClassificationLayer = classificationLayer('Classes', categorical(correctClasses), 'Name', 'output');

% Replace output layer
layers(end) = newClassificationLayer;

% Assemply Network with modified output layer
net = assembleNetwork(layers);
info = analyzeNetwork(net)

interface = "Ethernet"

if strcmp(interface , "JTAG")
    % If multiple Quartus instances, MATLAB cannot find the correct applications
    % Try to override PATH and LD_LIBRARY_PATH from MATLAB:
    %     setenv('PATH', ['/opt/FPGA/Intel/intelFPGA/21.1std/quartus/bin:' getenv('PATH')]);
    %     setenv('LD_LIBRARY_PATH', '/opt/FPGA/Intel/intelFPGA/21.1std/quartus/linux64');
    % or from Linux terminal before running MATLAB:
    %     export PATH=/opt/FPGA/Intel/intelFPGA/21.1std/quartus/bin:$PATH
    %     export LD_LIBRARY_PATH=/opt/FPGA/Intel/intelFPGA/21.1std/quartus/linux64:$LD_LIBRARY_PATH
    setenv('LD_LIBRARY_PATH', '/opt/FPGA/Intel/intelFPGA/21.1std/quartus/linux64');
    setenv('PATH', ['/opt/FPGA/Intel/intelFPGA/21.1std/quartus/bin:' getenv('PATH')]);
    hTarget = dlhdl.Target('Intel','Interface','JTAG');
elseif strcmp(interface , "Ethernet")
    hTarget = dlhdl.Target('Intel','Interface','Ethernet','IPAddress', ...
                           '192.168.1.101','Username', 'root', ...
                           'Password', 'cyclonevsoc');
end
hW = dlhdl.Workflow('network', net, ...
                    'Bitstream', 'arria10soc_single', ...
                    'Target', hTarget);
dn = hW.compile;
hW.deploy

%% Load MNIST data
train_images_file = 'data/train-images.idx3-ubyte';
train_labels_file = 'data/train-labels.idx1-ubyte';
test_images_file  = 'data/t10k-images.idx3-ubyte';
test_labels_file  = 'data/t10k-labels.idx1-ubyte';

% Load data
train_data = loadMNISTImages(train_images_file);
train_labels = loadMNISTLabels(train_labels_file);
test_data = loadMNISTImages(test_images_file);
test_labels = loadMNISTLabels(test_labels_file);

% Reshape input and test data to 28x28
train_data = reshape(train_data, 28, 28, []);
test_data = reshape(test_data, 28, 28, []);

test_labels_cat = categorical(test_labels);


%% Inference
% numSamples = size(test_data, 3);
correct = 0;
numSamples=10
preds = [];
true_labels = [];
for i = 1:numSamples
    % NN input
    data = test_data(:,:,i);

    % Hardware prediction
    [prediction, speed] = hW.predict(data,'Profile','on');

    % Prediction index
    [~, pred_idx] = max(prediction);
    pred_idx = pred_idx - 1; % MATLAB starting index is 1

    pred_label = categorical(classNames(pred_idx));

    % Compare labels
    if pred_label == true_label
        correct = correct + 1;
    end
    preds = [preds pred_idx];
    true_labels = [true_labels true_label];

end

acc = correct / numSamples;
fprintf('Accuracy en hardware: %.2f%%\n', acc * 100);
